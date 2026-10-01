import shutil

import pytest

from groovemap import fingerprint


@pytest.fixture
def package(tmp_path, monkeypatch):
    """A copy of the package source to edit; fingerprints read it instead of the real one."""
    copy = tmp_path / "groovemap"
    shutil.copytree(fingerprint.PACKAGE, copy, ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(fingerprint, "PACKAGE", copy)
    fingerprint._module.cache_clear()
    yield copy
    fingerprint._module.cache_clear()


def edit(path, old, new):
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def test_comments_and_docstrings_do_not_count(package):
    before = fingerprint._compute()
    edit(package / "models.py", "def track_beats(", "# a comment\ndef track_beats(")
    edit(package / "quantize.py", '"""', '"""Reworded. ', )
    assert fingerprint._compute() == before


def test_threshold_change_invalidates_what_reads_thresholds_only(package):
    before = fingerprint._compute()
    edit(package / "models.py", "DEFAULT_THRESHOLDS = (0.22,", "DEFAULT_THRESHOLDS = (0.2,")
    after = fingerprint._compute()
    changed = {kind for kind in before["producers"] if before["producers"][kind] != after["producers"][kind]}
    assert changed == {"snare"}  # the snare model picks strokes with the thresholds
    assert after["score"] != before["score"]


def test_a_remade_stem_invalidates_what_was_made_from_it(package):
    before = fingerprint._compute()
    edit(package / "models.py", "def separate_drums(audio_wav", "def separate_drums(stem_source_wav")
    after = fingerprint._compute()
    stamps = lambda prints: {name: fingerprint.cache_stamp(name, prints) for name in (
        "drums.htdemucs.wav", "beats.json", "activations.mix.npy", "activations.drums-htdemucs.npy",
        "snare.htdemucs.npy", "attacks.htdemucs.npy")}
    old, new = stamps(before), stamps(after)
    assert {name for name in old if old[name] != new[name]} == {
        "drums.htdemucs.wav", "activations.drums-htdemucs.npy", "snare.htdemucs.npy", "attacks.htdemucs.npy"}


def test_new_fine_tuned_weights_remake_the_stem_activations_only(package):
    before = fingerprint._compute()
    weights = package / fingerprint.PRODUCER_FILES["stem_onsets"][0]
    weights.write_bytes(weights.read_bytes() + b"retrained")
    after = fingerprint._compute()
    stamps = lambda prints: {name: fingerprint.cache_stamp(name, prints) for name in (
        "drums.htdemucs.wav", "activations.mix.npy", "activations.drums-htdemucs.npy", "snare.htdemucs.npy")}
    old, new = stamps(before), stamps(after)
    assert {name for name in old if old[name] != new[name]} == {"activations.drums-htdemucs.npy", "snare.htdemucs.npy"}
    assert after["score"] != before["score"]  # a score kept after its caches were cleared is dated too


def test_source_edited_after_start_is_not_vouched_for(package, monkeypatch):
    monkeypatch.setattr(fingerprint, "AT_START", fingerprint._compute())
    assert fingerprint.snapshot(check=True) == fingerprint.AT_START
    edit(package / "quantize.py", "_MARGIN = ", "_MARGIN = 0 + ")
    assert fingerprint.snapshot(check=True) is None
    assert fingerprint.cache_stamp("beats.json", None) is None
