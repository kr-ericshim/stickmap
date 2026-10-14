"""Old feature/dense caches cannot bypass the whole-song test seal."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import data
import splits
import train
import vault


@pytest.fixture
def training_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(vault, "CONFIG", tmp_path / "vault.json")
    monkeypatch.setattr(vault, "MANIFEST", tmp_path / "manifest.jsonl")
    vault.init(100.)
    rows = [dict(video=v, at=at, slug=f"cache-{v}", channel="drumshin", status="read")
            for v, at in [("old00000001", 90.), ("new00000001", 110.)]]
    vault.MANIFEST.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


@pytest.mark.parametrize("fold", ["0", "1", "2", "all"])
@pytest.mark.parametrize("objects", [False, True])
def test_split_excludes_seal_from_training_and_validation(training_seal, fold, objects):
    songs = [SimpleNamespace(**r) for r in training_seal] if objects else training_seal
    training, held = splits.split_songs(songs, fold)
    get = lambda s: s.slug if objects else s["slug"]
    assert {get(s) for s in training + held} == {training_seal[0]["slug"]}


def test_vault_unknown_channel_excluded_before_fold_validation(training_seal):
    training_seal[1]["channel"] = "future-channel"
    training, held = splits.split_songs(training_seal, "all")
    assert training == training_seal[:1] and held == []


def test_data_worker_refuses_vault_before_any_feature_or_truth_access(training_seal, monkeypatch):
    monkeypatch.setattr(data, "load", lambda *args: pytest.fail("vault truth was read"))
    assert data.one(training_seal[1]) is None


@pytest.mark.parametrize("dense", [False, True])
def test_direct_song_constructor_refuses_legacy_sealed_cache(training_seal, tmp_path, dense):
    folder = tmp_path / "cache"
    folder.mkdir()
    # No video_id in an old cache; the manifest and legacy slug still identify it.
    meta = {k: training_seal[1][k] for k in ("slug", "channel")}
    (folder / "truth.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="sealed vault song"):
        train.Song(folder, None, "cpu", dense=dense)
    assert not (folder / "teacher0.npy").exists()


def test_cached_song_list_never_constructs_a_vault_song(training_seal, tmp_path, monkeypatch):
    cache = tmp_path / "ft"
    cache.mkdir()
    monkeypatch.setattr(train, "CACHE", cache)
    monkeypatch.setattr(train, "original_model", lambda device: SimpleNamespace(eval=lambda: None))
    seen = []
    def song(folder, *args, **kw):
        seen.append(folder.name)
        assert folder.name == training_seal[0]["slug"]
        return folder.name
    monkeypatch.setattr(train, "Song", song)
    for row in training_seal:
        folder = cache / row["slug"]
        folder.mkdir()
        (folder / "truth.json").write_text(json.dumps({k: row[k] for k in ("slug", "channel")}))
    assert train.load_songs("cpu", dense=True) == [training_seal[0]["slug"]]
    assert seen == [training_seal[0]["slug"]]


def test_explicit_id_also_blocks_old_cache_without_manifest_match(training_seal, tmp_path):
    vault.add("ab-cd_efghi")
    folder = tmp_path / "legacy"
    folder.mkdir()
    (folder / "truth.json").write_text(json.dumps(dict(slug="unmatched-title-ab-cd_efghi", channel="drumshin")))
    with pytest.raises(ValueError, match="sealed vault song"):
        train.Song(folder, None, "cpu")
