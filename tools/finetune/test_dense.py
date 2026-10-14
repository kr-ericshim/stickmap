"""Synthetic selection and acoustic-window tests; no detector or audio files."""
import sys
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dense import band_window, candidates, in_window, quality_reasons  # noqa: E402


def fixture(times=None, names=None):
    times = [1.0, 1.08, 1.16, 1.24] if times is None else times
    names = ["kick", "snare", "tom_pos_3", "tom_pos_7"] if names is None else names
    strip = [{"system": 0, "staff": {"spacing": 10, "reasons": []},
              "notes": [{"x": i * 20, "inst": name, "exportable": True}
                        for i, name in enumerate(names)]}]
    report = {"system": 0, "interval": [times[0], times[-1]], "times": times,
              "usable": False, "reasons": ["ambiguous alignment in dense attacks"],
              "after": {"within_40ms": 0.75}, "assigned_fraction": 0.8}
    return strip, {"systems": [report], "intervals": []}


def test_runs_and_unisons():
    strip, alignment = fixture([1, 1.14, 1.28, 1.42, 2, 2.08, 2.16, 2.24],
                               ["kick", "snare", "tom_pos_3", "snare"] * 2)
    # An extra class in one moment must not inflate the moment count.
    strip[0]["notes"].append({"x": 40, "inst": "kick", "exportable": True})
    selected, _ = candidates(strip, alignment, [])
    s = selected[0]
    assert [r["moments"] for r in s["runs"]] == [4, 4]
    assert len(s["labels"]) == 9
    assert s["fast_runs"] == 1
    assert sum(q["fast_run"] for q in s["labels"]) == 4
    assert all(q["tolerance_frames"] == 3 for q in s["labels"])


@pytest.mark.parametrize("times,names", [
    ([1, 1.08, 1.16], ["kick", "tom", "snare"]),  # too short
    ([1, 1.08, 1.16, 1.24], ["kick", "snare", "kick", "snare"]),  # no tom
    ([1, 1.08, 1.23, 1.31], ["kick", "tom", "tom", "snare"]),  # split
    ([1, 1.08, 1.16, 1.24], ["kick", "hihat", "tom", "snare"]),  # only 3 drum moments
])
def test_run_exclusions(times, names):
    strip, alignment = fixture(times, names)
    assert not candidates(strip, alignment, [])[0]


@pytest.mark.parametrize("reason", ["held-out agreement below 55% / 80%",
                                   "fitted attack agreement below 85% / 95%",
                                   "ambiguous alignment in dense attacks",
                                   "no-cursor alignment needs 70% / 90% held-out agreement"])
def test_timing_prefixes(reason):
    strip, alignment = fixture()
    alignment["systems"][0]["reasons"] = [reason]
    assert len(candidates(strip, alignment, [])[0]) == 1


@pytest.mark.parametrize("extra", ["weak staff-line support", "phase alias: shifted",
                                  "warp is not strictly monotone", "too few audio attacks"])
def test_non_timing_reason_excludes(extra):
    strip, alignment = fixture()
    alignment["systems"][0]["reasons"].append(extra)
    assert not candidates(strip, alignment, [])[0]


def test_timing_and_exportability_gates():
    strip, alignment = fixture()
    assert candidates(strip, alignment, [])[0]  # exact timing thresholds admitted
    for key, value in (("assigned_fraction", 0.799), ("usable", True), ("reasons", [])):
        altered = deepcopy(alignment)
        altered["systems"][0][key] = value
        assert not candidates(strip, altered, [])[0]
    alignment["systems"][0]["after"]["within_40ms"] = 0.749
    assert not candidates(strip, alignment, [])[0]
    alignment["systems"][0]["after"]["within_40ms"] = 0.75
    strip[0]["notes"][0]["exportable"] = False
    assert not candidates(strip, alignment, [])[0]


def test_whole_system_overlap_and_window_guard():
    strip, alignment = fixture([1, 1.08, 1.16, 1.24, 3],
                               ["kick", "snare", "tom", "snare", "kick"])
    # A sparse tail overlaps: the earlier dense run still cannot be used.
    assert not candidates(strip, alignment, [[2.9, 3.1]])[0]
    strip, alignment = fixture()
    assert not candidates(strip, alignment, [[1.27, 2]])[0]  # tolerance reaches strict
    alignment["systems"][0]["interval"] = [0.5, 1.24]
    assert not candidates(strip, alignment, [[0.6, 0.7]])[0]  # source interval overlap
    alignment["systems"][0]["interval"] = [1, 1.24]
    assert candidates(strip, alignment, [[1.3, 2]])[0]
    # An accepted system protects its span even without top-level intervals.
    accepted = deepcopy(alignment["systems"][0])
    accepted.update(system=1, usable=True, reasons=[], times=[1.2, 1.4])
    alignment["systems"].append(accepted)
    assert not candidates(strip, alignment, [])[0]


def test_exact_positive_window_and_quality_shortfall():
    assert in_window([1.004, 2], np.array([1.03]), 3).tolist() == [True, False]
    assert not in_window([1.004], np.array([1.031]), 3)[0]
    empty = {"labels": 0, "attack_hits": 0, "band_hits": None, "both_hits": None}
    stats = {"kick": empty, "snare": empty,
             "tom": {**empty, "labels": 8, "attack_hits": 5}}
    baseline = {c: {"attack_rate": 1, "band_rate": None, "both_rate": None}
                for c in stats}
    assert quality_reasons(stats, baseline)  # 3 excess misses, 37.5-point deficit
    stats["tom"]["attack_hits"] = 6
    assert not quality_reasons(stats, baseline)  # only two excess misses


def test_fast_boundary_and_sparse_unexportable_drums():
    strip, alignment = fixture([1, 1.09, 1.18, 1.27])
    assert candidates(strip, alignment, [])[0][0]["fast_runs"] == 0  # strictly <90 ms
    strip[0]["notes"].append({"x": 100, "inst": "crash", "exportable": False})
    alignment["systems"][0]["times"].append(2)
    assert candidates(strip, alignment, [])[0]  # withheld cymbals are harmless
    strip[0]["notes"][-1]["inst"] = "snare"
    assert not candidates(strip, alignment, [])[0]  # even outside the dense run


def test_band_peaks_and_cross_band_leakage():
    audio = {"t": np.array([0.969, 1.0, 1.031]),
             "low": np.array([1.0, 0.0, 1.0]),
             "low_peaks": np.array([1.0, 0.0, 1.0]),
             "mid": np.zeros(3), "mid_peaks": np.zeros(3)}
    assert not band_window(audio, [1], "kick", 3)[0]  # peaks just outside the window
    audio["low"][1] = audio["low_peaks"][1] = 1.0
    assert band_window(audio, [1], "kick", 3)[0]
    assert not band_window(audio, [1], "snare", 3)[0]
    audio["mid"][1] = audio["mid_peaks"][1] = 2.0
    assert not band_window(audio, [1], "kick", 3)[0]  # mid band dwarfs low band
    assert band_window(audio, [1], "snare", 3)[0]


@pytest.mark.parametrize('source', ['channel', 'cohort'])
def test_quality_excludes_held_channels_and_same_song_before_pooling(source):
    from dense import fold_quality, pool
    def stats(n, hits):
        s = pool([])
        s['tom'].update(labels=n, attack_hits=hits)
        return s
    # Candidate 5/8 passes an 80% baseline but fails a leaked 100% baseline.
    target = dict(slug='target', channel='drumshin', strict=stats(0, 0),
                  candidates=[dict(agreement=stats(8, 5))])
    anchor = dict(slug='anchor', channel='drumshin' if source == 'channel' else 'swartz',
                  strict=stats(25, 20), candidates=[])
    held = dict(slug='held-love-me-or-leave-me', channel='yeol', strict=stats(100, 100), candidates=[])
    excluded = dict(slug='other-love-me-or-leave-me', channel='drumshin',
                    strict=stats(100, 100), candidates=[])
    rows = [target, anchor, held, excluded]
    selected, scope = fold_quality(rows, '2')
    baseline = selected[0]['quality_reference']['tom']
    assert baseline['source'] == source and baseline['attack_rate'] == .8
    assert len(selected[0]['retained']) == 1
    assert scope['training_slugs'] == ['anchor', 'target']
    assert scope['excluded_same_song_slugs'] == [excluded['slug']]
    assert all(not s['retained'] and s['quality_reference'] is None for s in selected[2:])
    held['strict'] = excluded['strict'] = stats(1000, 0)
    changed, changed_scope = fold_quality(rows, '2')
    assert changed[0]['quality_reference'] == selected[0]['quality_reference']
    assert changed[0]['retained'] == selected[0]['retained']
    assert changed_scope['strict'] == scope['strict']


@pytest.mark.parametrize('identity', ['cache_video_id', 'cache_video', 'canonical_video_id', 'canonical_video'])
def test_main_excludes_legacy_sealed_identity_before_evidence_and_quality(tmp_path, monkeypatch, identity):
    import dense
    import vault
    out = tmp_path / 'harvest'
    songs, cache = out / 'songs', tmp_path / 'ft'
    out.mkdir()
    monkeypatch.setattr(vault, 'OUT', out)
    monkeypatch.setattr(vault, 'CONFIG', out / 'vault.json')
    monkeypatch.setattr(vault, 'MANIFEST', out / 'manifest.jsonl')
    (out / 'vault.json').write_text(json.dumps({
        'cutoff': 100, 'video_ids': ['sealed00001'], 'train_channels': ['drumshin']}))
    monkeypatch.setattr(dense, 'SONGS', songs)
    monkeypatch.setattr(dense, 'CACHE', cache)
    strip, alignment = fixture([1 + i * .08 for i in range(8)], ['tom_pos_3'] * 8)
    for slug, count in [('target', 0), ('anchor', 25), ('legacy-cache', 100)]:
        song, folder = songs / slug, cache / slug
        (song / 'truth').mkdir(parents=True)
        folder.mkdir(parents=True)
        meta = dict(channel='drumshin', events={'kick': [], 'snare': [], 'tom': [i * .01 for i in range(count)]},
                    intervals=[[0, .4]] if count else [])
        if slug == 'legacy-cache':
            source, key = identity.split('_', 1)
            if source == 'cache':
                meta[key] = 'sealed00001'
            else:
                (song / 'meta.json').write_text(json.dumps({key: 'sealed00001'}))
            # A sealed source must not even reach legend/notation parsing.
            (song / 'truth/legend.json').write_text('corrupt sealed legend')
            (folder / 'dense-foldall.json').write_text('{"labels":["stale"]}')
        else:
            (song / 'truth/legend.json').write_text('{"enabled":true}')
            (song / 'truth/strip.json').write_text(json.dumps(strip))
            (song / 'truth/alignment.json').write_text(json.dumps(alignment))
        (folder / 'truth.json').write_text(json.dumps(meta))
        np.save(folder / 'feat.npy', np.zeros((300, 1)))
    seen = []
    monkeypatch.setattr(dense, 'audio_evidence', lambda song: (seen.append(song.name), None))

    def evidence(labels, *args):
        stats = dense.pool([])
        n = len(labels)
        stats['tom'].update(labels=n, attack_hits=20 if n == 25 else 5 if n == 8 else n)
        return stats

    monkeypatch.setattr(dense, 'agreement', evidence)
    monkeypatch.setattr(sys, 'argv', ['dense.py', '--fold', 'all'])
    dense.main()
    target = json.loads((cache / 'target/dense-foldall.json').read_text())
    summary = json.loads((cache / 'dense-foldall.summary.json').read_text())
    assert seen == ['anchor', 'target']
    assert target['quality_reference']['tom']['attack_rate'] == .8
    assert len(target['labels']) == 8  # leaked 100/100 evidence would reject it
    assert summary['folds']['all']['training_slugs'] == ['anchor', 'target']
    assert summary['agreement']['strict']['tom']['labels'] == 25
    sealed = json.loads((cache / 'legacy-cache/dense-foldall.json').read_text())
    assert sealed['status'] == 'sealed_vault' and sealed['labels'] == []
    assert not sealed['training_eligible'] and sealed['quality_reference'] is None
