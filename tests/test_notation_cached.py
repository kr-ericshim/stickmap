"""Read-only integration regression against the song cache, not Round 1's fixture."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from notation_metric import cached_inputs, measure, rescore
from stickmap import library, pulse
from stickmap.musicxml import to_musicxml
from stickmap.pipeline import Options
from stickmap.score import Hit


@pytest.fixture(scope="module")
def tony_cache():
    song = Path(__file__).resolve().parents[1] / "songs/tony-taylor-jr-bp-deadline-tour-tokyo-band-jam-clagz2kQWds"
    if not (song / "attacks.htdemucs.npy").exists():
        pytest.skip("Read-only Tony integration needs the local song cache")
    opts = Options(**library.load_options(song), pdf=False, preview=False)
    opts.triplets = "auto"
    hits, activations, attacks = cached_inputs(song, opts)
    return hits, json.loads((song / "beats.json").read_text()), opts, activations, attacks


def test_current_tony_cache_phase_is_stable_under_missing_hits_and_jitter(tony_cache, monkeypatch):
    hits, beats, opts, activations, attacks = tony_cache
    original_hits = list(hits)
    captured = {}
    refine = pulse.refine_display_times

    def record(grid, onsets, attacks, **kwargs):
        captured.update(grid=grid, onsets=onsets, attacks=attacks, kwargs=kwargs)
        return refine(grid, onsets, attacks, **kwargs)

    monkeypatch.setattr(pulse, "refine_display_times", record)
    score = rescore(hits, beats, opts, activations, attacks, "Tony cached regression")
    bar = score.bars[73]
    assert not any(h.inst == "kick" and h.time == 109.35 for h in hits)
    assert len(bar.notes) == 22
    assert bar.subdivisions == [4] * 4
    assert measure(score, to_musicxml(score), hits)["bars"][73]["timing_mae_ms"] < 13
    assert hits == original_hits
    assert score.merged_strokes == 0
    assert sorted((n.raw_time, n.inst) for b in score.bars for n in b.notes) == sorted((h.time, h.inst) for h in hits)

    base = np.asarray(refine(captured["grid"], captured["onsets"], attacks, **captured["kwargs"]))
    window = (base >= 108.8) & (base <= 110.31)
    variants = [hits + [Hit(109.35, "kick", 1)]]
    variants += [hits[:i] + hits[i + 1:] for i, h in enumerate(hits) if 108.8 < h.time < 110.3]
    variants += [[replace(h, time=h.time + delta) if 108.8 < h.time < 110.3 else h for h in hits] for delta in (-0.01, 0.01)]
    variants += [hits[:i] + [replace(h, time=h.time + delta)] + hits[i + 1:]
                 for i, h in enumerate(hits) if 108.8 < h.time < 110.3 for delta in (-0.01, 0.01)]
    for changed in variants:
        result = np.asarray(refine(captured["grid"], [h.time for h in changed], attacks, **captured["kwargs"]))
        assert max(abs(result[window] - base[window])) < 0.02
    # Also exercise the complete quantizer with the formerly decisive kick.
    added = rescore(variants[0], beats, opts, activations, attacks, "Added kick")
    assert added.bars[73].subdivisions == [4] * 4
    assert added.bars[73].beat_times == pytest.approx(bar.beat_times, abs=0.001)
    assert len(added.bars[73].notes) == 23
