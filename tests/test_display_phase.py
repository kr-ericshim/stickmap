import numpy as np
import pytest

from stickmap.pulse import refine_display_times
from stickmap.quantize import quantize
from stickmap.score import Hit, Score


def test_local_sixteenth_phase_drift_is_repaired_without_changing_raw_hits():
    grid = np.arange(25) * 0.375
    raw = [b * 0.375 + k * 0.375 / 4 for b in range(24) for k in range(4)]
    tracked = grid.copy()
    tracked[8:13] += 0.04
    display = refine_display_times(tracked, raw, raw, triplets=False)
    assert max(abs(np.asarray(display)[9:12] - grid[9:12])) < 0.012
    assert np.all(np.diff(display) > 0)
    assert max(abs(np.diff(display) / np.diff(tracked) - 1)) <= 0.080001
    hits = [Hit(t, "snare", 1) for t in raw]
    score = quantize(hits, tracked, grid[::4], attacks=raw, triplets="off")
    assert sorted(n.raw_time for b in score.bars for n in b.notes) == raw
    assert [b.subdivisions for b in score.bars[2:3]] == [[4] * 4]
    assert Score.from_json(score.to_json()) == score


def test_sparse_syncopation_cannot_move_the_beat_phase():
    grid = np.arange(17) * 0.5
    onsets = [b * 0.5 + 0.045 for b in range(16)]
    assert refine_display_times(grid, onsets) == grid.tolist()


def test_waveform_disagreement_vetoes_dense_phase_fit():
    grid = np.arange(17) * 0.5
    onsets = [b * 0.5 + k * 0.125 + 0.04 for b in range(16) for k in range(4)]
    attacks = [b * 0.5 + k * 0.125 for b in range(16) for k in range(4)]
    output = refine_display_times(grid, onsets, attacks)
    assert np.asarray(output)[2:-2] == pytest.approx(grid[2:-2])


def test_shifted_sextuplets_keep_all_strokes_and_true_thirty_seconds():
    grid = np.arange(17) * 0.375
    hits = [Hit(b * 0.375 + k * 0.375 / 6 - 0.018, "tom", 1) for b in range(4, 8) for k in range(6)]
    hits += [Hit(b * 0.375 + k * 0.375 / 8, "snare", 1) for b in range(10, 14) for k in range(8)]
    score = quantize(hits, grid, grid[::4])
    modes = [s for b in score.bars for s in b.subdivisions]
    assert modes[5:7] == [6, 6]
    assert modes[11:13] == [8, 8]
    assert sum(len(b.notes) for b in score.bars) == len(hits)
    assert score.merged_strokes == 0
