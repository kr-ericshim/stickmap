"""Tempo evidence must survive an unrepresentative intro and dense fills."""
import numpy as np
import pytest

from groovemap.metre import Signature
from groovemap.pulse import _evidence, _follow, _resolve_residue, build_pulse


@pytest.mark.parametrize('opening_weight', [.15, 1.])
def test_off_tempo_intro_does_not_lock_dense_pulse(opening_weight):
    reference = .283
    opening = np.arange(.1, 5, .326)
    # Dense subdivisions make it easy for an incorrect grid to collect nearby
    # events indefinitely, despite the strong reference pulse underneath.
    steady = np.arange(12, 320, reference / 4)
    weights = np.where(np.arange(len(steady)) % 4 == 0, 1., .35)
    times = np.r_[opening, steady]
    # The loud variant also requires recovery after an initially plausible
    # wrong tempo: selecting a confident starting window alone cannot fix it.
    grid = np.asarray(_follow(times, np.r_[np.full(len(opening), opening_weight), weights], reference, 320))
    assert np.all(np.diff(grid) > 0)
    for start in range(0, 320, 40):
        gaps = np.diff(grid)[(grid[:-1] >= start) & (grid[:-1] < start + 40)]
        assert np.median(gaps) == pytest.approx(reference, rel=.015)


@pytest.mark.parametrize('direction', [-1, 1])
def test_follow_preserves_ten_percent_live_drift(direction):
    reference = .3
    # Begin off reference and smoothly pass through it to the opposite side.
    gaps = reference * (1 + direction * np.linspace(-.1, .1, 600))
    times = np.r_[.1, .1 + np.cumsum(gaps)]
    grid = np.asarray(_follow(times, np.ones(len(times)), reference, times[-1]))
    error = np.min(abs(times[:, None] - grid), axis=1)
    assert np.quantile(error, .95) < .02 * reference
    for start, stop in ((10, 40), (times[-1] - 40, times[-1] - 10)):
        expected = np.median(np.diff(times)[(times[:-1] >= start) & (times[:-1] < stop)])
        actual = np.median(np.diff(grid)[(grid[:-1] >= start) & (grid[:-1] < stop)])
        assert actual == pytest.approx(expected, rel=.01)


def test_dense_sixteenth_hats_with_misleading_opening_choose_eighth_residue():
    period, duration, phase = .3, 96, .12
    activations = np.zeros((duration * 100, 5))
    attacks = np.arange(phase, duration - .1, period / 2)
    for i, time in enumerate(attacks):
        frame = round(time * 100)
        activations[frame, 3] = .45  # hats on every sixteenth
        # The opening accents the wrong residue. Thereafter kick/snare and
        # cymbals establish eighths, while hats keep both residues viable.
        if (i % 2 == 1) if time < 6 else (i % 2 == 0):
            activations[frame, 0 if i // 2 % 7 in (0, 2, 4) else 1] = .85
            activations[frame, 4] = .65
    _, times, weights = _evidence(activations, attacks)
    provisional = np.asarray(_follow(times, weights, period, duration))
    inside = provisional[(provisional >= 12) & (provisional <= 90)]
    error = abs((inside - phase + period / 2) % period - period / 2)
    assert np.median(error) > .45 * period  # reproduces the initial wrong residue

    result = build_pulse(activations, attacks, [], [], Signature.parse('7/8'))
    grid = np.asarray(result.times)
    inside = grid[(grid >= 12) & (grid <= 90)]
    error = abs((inside - phase + period / 2) % period - period / 2)
    assert np.quantile(error, .95) < .01
    assert result.period == pytest.approx(period, rel=.01)
    assert np.all(np.diff(grid) > 0)


@pytest.mark.parametrize('fraction', [.25, .5, .75])
def test_residue_candidates_preserve_live_tempo_curve(fraction):
    gaps = np.linspace(.27, .33, 240)
    grid = np.r_[.1, .1 + np.cumsum(gaps)].tolist()
    expected = np.asarray(grid) + fraction * np.r_[gaps, gaps[-1]]
    activations = np.zeros((int((expected[-1] + 1) * 100), 5))
    activations[np.rint(expected * 100).astype(int), :2] = .8
    activations[np.rint(expected * 100).astype(int), 4] = .7
    resolved = _resolve_residue(grid, activations)
    assert resolved == pytest.approx(expected)
    assert len(resolved) == len(grid)
    assert np.all(np.diff(resolved) > 0)
    assert np.diff(resolved)[0] == pytest.approx(.27, abs=.001)
    assert np.diff(resolved)[-1] == pytest.approx(.33, abs=.001)


@pytest.mark.parametrize('pattern', ['equal', 'hats_only', 'one_loud_section'])
def test_residue_requires_sustained_instrument_agreement(pattern):
    grid = np.arange(.1, 65, .3).tolist()
    activations = np.zeros((6600, 5))
    on = np.rint(np.asarray(grid) * 100).astype(int)
    off = np.rint((np.asarray(grid) + .15) * 100).astype(int)
    if pattern == 'equal':
        activations[on] = .5
        activations[off] = .5
    elif pattern == 'hats_only':
        activations[off, 3] = 1
    else:
        activations[on] = .1
        activations[off[off < 1600]] = 1
    assert _resolve_residue(grid, activations) is grid
