"""Recover a stated metre from cached evidence, without running a detector.

Repetition chooses the counting level; onset alignment chooses its phase. A
local fit then follows live tempo, rather than stretching a single metronome
across the recording. This is evidence for a grid, not proof of a downbeat.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, find_peaks

from .metre import Signature

FPS = 100


def onset_events(times: Sequence[float], width: float = 0.015) -> np.ndarray:
    """One temporal vote per unison, without merging the instrumental strokes."""
    groups: list[list[float]] = []
    for time in sorted(times):
        if groups and time - groups[-1][0] <= width + 1e-9:
            groups[-1].append(time)
        else:
            groups.append([time])
    return np.asarray([np.median(group) for group in groups], dtype=float)


def refine_display_times(
    grid: Sequence[float], onsets: Sequence[float], attacks: Sequence[float] = (), *,
    triplets: bool = True, unit_ticks: int = 24,
) -> list[float]:
    """Bounded local phase/period fit for notation, preserving the counting level.

    Dense neighbouring attacks must agree with one lattice and with its IOIs.
    Sparse beats cannot fit their own phase. Regular, onset-supported waveform
    sixteenths supply stable votes across detector thresholds. Other patterns
    use detected events with a waveform veto. Alternating events check the fit;
    this is internal consistency, not independent beat ground truth.
    The final smooth, bounded displacement changes no raw onset timestamps.
    """
    original = np.asarray(grid, dtype=float)
    events = onset_events(onsets)
    if len(events) < 6 or len(original) < 3:
        return original.tolist()
    waveform = onset_events(attacks)
    shifts = np.zeros(len(original))
    weights = np.zeros(len(original))
    subdivisions = [max(1, unit_ticks // 6), max(1, unit_ticks // 3)]
    if triplets and unit_ticks >= 12:
        subdivisions += [unit_ticks // 4]
        if unit_ticks == 24:
            subdivisions += [3]
    for i, anchor in enumerate(original):
        period = original[min(i + 1, len(original) - 1)] - original[max(0, i - 1)]
        period /= 2 if 0 < i < len(original) - 1 else 1
        local = events[(events >= anchor - period * 0.85) & (events <= anchor + period * 0.85)]
        acoustic = waveform[(waveform >= anchor - period * 0.85) & (waveform <= anchor + period * 0.85)]
        # Thresholding a single drum must not change which alternating events
        # train the phase fit. A dense, onset-supported attack sequence provides
        # the temporal votes even when a detector misses one of those attacks.
        if len(acoustic) >= 6 and len(local) >= 4 and np.ptp(acoustic) >= period * 0.8:
            near = np.min(np.abs(local[:, None] - acoustic), axis=1)
            gaps = np.diff(acoustic) / (period / max(1, unit_ticks // 6))
            # Only a sustained sixteenth run earns this replacement. Denser
            # chops, ternary runs and mixed halves retain their original fit.
            regular = np.mean(np.abs(gaps - 1) <= 0.25) >= 0.8
            if regular and np.mean(near <= 0.025) >= 0.8:
                local = acoustic
        if len(local) < 6 or np.ptp(local) < period * 0.8:
            continue
        # Work relative to the tracker's local interpolation, so genuine tempo
        # changes remain in place instead of being flattened to a metronome.
        x = np.interp(local, original, np.arange(len(original))) - i
        train, test = x[::2], x[1::2]
        candidates = []
        limit = min(0.125, 0.05 / period)
        for g in sorted(set(subdivisions)):
            for scale in (0.98, 1.0, 1.02):
                phases = np.linspace(-limit, limit, 51)
                residual = train[None, :] - (np.rint((train[None, :] - phases[:, None]) * g / scale) * scale / g + phases[:, None])
                j = int(np.argmin(np.mean(np.abs(residual), axis=1) + 0.08 * np.abs(phases)))
                phase = float(phases[j] + np.median(residual[j]))
                phase = float(np.clip(phase, -limit, limit))
                fitted = np.rint((x - phase) * g / scale) * scale / g + phase
                err = float(np.mean(np.abs(x - fitted)))
                ioi = np.diff(x)
                gap_err = float(np.mean(np.abs(ioi - np.rint(ioi * g / scale) * scale / g)))
                complexity = 0 if g == unit_ticks // 6 else (0.004 if g in (3, 6) else 0.007)
                cost = (err + 0.3 * gap_err + 0.08 * abs(phase) + 0.1 * abs(scale - 1)) * period + complexity
                candidates.append((cost, err, g, scale, phase))
        _, err, g, scale, phase = min(candidates)
        fixed = float(np.mean(np.abs(x - np.rint(x * g) / g)))
        held = float(np.mean(np.abs(test - (np.rint((test - phase) * g / scale) * scale / g + phase))))
        held_fixed = float(np.mean(np.abs(test - np.rint(test * g) / g)))
        if err * period > 0.015 or (fixed - err) * period < 0.004 or held > held_fixed + 0.002 / period:
            continue
        if len(waveform):
            check = waveform[(waveform >= local[0]) & (waveform <= local[-1])]
            if len(check) >= 4:
                y = np.interp(check, original, np.arange(len(original))) - i
                before = np.mean(np.abs(y - np.rint(y * g) / g))
                after = np.mean(np.abs(y - (np.rint((y - phase) * g / scale) * scale / g + phase)))
                if after > before + 0.002 / period:
                    continue
        shifts[i] = phase * period
        weights[i] = min(1.0, len(local) / 8)
    # The kernel shares phase only with immediate neighbours. An unsupported
    # region returns to the tracker, instead of accumulating unbounded drift.
    smooth = gaussian_filter1d(shifts * weights, 0.65, mode="nearest")
    support = gaussian_filter1d(weights, 0.65, mode="nearest")
    smooth /= np.maximum(support, 0.65)
    gaps = np.diff(original)
    for i in range(1, len(smooth)):
        smooth[i] = np.clip(smooth[i], smooth[i - 1] - gaps[i - 1] * 0.08, smooth[i - 1] + gaps[i - 1] * 0.08)
    for i in range(len(smooth) - 2, -1, -1):
        smooth[i] = np.clip(smooth[i], smooth[i + 1] - gaps[i] * 0.08, smooth[i + 1] + gaps[i] * 0.08)
    return (original + smooth).tolist()


@dataclass
class Pulse:
    times: list[float]  # denominator-note boundaries, extended to complete bars
    source: str
    period: float
    repetition: float
    alignment: float
    tracker_compatible: bool


def _evidence(activations: np.ndarray, attacks: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Give body drums equal votes; hats alone do not establish bar repetition."""
    a = np.asarray(activations, dtype=float)
    if a.ndim != 2 or a.shape[1] < 5 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError("Pulse building needs finite activations with five drum columns")
    body = a[:, :3].copy()
    body /= np.maximum(np.percentile(body, 95, axis=0), 0.1)
    body = np.minimum(body, 3)
    envelope = np.max(a, axis=1)
    peaks, _ = find_peaks(envelope, height=0.08, distance=4, prominence=0.03)
    times = peaks / FPS
    weights = envelope[peaks]
    attack = np.asarray(attacks, dtype=float)
    if attack.ndim != 1 or not np.isfinite(attack).all():
        raise ValueError("Attack times must be a finite one-dimensional array")
    attack = attack[(attack >= 0) & (attack < len(a) / FPS)]
    # Waveform attacks add temporal precision without outvoting drum identity.
    near = np.abs(attack[:, None] - times).min(axis=1) if len(times) else np.full(len(attack), 1.0)
    extra = attack[near > 0.025]
    times, weights = np.r_[times, extra], np.r_[weights, np.full(len(extra), 0.18)]
    order = np.argsort(times)
    return body, times[order], weights[order]


def _autocorrelation(body: np.ndarray) -> np.ndarray:
    centered = body - body.mean(axis=0)
    corr = sum(correlate(c, c, mode="full", method="fft")[len(c) - 1:] for c in centered.T)
    return corr / max(float(corr[0]), 1e-9)


def _comb(times: np.ndarray, weights: np.ndarray, periods: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Circular concentration is independent of the unknown first downbeat."""
    angles = 2 * np.pi * times[None, :] / periods[:, None]
    vector = (np.exp(1j * angles) * weights).sum(axis=1) / max(weights.sum(), 1e-9)
    return np.abs(vector), (np.angle(vector) / (2 * np.pi) % 1) * periods


def _period(body: np.ndarray, times: np.ndarray, weights: np.ndarray, signature: Signature, beats: Sequence[float] = (),
            signatures: Sequence[Signature] = ()) -> tuple[float, float]:
    """Compare local combs so drift cannot cancel alignment across a whole song."""
    # Quarter-note equivalents 60..250 BPM. The scale correction extends this
    # range when the musician wants a different counting level.
    low, high = 60 / 250 * 4 / signature.denominator, 4 / signature.denominator
    periods = np.arange(low, high + 0.0005, 0.0005)
    scores, repeats = [], []
    window = min(len(body), max(800, int(signature.numerator * high * FPS * 3)))
    for start in range(0, max(1, len(body) - window // 2), max(1, window // 2)):
        segment = body[start:start + window]
        if len(segment) < min(window, 200):
            continue
        keep = (times >= start / FPS) & (times < (start + len(segment)) / FPS)
        if keep.sum() < 4:
            continue
        alignment = np.max([
            _comb(times[keep] - start / FPS, weights[keep], periods / division)[0] * discount
            for division, discount in ((1, 1.0), (2, 0.98), (4, 0.94))
        ], axis=0)
        ac = _autocorrelation(segment)
        repetition = np.max([
            sum(np.interp(periods * sig.numerator * FPS * m, np.arange(len(ac)), ac, right=0) / m
                for m in (1, 2, 3)) / (1 + 1/2 + 1/3)
            for sig in signatures or (signature,)
        ], axis=0)
        scores.append(alignment * 0.65 + repetition * 0.8)
        repeats.append(repetition)
    if not scores:
        raise ValueError("Not enough onset evidence to rebuild a pulse")
    # Aggregate local evidence before applying weak counting-level priors.
    combined = np.mean(scores, axis=0)
    # A weak prior resolves subdivision aliases without overriding a clear
    # repeating drum pattern (including a genuinely wrong tracker).
    gaps = np.diff(np.asarray(beats, dtype=float))
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if len(gaps):
        ratios = gaps[:, None] / periods[None, :]
        levels = np.array([1/4, 1/3, 1/2, 1, 2, 3, 4])
        distance = np.min(np.abs(np.log(ratios[:, :, None] / levels)), axis=2)
        combined += 0.06 * np.mean(np.exp(-0.5 * (distance / 0.08) ** 2), axis=0)
    quarter_bpm = 240 / (signature.denominator * periods)
    combined -= 0.22 * np.log(quarter_bpm / 110) ** 2
    combined -= 0.4 * (np.maximum(0, np.log(quarter_bpm / 220)) ** 2
                       + np.maximum(0, np.log(50 / quarter_bpm)) ** 2)
    best = int(np.argmax(combined))
    return float(periods[best]), float(np.mean(repeats, axis=0)[best])


def _follow(times: np.ndarray, weights: np.ndarray, period: float, duration: float) -> list[float]:
    """Follow phase and tempo while retaining the selected counting level.

    The fit includes only complete pulse cells: truncating the outside cells
    biases the regression slope downwards, cumulatively accelerating the grid.
    Reference bounds prevent that feedback from changing subdivision levels.
    """
    reference = period
    if not len(times):
        raise ValueError("Not enough onset evidence to follow a pulse")
    # Quiet bleed before the drums enter must not choose the starting tempo.
    # Use a relative threshold so quiet recordings retain their evidence.
    strong = weights >= 0.5 * np.percentile(weights, 90)
    beginning = times[np.flatnonzero(strong)[0]]
    keep = (times >= beginning) & (times < min(duration, beginning + max(4, 16 * period)))
    candidates = np.linspace(period * 0.80, period * 1.20, 401)
    fits = [_comb(times[keep], weights[keep], candidates / d) for d in (1, 2, 4)]
    alignment = np.max([fit[0] * discount for fit, discount in zip(fits, (1, .98, .94))], axis=0)
    best = int(np.argmax(alignment))
    period = float(candidates[best])
    division = int(np.argmax([fit[0][best] * discount for fit, discount in zip(fits, (1, .98, .94))]))
    unit = period / (1, 2, 4)[division]
    phase = float(fits[division][1][best])
    # The short-window fundamental selects a provisional residue. build_pulse
    # checks the harmonic alternatives against longer instrumental evidence.
    candidates_phase = phase + np.arange((1, 2, 4)[division]) * unit
    fundamental = float(fits[0][1][best])
    distance = np.abs((candidates_phase - fundamental + period / 2) % period - period / 2)
    start = float(candidates_phase[np.argmin(distance)])
    if start > period / 2:
        start -= period
    pulse = [start]
    while pulse[-1] < duration + period:
        predicted = pulse[-1] + period
        keep = (times >= predicted - 8 * period) & (times <= predicted + 8 * period)
        local, weight = times[keep], weights[keep]
        # Dense subdivisions can sustain an incorrect local regression for
        # minutes. Reconsider the song-level period only when it explains this
        # window substantially better; coherent live drift keeps its own tempo.
        if len(pulse) % 8 == 1 and len(local) >= 4 and abs(period / reference - 1) > 0.03:
            scores = np.max([
                _comb(local - predicted, weight, np.array([period, reference]) / d)[0] * discount
                for d, discount in ((1, 1.0), (2, .98), (4, .94))
            ], axis=0)
            if scores[1] > scores[0] + 0.1:
                period = reference
                predicted = pulse[-1] + period
        indices = np.rint((local - predicted) / period)
        residual = local - (predicted + indices * period)
        use = (np.abs(residual) < period * 0.22) & (np.abs(indices) <= 6)
        adjustment = 0.0
        if use.sum() >= 2 and weight[use].sum() > 0:
            adjustment = float(np.average(residual[use], weights=weight[use])) * 0.5
            if len(np.unique(indices[use])) >= 3:
                slope = float(np.polyfit(indices[use], local[use], 1, w=np.sqrt(weight[use]))[0])
                period = float(np.clip(0.85 * period + 0.15 * np.clip(slope, period * 0.92, period * 1.08),
                                       reference * 0.8, reference * 1.2))
        pulse.append(float(predicted + adjustment))
    return pulse


def _tracker_grid(beats: Sequence[float], period: float) -> list[float] | None:
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 3 or not np.isfinite(beats).all() or np.any(np.diff(beats) <= 0):
        return None
    ratio = float(np.median(np.diff(beats)) / period)
    levels = (0.25, 0.5, 1, 2, 3, 4, 6)
    level = min(levels, key=lambda x: abs(ratio / x - 1))
    # A 0.60 s tracker is NOT compatible with 4 * 0.1486 s merely because
    # their averages are close. Phase evidence below must also agree.
    if abs(ratio / level - 1) > 0.06:
        return None
    if level < 1:
        result = beats[::round(1 / level)].tolist()
    else:
        # A tracker may alternate between one and two denominator notes.
        # Infill each gap separately; a median stride would preserve the jumps.
        steps = np.rint(np.diff(beats) / period).astype(int)
        if np.any(steps < 1) or np.any(steps > 6):
            return None
        result = [float(a + (b - a) * j / count)
                  for a, b, count in zip(beats, beats[1:], steps) for j in range(count)]
        result.append(float(beats[-1]))
    if len(result) < 3 or np.any(np.abs(np.diff(result) / period - 1) > 0.2):
        return None
    return result


def _grid_alignment(grid: Sequence[float], times: np.ndarray, weights: np.ndarray) -> float:
    inside = (times >= grid[0]) & (times <= grid[-1])
    if not inside.any():
        return 0.0
    positions = np.interp(times[inside], grid, np.arange(len(grid)))
    residual = positions - np.rint(positions)
    return float(np.average(np.exp(-0.5 * (residual / 0.12) ** 2), weights=weights[inside]))


def _resolve_residue(grid: list[float], activations: np.ndarray) -> list[float]:
    """Change harmonic residue only with sustained, multi-instrument support.

    The follower's onset envelope loses drum identity and can lock to alternate
    subdivisions during a busy opening. Compare four residues along its *local*
    tempo curve, without refitting the tempo or changing the counting level.
    Ambiguous/syncopated evidence keeps the provisional grid.
    """
    g = np.asarray(grid)
    if len(g) < 3:
        return grid
    a = np.asarray(activations)[:, :5]
    fractions = np.arange(4) / 4
    queries = g[:-1][None, :] + fractions[:, None] * np.diff(g)
    # All candidates use the same complete cells, never padded endpoint data.
    valid = (queries[0] >= .02) & (queries[-1] <= (len(a) - 1) / FPS - .02)
    queries = queries[:, valid]
    if queries.shape[1] < 32:
        return grid
    frames = queries * FPS
    indices = np.floor(frames).astype(int)[..., None] + np.arange(-2, 4)
    nearby = np.abs(indices - frames[..., None]) <= 2 + 1e-9
    values = np.max(np.where(nearby[..., None], a[indices.clip(0, len(a) - 1)], 0), axis=2)
    means = values.mean(axis=1)
    # Equalise active instruments, without amplifying almost absent channels.
    # Dense hats and fill toms carry less metrical evidence than body/crashes.
    scale = np.maximum(means.max(axis=0), .03)
    importance = np.array([1., 1., .5, .25, 1.]) / scale
    scores = means @ importance
    best = int(np.argmax(scores))
    if best == 0 or scores[best] <= 1.10 * np.max(np.delete(scores, best)):
        return grid
    supports = (means[best] > 1.05 * means[0]) & (means[best] >= .03)
    if supports.sum() < 3 or not supports[:3].any():
        return grid
    # A loud fill or intro cannot overrule the rest of the recording. Require
    # agreement in at least two thirds of eligible, non-overlapping 16 s spans.
    wins = []
    for start in np.arange(0, len(a) / FPS, 16):
        inside = (queries[0] >= start) & (queries[-1] < start + 16)
        if inside.sum() < 16:
            continue
        local = values[:, inside].mean(axis=1) @ importance
        wins.append(local[best] > 1.05 * local[0])
    if len(wins) < 2 or np.mean(wins) < 2 / 3:
        return grid
    # Interpolation preserves live drift and monotonicity. Extrapolate only the
    # final boundary, which build_pulse subsequently pads to complete bars.
    gaps = np.r_[np.diff(g), g[-1] - g[-2]]
    return (g + fractions[best] * gaps).tolist()


def _bar_phase(grid: list[float], body: np.ndarray, downbeats: Sequence[float], numerator: int, compatible: bool) -> int:
    positions = np.arange(len(grid))
    # Repeated kicks/crashes suggest a first beat; downbeats only vote when
    # the tracker's pulse itself has survived the compatibility test.
    kick = np.interp(np.asarray(grid) * FPS, np.arange(len(body)), gaussian_filter1d(body[:, 0], 1))
    votes = np.array([np.mean(kick[positions % numerator == i]) if np.any(positions % numerator == i) else 0 for i in range(numerator)])
    if compatible and len(downbeats):
        indices = np.rint(np.interp(downbeats, grid, positions)).astype(int)
        counts = np.bincount(indices % numerator, minlength=numerator)
        if counts.max() >= 0.65 * len(indices):
            votes += counts / max(1, counts.max()) * (votes.max() + 0.1)
    return int(np.argmax(votes))


def build_pulse(
    activations: np.ndarray, attacks: Sequence[float], beats: Sequence[float], downbeats: Sequence[float],
    signature: Signature, *, beat_shift: float = 0, tempo_scale: float = 1,
    signatures: Sequence[Signature] = (),
) -> Pulse:
    body, times, weights = _evidence(activations, attacks)
    period, repetition = _period(body, times, weights, signature, beats, signatures)
    duration = len(body) / FPS
    rebuilt = _follow(times, weights, period, duration)
    tracked = _tracker_grid(beats, period)
    alignment = _grid_alignment(rebuilt, times, weights)
    compatible = tracked is not None and _grid_alignment(tracked, times, weights) >= max(0.55, alignment - 0.08)
    grid = tracked if compatible else rebuilt
    resolved = _resolve_residue(grid, activations)
    if resolved is not grid:
        # Tracker downbeats refer to its old residue and cannot vote here.
        compatible = False
    grid = resolved
    alignment = _grid_alignment(grid, times, weights)
    # Extend the usable tracker through the beginning/end of the recording.
    while grid[0] > 0:
        grid.insert(0, 2 * grid[0] - grid[1])
    while grid[-1] < duration:
        grid.append(2 * grid[-1] - grid[-2])
    if tempo_scale == 2:
        grid = np.interp(np.arange(0, len(grid) - 1 + 0.1, 0.5), np.arange(len(grid)), grid).tolist()
    elif tempo_scale == 0.5:
        grid = grid[::2]
    phase = _bar_phase(grid, body, downbeats, signature.numerator, compatible) + beat_shift
    # Whole bars cover the evidence. A negative first boundary is a padded
    # pickup, and avoids silently throwing away the first notes after a shift.
    first = phase + np.floor(-phase / signature.numerator) * signature.numerator
    last = phase + np.ceil((len(grid) - 1 - phase) / signature.numerator) * signature.numerator
    indices = np.arange(first, last + 0.1)
    output = np.interp(indices, np.arange(len(grid)), grid)
    output[indices < 0] = grid[0] + indices[indices < 0] * (grid[1] - grid[0])
    output[indices > len(grid) - 1] = grid[-1] + (indices[indices > len(grid) - 1] - len(grid) + 1) * (grid[-1] - grid[-2])
    # Remove complete padding bars without removing a real pickup.
    n = signature.numerator
    while len(output) > n + 1 and output[n] <= times[0] + 0.03:
        output = output[n:]
    while len(output) > n + 1 and output[-n-1] > times[-1] + period / 2:
        output = output[:-n]
    return Pulse(output.tolist(), "tracker" if compatible else "rebuilt", float(np.median(np.diff(output))), repetition, alignment, compatible)


def mixed_bars(grid: Sequence[float], activations: np.ndarray,
               signatures: Sequence[Signature], attacks: Sequence[float] = ()) -> tuple[list[float], list[int]]:
    """Semi-Markov decode: pulse boundary + previous length are the states.

    Local body-drum repetition supplies section evidence; adjacent patterns and
    kick/crash accents choose boundaries. Scores accrue per pulse, so shorter
    bars do not win merely by having more boundaries. A transition cost favours
    sustained sections. No tracker downbeat is assumed to be a true bar line.
    """
    from scipy.ndimage import maximum_filter1d, uniform_filter1d

    lengths = [s.numerator for s in signatures]
    largest = max(lengths)
    original = np.asarray(grid, dtype=float)
    period = float(np.median(np.diff(original)))
    body, onsets, _ = _evidence(activations, attacks)
    if not len(onsets):
        raise ValueError("Not enough onset evidence to decode bars")
    first = max(0, int(np.searchsorted(original, onsets[0] - 0.03)) - 1)
    target = min(len(original) - 1, int(np.searchsorted(original, onsets[-1] + 1e-6)))
    # Permit a padded pickup and complete the last bar without inventing evidence.
    indices = np.arange(first - largest, target + largest + 1)
    times = np.interp(indices, np.arange(len(original)), original)
    left, right = indices < 0, indices >= len(original)
    times[left] = original[0] + indices[left] * (original[1] - original[0])
    times[right] = original[-1] + (indices[right] - len(original) + 1) * (original[-1] - original[-2])
    end = target - first + largest
    samples = np.interp(np.arange((len(times) - 1) * 4 + 1) / 4, np.arange(len(times)), times) * FPS
    smooth = gaussian_filter1d(body, 1, axis=0)
    # Pool each subdivision cell: point samples can miss a narrow onset and
    # turn a few milliseconds of performance jitter into a different pattern.
    pooled = maximum_filter1d(smooth, max(3, round(period * FPS / 4) | 1), axis=0)
    features = np.stack([np.interp(samples, np.arange(len(body)), c, left=0, right=0) for c in pooled.T], axis=1)
    cymbal = np.asarray(activations)[:, 4]
    cymbal = np.minimum(cymbal / max(float(np.percentile(cymbal, 95)), 0.1), 3)
    accent = np.interp(times * FPS, np.arange(len(body)), smooth[:, 0] + 0.5 * gaussian_filter1d(cymbal, 1), left=0, right=0)
    accent /= max(float(np.percentile(accent, 90)), 0.1)
    accent = np.minimum(accent, 2)
    # A symmetric ~16 second window averages fills while retaining section changes.
    window = max(largest * 8, round(16 / period * 4))
    mean = uniform_filter1d(features, window, axis=0, mode="constant")
    centered = features - mean
    local = []
    for n in lengths:
        lag = n * 4
        delayed = np.zeros_like(centered)
        delayed[lag:] = centered[:-lag]
        cross = uniform_filter1d(np.sum(centered * delayed, axis=1), window, mode="constant")
        power = uniform_filter1d(np.sum(centered ** 2, axis=1), window, mode="constant")
        other = uniform_filter1d(np.sum(delayed ** 2, axis=1), window, mode="constant")
        local.append(cross / np.maximum(np.sqrt(power * other), 1e-8))
    count = len(times)
    values = np.full((count, len(lengths)), -np.inf)
    previous = np.full((count, len(lengths)), -1, dtype=int)
    starts = np.full((count, len(lengths)), -1, dtype=int)
    for i in range(end):
        for k, n in enumerate(lengths):
            j = i + n
            if j >= count:
                continue
            center = min(len(features) - 1, (i + n // 2) * 4)
            reward = n * (3 * local[k][center] + 0.12 * accent[i])
            best, parent = -np.inf, -1
            if i <= largest and j > largest:
                # Initial phase is free; a partial first bar is a pickup.
                best = reward - max(0, largest - i) * (3 * local[k][center] + 0.12 * accent[i])
            for q, m in enumerate(lengths):
                if not np.isfinite(values[i, q]):
                    continue
                similarity = 0.0
                if i >= m:
                    # Compare bar shapes on a common normalized phase axis.
                    phase = np.arange(28) / 28
                    a = np.stack([np.interp((i + phase * n) * 4, np.arange(len(features)), c) for c in features.T])
                    b = np.stack([np.interp((i - m + phase * m) * 4, np.arange(len(features)), c) for c in features.T])
                    a -= a.mean(axis=1, keepdims=True)
                    b -= b.mean(axis=1, keepdims=True)
                    similarity = float(np.sum(a * b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-8))
                score = values[i, q] + reward + 0.3 * n * similarity - (2.5 if m != n else 0)
                if score > best:
                    best, parent = score, q
            if best > values[j, k]:
                values[j, k], previous[j, k], starts[j, k] = best, parent, i
    # Do not reward evidence beyond the last onset just to fill a longer bar.
    endings = [(values[j, k] - (j - end) * (3 * local[k][min((end - 1) * 4, len(features) - 1)] + 0.12), j, k)
               for j in range(end, min(count, end + largest)) for k in range(len(lengths))]
    _, j, k = max(endings)
    stop = j
    decoded = []
    while k >= 0:
        decoded.append(lengths[k])
        i = starts[j, k]
        if i < 0:
            raise ValueError("No complete mixed-meter path")
        k, j = previous[j, k], i
    return times[j:stop + 1].tolist(), decoded[::-1]
