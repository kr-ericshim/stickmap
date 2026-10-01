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


def _period(body: np.ndarray, times: np.ndarray, weights: np.ndarray, signature: Signature) -> tuple[float, float]:
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
        alignment, _ = _comb(times[keep] - start / FPS, weights[keep], periods)
        ac = _autocorrelation(segment)
        lags = periods * signature.numerator * FPS
        repetition = sum(np.interp(lags * m, np.arange(len(ac)), ac, right=0) / m for m in (1, 2, 3)) / (1 + 1/2 + 1/3)
        scores.append(alignment * 0.65 + repetition * 0.8)
        repeats.append(repetition)
    if not scores:
        raise ValueError("Not enough onset evidence to rebuild a pulse")
    # The strongest sustained mode wins, with no preference for a tracker that
    # may have latched onto an unrelated pulse.
    combined = np.mean(scores, axis=0)
    best = int(np.argmax(combined))
    return float(periods[best]), float(np.mean(repeats, axis=0)[best])


def _follow(times: np.ndarray, weights: np.ndarray, period: float, duration: float) -> list[float]:
    """Re-fit phase and slope over four pulses, coasting through sparse passages."""
    keep = times < min(duration, max(4, 16 * period))
    candidates = np.linspace(period * 0.80, period * 1.20, 401)
    alignment, phase = _comb(times[keep], weights[keep], candidates)
    best = int(np.argmax(alignment))
    period, start = float(candidates[best]), float(phase[best])
    if start > period / 2:
        start -= period
    pulse = [start]
    while pulse[-1] < duration + period:
        predicted = pulse[-1] + period
        keep = (times >= predicted - 2 * period) & (times <= predicted + 2 * period)
        local, weight = times[keep], weights[keep]
        indices = np.rint((local - predicted) / period)
        residual = local - (predicted + indices * period)
        use = np.abs(residual) < period * 0.22
        adjustment = 0.0
        if use.sum() >= 2 and weight[use].sum() > 0:
            adjustment = float(np.average(residual[use], weights=weight[use])) * 0.5
            if len(np.unique(indices[use])) >= 3:
                slope = float(np.polyfit(indices[use], local[use], 1, w=np.sqrt(weight[use]))[0])
                period = 0.85 * period + 0.15 * np.clip(slope, period * 0.92, period * 1.08)
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
        return beats[::round(1 / level)].tolist()
    result = [float(a + (b - a) * j / level) for a, b in zip(beats, beats[1:]) for j in range(round(level))]
    return result + [float(beats[-1])]


def _grid_alignment(grid: Sequence[float], times: np.ndarray, weights: np.ndarray) -> float:
    inside = (times >= grid[0]) & (times <= grid[-1])
    if not inside.any():
        return 0.0
    positions = np.interp(times[inside], grid, np.arange(len(grid)))
    residual = positions - np.rint(positions)
    return float(np.average(np.exp(-0.5 * (residual / 0.12) ** 2), weights=weights[inside]))


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
) -> Pulse:
    body, times, weights = _evidence(activations, attacks)
    period, repetition = _period(body, times, weights, signature)
    duration = len(body) / FPS
    rebuilt = _follow(times, weights, period, duration)
    tracked = _tracker_grid(beats, period)
    alignment = _grid_alignment(rebuilt, times, weights)
    compatible = tracked is not None and _grid_alignment(tracked, times, weights) >= max(0.55, alignment - 0.08)
    grid = tracked if compatible else rebuilt
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
