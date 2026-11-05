"""Strokes inside a fill that the peak picker passed over.

Through a fast fill the tom activation stays high from stroke to stroke, and
a picker that measures each peak against the last 100 ms finds nothing to
pick. A stroke is still there when the raw activation peaks, the drum stem
has an attack at that moment, no other hit explains it, and it joins a run of
toms that were picked.

Quiet snares between the toms are lost the same way: the snare rescue in
models.py keeps clear of every other hit, and in a fill there always is one.
Inside a fill only another snare rules a quiet one out, and the snare must take
more of the sound at that moment than it does elsewhere.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import find_peaks, stft

from .models import _RESCUE_ACTIVATION, _RESCUE_LOUDNESS, ADTOF_CLASSES, ADTOF_FPS, DEFAULT_THRESHOLDS, GHOST_RATIO, TOM_BLEND
from .score import Hit

_HOP = 0.005  # seconds between frames of the attack detector
_BANDS = ((40, 180), (180, 500), (500, 2000), (2000, 6000), (6000, 16000))  # Hz
_RUN_GAP = 0.3  # seconds: strokes this close belong to one fill
_NEAR_ATTACK = 0.02  # seconds between an activation peak and the attack it stands for
_NEAR_HIT = 3  # frames: a peak this close to a picked hit is that hit
# A fill, as the detector heard it: at least this many kick, snare or tom moments with
# a tom among them, at most this many frames apart and this many apart on the median.
_FILL = (4, 14, 10)
_FILL_EDGE = 7  # frames around a fill that still belong to it
_FILL_SHARE = 0.6  # a quiet snare in a fill takes at least this share of what the drums play


def attack_times(drums_wav: Path) -> np.ndarray:
    """Seconds at which something is struck in the drum stem, whatever the drum.

    Rising energy summed over five frequency bands, each scaled to its own
    level so that a tom is not drowned by the cymbals.
    """
    audio, sr = sf.read(drums_wav, dtype="float32", always_2d=True)
    if len(audio) < 1024:
        return np.empty(0)
    hop = max(1, round(sr * _HOP))
    freqs, times, spectrum = stft(audio.mean(axis=1), fs=sr, nperseg=1024, noverlap=1024 - hop, boundary="zeros", padded=True)
    magnitude = np.abs(spectrum)
    scale = np.maximum(np.percentile(magnitude, 75, axis=1, keepdims=True), max(float(magnitude.max()) * 1e-4, 1e-12))
    level = np.log1p(magnitude / scale)
    # Against the loudest of the neighbouring bins two frames back: a pitch that wobbles is not an attack.
    before = maximum_filter1d(level, size=3, axis=0)
    rise = np.pad(np.maximum(level[:, 2:] - before[:, :-2], 0), ((0, 0), (2, 0)))
    bands = np.array([rise[(freqs >= low) & (freqs < high)].mean(axis=0) for low, high in _BANDS])
    bands /= np.maximum(np.percentile(bands, 90, axis=1, keepdims=True), 1e-6)
    envelope = bands.mean(axis=0)
    floor = uniform_filter1d(envelope, size=round(0.2 / _HOP), mode="nearest")
    peaks, _ = find_peaks(envelope, distance=round(0.03 / _HOP), prominence=0.3)
    return times[peaks[(envelope[peaks] > 0.6) & (envelope[peaks] > floor[peaks] + 0.12)]]


def tom_runs(
    hits: list[Hit], activations: np.ndarray, mix: np.ndarray, attacks: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[Hit]:
    """The same hits, plus the toms that complete the fills."""
    picked = np.array([hit.time for hit in hits if hit.inst == "tom"])
    frames = min(len(activations), len(mix))
    if not len(picked) or not len(attacks) or frames < 5:
        return hits
    tom = ADTOF_CLASSES.index("tom")
    threshold = list(thresholds or DEFAULT_THRESHOLDS)[tom] * TOM_BLEND
    blend = (activations[:frames, tom] + mix[:frames, tom]) / 2
    peaks, _ = find_peaks(blend, height=threshold, prominence=threshold / 2, distance=4)
    taken = np.array([round(hit.time * ADTOF_FPS) for hit in hits])
    waiting = [
        int(f) for f in peaks
        if np.abs(taken - f).min() > _NEAR_HIT and np.abs(attacks - f / ADTOF_FPS).min() <= _NEAR_ATTACK + 1e-8
    ]
    added: list[int] = []
    while True:  # a found stroke can link the next one to the run
        joined = [f for f in waiting if np.abs(picked - f / ADTOF_FPS).min() <= _RUN_GAP + 1e-8]
        if not joined:
            break
        added += joined
        waiting = [f for f in waiting if f not in joined]
        picked = np.r_[picked, np.array(joined) / ADTOF_FPS]
    return sorted(hits + [Hit(f / ADTOF_FPS, "tom", float(blend[f])) for f in added], key=lambda hit: hit.time)


def _fills(hits: list[Hit]) -> list[tuple[int, int]]:
    """First and last frame of every fill among the hits."""
    moments = sorted({round(hit.time * ADTOF_FPS) for hit in hits if hit.inst in ("kick", "snare", "tom")})
    toms = np.array([round(hit.time * ADTOF_FPS) for hit in hits if hit.inst == "tom"])
    runs, run = [], []
    for f in moments:
        if run and f - run[-1] > _FILL[1]:
            runs.append(run)
            run = []
        run.append(f)
    runs.append(run)
    return [
        (run[0] - _FILL_EDGE, run[-1] + _FILL_EDGE) for run in runs
        if len(run) >= _FILL[0] and np.median(np.diff(run)) <= _FILL[2] and np.any((toms >= run[0]) & (toms <= run[-1]))
    ]


def fill_snares(hits: list[Hit], activations: np.ndarray, response: np.ndarray, attacks: np.ndarray) -> list[Hit]:
    """The same hits, plus the quiet snares inside fills (`response` as from dynamics.snare_response)."""
    fills = _fills(hits)
    frames = min(len(activations), len(response))
    if not fills or not len(attacks):
        return hits
    column = ADTOF_CLASSES.index("snare")
    taken = [round(hit.time * ADTOF_FPS) for hit in hits if hit.inst == "snare"]
    added = []
    for first, last in fills:
        for f in range(max(2, first), min(frames - 2, last + 1)):
            loudness, share = response[f]
            # A peak of the loudness, the first frame where it stays level, as in models._snares.
            if loudness < _RESCUE_LOUDNESS or loudness <= response[f - 1, 0] or loudness < response[f - 2:f + 3, 0].max():
                continue
            if share < _FILL_SHARE or activations[f - 2:f + 3, column].max() < _RESCUE_ACTIVATION:
                continue
            if any(abs(g - f) <= _NEAR_HIT for g in taken) or np.abs(attacks - f / ADTOF_FPS).min() > _NEAR_ATTACK + 1e-8:
                continue
            added.append(Hit(f / ADTOF_FPS, "snare", float(loudness), ghost=float(loudness) < GHOST_RATIO))
            taken.append(f)
    return sorted(hits + added, key=lambda hit: hit.time)
