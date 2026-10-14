"""Conservative metre hints are reviewable metadata, never automatic edits."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from .metre import Signature
from .pulse import FPS, _autocorrelation, _evidence, _period
from .score import Score


def suggest_signature(score: Score, activations: np.ndarray, attacks: Sequence[float], beats: Sequence[float], downbeats: Sequence[float]) -> dict | None:
    if len(score.bars) < 4 or any(bar.beats != 4 for bar in score.bars) or len(beats) < 8 or len(downbeats) < 4:
        return None
    positions = np.interp(downbeats, beats, np.arange(len(beats)))
    gaps = np.diff(positions)
    erratic = float(np.mean(np.abs(gaps - 4) > 0.3))
    if erratic < 0.5:
        return None
    body, times, weights = _evidence(activations, attacks)
    if len(times) < 12:
        return None
    bar_seconds = float(np.median([bar.beat_times[-1] - bar.start for bar in score.bars]))
    ac = _autocorrelation(body)
    current = float(np.interp(bar_seconds * FPS, np.arange(len(ac)), ac))
    hint = re.search(r"(?<!\d)((?:[1-9]|[12][0-9]|3[0-2])/(?:16|8|4))(?!\d)", score.title)
    candidates = [hint[1]] if hint and hint[1] != "4/4" else ["3/4", "5/4", "6/8", "7/8", "9/8", "12/8"]
    ranked = []
    for value in candidates:
        signature = Signature.parse(value)
        period, repetition = _period(body, times, weights, signature)
        length = period * signature.numerator
        # If the existing bar already repeats, another subdivision of that bar
        # is not evidence for replacing its signature.
        if abs(length / bar_seconds - 1) > 0.15:
            ranked.append((repetition, value, length))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    repetition, value, length = ranked[0]
    if repetition < (0.06 if hint else 0.2) or repetition - current < (0.04 if hint else 0.12):
        return None
    if not hint and len(ranked) > 1 and repetition - ranked[1][0] < 0.06:
        return None  # audio alone often cannot distinguish related counting levels
    reasons = ["erratic_downbeats", "bar_period_mismatch"]
    if hint:
        reasons.insert(0, "title_hint")
    return {"time_signature": value, "reasons": reasons, "suggested_bar_seconds": round(length, 4),
            "current_bar_seconds": round(bar_seconds, 4), "repetition": round(repetition, 4),
            "current_repetition": round(current, 4), "erratic_downbeat_fraction": round(erratic, 4)}
