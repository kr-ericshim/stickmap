"""Quantize an explicitly stated signature without the legacy quarter recount."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from math import isfinite
from statistics import median

import numpy as np

from .metre import Signature, parse_meter_changes, parse_signatures
from .pulse import build_pulse, mixed_bars, refine_display_times
from .quantize import _display_onsets, _half_plan, _strokes
from .score import DIVISIONS, INSTRUMENTS, Bar, Hit, Note, Score


def _unit_plan(group: list[tuple[Hit, float]], ticks: int, reading: bool, triplets: bool) -> tuple[list[int], int]:
    """Prefer the readable grid unless a fine grid saves a stroke or improves fit."""
    base = max(1, ticks // (12 if reading else 6))
    choices = [base]
    if not reading:
        choices += [ticks // 3]
        if triplets and ticks == 12:
            choices += [3]
    candidates = []
    for divisions in sorted(set(choices)):
        slots = [int(np.floor(f * divisions + 0.5)) * (ticks // divisions) for _, f in group]
        lost = len(slots) - len({(h.inst, t) for (h, _), t in zip(group, slots)})
        error = sum(abs(f - t / ticks) for (_, f), t in zip(group, slots))
        penalty = sum(0.08 for t in slots if t % (ticks // base))
        candidates.append((lost, error + penalty, divisions, slots))
    _, _, divisions, slots = min(candidates)
    return slots, divisions


def placed_bars(grid: Sequence[float], numerator: int, changes: Sequence[tuple[float, Signature]]) -> tuple[list[float], list[int]]:
    """Cut bars at user-placed changes; the pulse grid's own end bounds the music.

    A change snaps to the nearest pulse. If the bars before it do not fill the
    gap exactly, the remainder becomes one short bar, so the changed section
    still starts where the user heard it.
    """
    grid = [float(t) for t in grid]
    end, period = grid[-1], float(np.median(np.diff(grid)))

    def index(time: float) -> int:
        if time >= grid[-1]:
            return len(grid) - 1 + round((time - grid[-1]) / period)
        if time <= grid[0]:
            return round((time - grid[0]) / period)
        return int(round(float(np.interp(time, grid, np.arange(len(grid))))))

    position, n, lengths = 0, numerator, []
    for time, signature in changes:
        target = index(time)
        if target > position:
            full, rest = divmod(target - position, n)
            lengths += [n] * full + ([rest] if rest else [])
            position = target
        n = signature.numerator
    while not lengths or position < len(grid) - 1 and grid[position] < end - period / 2:
        lengths.append(n)
        position += n
    while len(grid) <= position:
        grid.append(2 * grid[-1] - grid[-2])
    return grid[:position + 1], lengths


def quantize_signature(
    hits: Sequence[Hit], beats: Sequence[float], downbeats: Sequence[float], *,
    activations: np.ndarray, attacks: Sequence[float], time_signature: str,
    title: str = "", subtitle: str = "", grid: int = 16, triplets: str = "auto",
    beat_shift: float = 0, tempo_scale: float = 1, meter_changes: Sequence[dict] = (),
) -> Score:
    signatures = parse_signatures(time_signature)
    changes = parse_meter_changes(list(meter_changes), time_signature)
    signature = signatures[0]
    if grid not in (8, 16) or triplets not in ("auto", "off"):
        raise ValueError("grid must be 8 or 16; triplets must be auto or off")
    if not isfinite(beat_shift) or (beat_shift * 4) % 1 or tempo_scale not in (0.5, 1, 2):
        raise ValueError("beat_shift uses steps of 0.25; tempo_scale must be 0.5, 1 or 2")
    if any(h.inst not in INSTRUMENTS or not isfinite(h.time) or not isfinite(h.strength) for h in hits):
        raise ValueError("Hits must have known instruments and finite times and strengths")
    pulse = build_pulse(activations, attacks, beats, downbeats, signature, beat_shift=0 if len(signatures) > 1 and not changes else beat_shift, tempo_scale=tempo_scale, signatures=signatures)
    times = pulse.times
    if changes:
        # The user's changes replace the automatic choice of bar lengths.
        times, lengths = placed_bars(times, signature.numerator, changes)
    elif len(signatures) > 1:
        times, lengths = mixed_bars(times, activations, signatures, attacks)
        # Preserve the existing explicit-signature shift unit: one denominator
        # pulse, with controls in steps of 0.25 of that pulse.
        shifted = np.arange(len(times)) + beat_shift
        moved = np.interp(shifted, np.arange(len(times)), times)
        moved[shifted < 0] = times[0] + shifted[shifted < 0] * (times[1] - times[0])
        moved[shifted > len(times) - 1] = times[-1] + (shifted[shifted > len(times) - 1] - len(times) + 1) * (times[-1] - times[-2])
        times = moved.tolist()
    else:
        lengths = [signature.numerator] * ((len(times) - 1) // signature.numerator)
    boundaries = np.r_[0, np.cumsum(lengths)].tolist()
    n, unit = signature.numerator, DIVISIONS * 4 // signature.denominator
    strokes = _strokes([h for h in hits if times[0] - pulse.period / 8 <= h.time < times[-1]])
    times = refine_display_times(times, [h.time for h in strokes], attacks, triplets=triplets == "auto", unit_ticks=unit)
    groups: list[list[tuple[Hit, float]]] = [[] for _ in times[:-1]]
    for hit, onset in zip(strokes, _display_onsets(strokes, times)):
        time = max(times[0], onset)
        i = min(len(groups) - 1, bisect_right(times, time) - 1)
        groups[i].append((hit, (time - times[i]) / (times[i + 1] - times[i])))
    if unit == DIVISIONS:
        # Reuse the tested straight/triplet planner, but never the 4/4 recount.
        plans = _half_plan(groups, times, [size for size in lengths for _ in range(size)], triplets == "auto", grid == 8)
        slots = [p.slots for p in plans]
        subdivisions = [p.modes[0] if p.modes[0] == p.modes[1] else list(p.modes) for p in plans]
    else:
        planned = [_unit_plan(g, unit, grid == 8, triplets == "auto") for g in groups]
        slots, subdivisions = [p[0] for p in planned], [p[1] for p in planned]
    bars = [Bar(
        index=index, beats=n, beat_times=times[i:i + n + 1], subdivisions=subdivisions[i:i + n],
        tempo=60 * n * 4 / signature.denominator / (times[i + n] - times[i]),
        denominator=signature.denominator, beat_groups=Signature(n, signature.denominator).groups,
    ) for index, (i, n) in enumerate(zip(boundaries, lengths))]
    strengths = {inst: float(np.percentile([h.strength for h in hits if h.inst == inst], 90))
                 for inst in INSTRUMENTS if any(h.inst == inst for h in hits)}
    from .dynamics import ghost_states

    ghosts = ghost_states(strokes)
    snapped: dict[tuple[int, str], Hit] = {}
    merged = 0
    for i, (group, positions) in enumerate(zip(groups, slots)):
        for (hit, _), position in zip(group, positions):
            if unit == DIVISIONS and grid == 8 and plans[i].modes[0 if position < 12 else 1] == 4 and position % 12:
                continue
            tick = i * unit + position
            if tick >= len(groups) * unit:
                tick = len(groups) * unit - 3
            key = tick, hit.inst
            merged += int(key in snapped)
            if key not in snapped or hit.strength > snapped[key].strength:
                snapped[key] = hit
    for (tick, inst), hit in snapped.items():
        bar_index = bisect_right(boundaries, tick / unit) - 1
        local = tick - boundaries[bar_index] * unit
        ratio = min(1, hit.strength / strengths[inst]) if strengths[inst] > 0 else 0
        state = ghosts.get((hit.time, inst), "normal")
        bars[bar_index].notes.append(Note(local, inst, max(1, min(127, round(40 + 70 * ratio))), state == "ghost", hit.time, state))
    order = {inst: i for i, inst in enumerate(INSTRUMENTS)}
    for bar in bars:
        bar.notes.sort(key=lambda note: (note.tick, order[note.inst]))
    return Score(title, bars, grid=grid, tempo=round(median(b.tempo for b in bars), 1), subtitle=subtitle,
                 merged_strokes=merged, pulse={
                     "source": pulse.source, "denominator_period": float(np.median(np.diff(times))),
                     "meters_used": list(dict.fromkeys(f"{n}/{signature.denominator}" for n in lengths)),
                     "alignment": pulse.alignment, "repetition": pulse.repetition,
                     "tracker_compatible": pulse.tracker_compatible,
                     "tempo_unit": "quarter", "display_tempo_unit": "dotted-quarter" if all(s.compound for s in signatures) else "quarter",
                 })
