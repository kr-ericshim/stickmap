"""Turn detector timestamps into a beat-relative drum score."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import groupby, product
from math import floor, isfinite
from statistics import median

import numpy as np

from .score import DIVISIONS, INSTRUMENTS, Bar, Hit, Note, Score


def _nearest(values: Sequence[float], value: float) -> int:
    right = bisect_left(values, value)
    if right == 0:
        return 0
    if right == len(values):
        return right - 1
    return right - 1 if value - values[right - 1] <= values[right] - value else right


# With odd_bars="auto": how many downbeat detections must disagree with a
# regular grid before an odd-length bar is believed. Trackers often mark beat 3
# as a downbeat for a few bars, mostly in sparse intros; a real odd bar shifts
# every later downbeat and soon outweighs this.
ODD_BAR_PENALTY = 12
ODD_BAR_MODES = ("none", "auto", "detected")
_MAX_BAR = 12


def _base_period(gaps: np.ndarray) -> float:
    """The beat interval the song is most plausibly counted in.

    Scores each interval by the running time the tracker spent at it, weighted
    by how natural that tempo is to count (a broad preference around 110 BPM),
    so a tracker that split its time between 90 and 180 BPM resolves to 90.
    """
    bins = np.round(np.log2(gaps) * 8)  # eighth-octave bins, about 9% wide
    seconds = {key: gaps[bins == key].sum() for key in np.unique(bins)}

    def score(key: float) -> float:
        covered = sum(seconds.get(key + step, 0.0) for step in (-1, 0, 1))
        octaves_off = np.log2(60 / 2 ** (key / 8) / 110)
        return covered * float(np.exp(-0.5 * (octaves_off / 0.6) ** 2))

    return float(np.median(gaps[bins == max(seconds, key=score)]))


_LEVELS = (0.25, 1 / 3, 0.5, 1, 2, 3, 4)  # the same tempo counted faster or slower
_SHORT_COAST = 8  # beats the walk can keep its own time without drifting far


def _settled(times: Sequence[float], i: int) -> float | None:
    """The tracker's beat period right after times[i], if its next four intervals agree."""
    gaps = np.diff(times[i + 1:i + 6])
    if len(gaps) < 4 or gaps.max() > 1.15 * gaps.min():
        return None
    return float(gaps.mean())


def _walk(times: Sequence[float], period: float) -> list[float]:
    """Follow the beats at one tempo level, starting from times[0].

    Each step takes the detected beat closest to one period ahead. While the
    tracker is erratic the walk keeps its own time; as soon as the tracker is
    steady again the walk lines up with it, because its own time has drifted.
    """
    kept = [times[0]]
    i = 0  # last detected beat at or before kept[-1]
    coast = 0  # consecutive beats kept on the walk's own time
    while True:
        expected = kept[-1] + period
        near = [j for j in range(i + 1, len(times)) if abs(times[j] - expected) <= 0.2 * period]
        if near:
            i = min(near, key=lambda j: abs(times[j] - expected))
            period = 0.9 * period + 0.1 * (times[i] - kept[-1])
            kept.append(times[i])
            coast = 0
            continue
        if i + 1 >= len(times) or times[-1] < expected - 0.2 * period:
            return kept
        settled = _settled(times, i)
        if settled is None:
            kept.append(expected)
            coast += 1
        else:
            ratio = settled / period
            level = min(_LEVELS, key=lambda value: abs(ratio / value - 1))
            if abs(ratio / level - 1) >= 0.12:  # an unrelated tempo: a real change
                i += 1
                period = settled
                kept.append(times[i])
                coast = 0
                continue
            # The tracker is steady at this tempo but out of step with the walk.
            step = settled / max(1, round(level))
            anchor = times[i + 1]
            phase = (expected - anchor) / step % 1
            if level >= 1 and coast <= _SHORT_COAST and abs(phase - 0.5) < 0.15:
                step /= 2  # too soon to have drifted half a beat: the tracker moved to the off-beats
            kept.append(anchor + round((expected - anchor) / step) * step)
            coast = 0
        while i + 1 < len(times) and times[i + 1] <= kept[-1] + 1e-9:
            i += 1


def _steady_beats(beats: Sequence[float]) -> list[float]:
    """Hold one tempo level throughout.

    Busy fills make trackers slip into double time for a while, or skip a beat.
    Walking outward from a stretch at the prevailing tempo drops the extra
    beats and fills in the missing ones; real tempo changes are kept.
    """
    gaps = np.diff(beats)
    period = _base_period(gaps)
    # Start in the longest steady stretch: when the tracker changes its mind about
    # where the beat falls, the side it held longest is the one to believe.
    steady = np.abs(gaps / period - 1) < 0.12
    anchor = best = run = 0
    for i, ok in enumerate(steady):
        run = run + 1 if ok else 0
        if run > best:
            best, anchor = run, i - run + 1
    before = _walk([-time for time in reversed(beats[:anchor + 1])], period)
    return [-time for time in reversed(before)][:-1] + _walk(beats[anchor:], period)


def _rescale(beats: list[float], downbeats: Sequence[float], scale: float) -> list[float]:
    """Count twice as fast (2) or half as fast (0.5) as the tracker did."""
    if scale == 2:
        middles = [(a + b) / 2 for a, b in zip(beats, beats[1:])]
        return sorted(beats + middles)
    if scale == 0.5:
        on_even = sum(_nearest(beats, time) % 2 == 0 for time in downbeats)
        return beats[0 if 2 * on_even >= len(downbeats) else 1::2]
    return beats


def _backbeat(hits: Sequence[Hit], beats: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """For the stronger half of the snare hits (no ghosts, few fill notes): the
    beat each one belongs to and how far through that beat it falls (0..1)."""
    snares = [h for h in hits if h.inst in ("snare", "sidestick") and beats[0] <= h.time < beats[-1]]
    if len(snares) < 16:
        return np.array([], dtype=int), np.array([])
    floor = median(h.strength for h in snares)
    times = np.array([h.time for h in snares if h.strength >= floor])
    index = np.searchsorted(beats, times, side="right") - 1
    starts, ends = np.asarray(beats)[index], np.asarray(beats)[index + 1]
    return index, (times - starts) / (ends - starts)


def _recount(hits: Sequence[Hit], beats: list[float]) -> list[float]:
    """Move the beats onto the backbeat when the snare sits halfway between them.

    A backbeat snare marks beats 2 and 4. Snares after every tracked beat mean
    the song is counted twice as fast (kick-snare-kick-snare): the halfway points
    become beats too. Snares after every other beat mean the tracker has the
    right tempo but sits half a beat off: the halfway points replace the beats.
    """
    index, phase = _backbeat(hits, beats)
    if not len(phase):
        return beats
    between = np.abs(phase - 0.5) < 0.09
    on_beat = float(np.mean((phase < 0.09) | (phase > 0.91)))
    if between.mean() < 0.45 or between.mean() < 2.5 * on_beat:
        return beats
    middles = [(a + b) / 2 for a, b in zip(beats, beats[1:])]
    odd = float(np.mean(index[between] % 2))
    return middles if max(odd, 1 - odd) >= 0.65 else sorted(beats + middles)


def _snare_parity(hits: Sequence[Hit], beats: Sequence[float]) -> int | None:
    """Parity of the beats that carry the backbeat, when the snare clearly favours one."""
    index, phase = _backbeat(hits, beats)
    on_beat = index[(phase < 0.09) | (phase > 0.91)] + (phase[(phase < 0.09) | (phase > 0.91)] > 0.5)
    if len(on_beat) < 16:
        return None
    odd = float(np.mean(on_beat % 2))
    return int(odd > 0.5) if max(odd, 1 - odd) >= 0.65 else None


def _boundaries(
    beats: Sequence[float], downbeats: Sequence[float], meter: int | None, odd_bars: str, shift: int,
    backbeat: int | None = None,
) -> list[int]:
    indices = sorted({_nearest(beats, time) for time in downbeats})
    # Bar length: the downbeat spacing the tracker held steadily for the most beats.
    # Where it stutters through a fill (1, 3, 1, 3, ...) the spacing says nothing.
    gaps = [b - a for a, b in zip(indices, indices[1:])]
    counts: Counter[int] = Counter()
    for gap, run in ((gap, len(list(group))) for gap, group in groupby(gaps)):
        if run >= 3 and 2 <= gap <= _MAX_BAR:
            counts[gap] += gap * run
    size = meter or (min(counts, key=lambda n: (-counts[n], n != 4, n)) if counts else 4)
    if size in (2, 8) and meter is None:
        size = 4  # a downbeat every 2 or 8 beats is almost always 4/4 counted at another level
    if odd_bars == "none" or not indices:
        phases = Counter(index % size for index in indices)
        allowed = range(size)
        if backbeat is not None and size % 2 == 0:
            allowed = [n for n in range(size) if n % 2 != backbeat]  # the snare lands on 2 and 4
        phase = (min(allowed, key=lambda n: (-phases[n], n)) + shift) % size
        start = phase - size if phase else 0
        return list(range(start, len(beats) + size, size))
    odd_bar_penalty = ODD_BAR_PENALTY if odd_bars == "auto" else 0

    # Cheapest chain of bar boundaries: regular bars are free, an odd-length bar
    # costs `odd_bar_penalty`, and each disagreement with a detection costs 1.
    detected, lo, hi = set(indices), indices[0], indices[-1]

    def boundary_cost(position: int) -> int:
        return int(lo <= position <= hi and position not in detected)

    last_beat = len(beats) - 1
    best = {start: (boundary_cost(start), None) for start in range(0, -size, -1)}
    for position in range(1, last_beat + _MAX_BAR + 1):
        options = []
        for length in range(1, _MAX_BAR + 1):
            previous = position - length
            if previous not in best or previous > last_beat:
                continue
            skipped = sum(1 for index in range(previous + 1, position) if index in detected)
            penalty = 0 if length == size else odd_bar_penalty
            options.append((best[previous][0] + penalty + skipped + boundary_cost(position), length != size, previous))
        if options:
            cost, _, previous = min(options)
            best[position] = (cost, previous)
    end = min(range(last_beat + 1, last_beat + _MAX_BAR + 1), key=lambda p: (best[p][0], p))
    boundaries = [end]
    while best[boundaries[-1]][1] is not None:
        boundaries.append(best[boundaries[-1]][1])
    return boundaries[::-1]


# Same-instrument onsets closer than this are one stroke: a flam or a detector double.
_MIN_GAP = 0.035
# Candidate subdivisions of a beat, simplest first, each with how much closer (in
# beats) a hit must be to a slot only that grid has than to its 16th slot.
_MARGIN = {4: 0.0, 3: 0.02, 8: 0.05, 6: 0.02}
# What each neighbouring beat takes off the cost of the grid it chose on its own.
_NEIGHBOUR = 0.008


def _strokes(hits: Sequence[Hit]) -> list[Hit]:
    """Merge flams and double detections: the earlier time, the stronger hit."""
    strokes: list[Hit] = []
    last: dict[str, int] = {}
    for hit in sorted(hits, key=lambda h: h.time):
        i = last.get(hit.inst)
        if i is not None and hit.time - strokes[i].time < _MIN_GAP:
            stronger = max(strokes[i], hit, key=lambda h: h.strength)
            strokes[i] = replace(stronger, time=strokes[i].time)
        else:
            last[hit.inst] = len(strokes)
            strokes.append(hit)
    return strokes


def _offsets(hits: Sequence[Hit], times: Sequence[float]) -> dict[str, float]:
    """How late each instrument is played (or detected), in seconds.

    Measured on hits next to a beat only: every grid has that slot, so a
    shuffle or a fill cannot pass for lateness. Instruments with few such hits
    share the offset of all of them; a handful of hits says nothing, and the
    offset is 0.
    """
    # One stroke per beat, the nearest: the second stroke of a fast double is not a late first.
    closest: dict[str, dict[int, float]] = {}
    for hit in hits:
        nearest = _nearest(times, hit.time)
        interval = times[min(nearest + 1, len(times) - 1)] - times[max(0, nearest - 1)]
        residual = hit.time - times[nearest]
        if abs(residual) <= min(0.06, interval * 0.06):
            best = closest.setdefault(hit.inst, {})
            if abs(residual) < abs(best.get(nearest, 1.0)):
                best[nearest] = residual
    residuals = {inst: list(best.values()) for inst, best in closest.items()}

    def clamped(values: list[float]) -> float:
        return max(-0.03, min(0.03, median(values))) if values else 0.0

    together = [r for values in residuals.values() for r in values]
    shared = clamped(together) if len(together) >= 8 else 0.0
    return {inst: clamped(residuals[inst]) if len(residuals.get(inst, ())) >= 20 else shared for inst in INSTRUMENTS}


_SWITCH = 0.016
_MAX_ERROR = 3.0  # alternative slots within 1/8 quarter; otherwise nearest only
_BITS = {inst: 1 << i for i, inst in enumerate(INSTRUMENTS)}


@dataclass
class _Choice:
    modes: tuple[int, int]
    slots: tuple[int, ...]
    start: int
    end: int
    lost: int
    cost: float
    error: float


@lru_cache(maxsize=8192)
def _notation_cost(slots: tuple[int, ...]) -> float:
    """Seconds of readability cost, measured on the actual one-voice spelling.

    Chords vote once. A real fast run can still win on timing/retention; an
    unused fine lattice alone does not earn a finer written resolution.
    """
    from .musicxml import _rhythm

    bar = Bar(0, 1, [0, 1], [4], 60, [Note(t, "snare", 90) for t in slots if t < 24])
    events = _rhythm(bar, bar.notes)
    return sum(
        0.004 * int(not e.notes and e.duration < 6)
        + 0.003 * int(bool(e.notes) and e.note_type in ("eighth", "16th", "32nd") and 1 not in e.beams)
        + 0.001 * int(e.duration in (9, 18, 36))
        + 0.0015 * int(bool(e.notes) and e.note_type == "32nd")
        for e in events
    ) + 0.002 * sum(a.ratio != b.ratio or a.tuplet != b.tuplet for a, b in zip(events, events[1:]))


def _event_cost(group: Sequence[tuple[Hit, float]], slots: tuple[int, ...], modes: tuple[int, int], duration: float) -> float:
    """IOI and split-unison evidence, independent of the number of drums struck."""
    events: list[list[tuple[float, int, float]]] = []
    for (hit, fraction), slot in sorted(zip(group, slots), key=lambda pair: pair[0][0].time):
        if events and hit.time - events[-1][0][2] <= 0.015 + 1e-9:
            events[-1].append((fraction, slot, hit.time))
        else:
            events.append([(fraction, slot, hit.time)])
    points = [(median(e[0] for e in event), median(e[1] for e in event) / 24) for event in events]
    timing = sum(sum(
        abs(fraction - slot / 24)
        + (max(_MARGIN[modes[0 if slot < 12 else 1]], 0.004 / duration) if slot % 6 else 0)
        for fraction, slot, _ in event
    ) / len(event) for event in events)
    gaps = np.diff([p[0] for p in points])
    regular = len(points) >= 4 and gaps.mean() > 0 and gaps.std() < 0.35 * gaps.mean()
    ioi = sum(abs((b[0] - a[0]) - (b[1] - a[1])) for a, b in zip(points, points[1:])) if regular else 0.0
    split = sum(max(e[1] for e in event) != min(e[1] for e in event) for event in events)
    return timing + 0.3 * ioi + 0.025 * split


def _display_onsets(strokes: Sequence[Hit], times: Sequence[float]) -> list[float]:
    """Apply instrument latency without pulling exact unisons onto different slots."""
    offsets = _offsets(strokes, times)
    simultaneous: dict[float, list[float]] = defaultdict(list)
    for hit in strokes:
        simultaneous[hit.time].append(offsets[hit.inst])
    return [hit.time - median(simultaneous[hit.time]) for hit in strokes]


def _instrument_options(
    items: Sequence[tuple[int, float]], ticks: Sequence[int], modes: tuple[int, int], duration: float,
) -> dict[tuple[bool, bool], tuple[int, float, float, tuple[tuple[int, int], ...]]]:
    """Keep each instrument's strokes in order, with a choice for each occupied edge."""
    # State: last assigned tick, and whether the beat start is occupied.
    # Equal ticks count as an explicit merge.
    dp = {(-1, False): (0, 0.0, 0.0, ())}
    for index, fraction in items:
        x = fraction * 24
        allowed = [
            t for t in ticks
            # One detector frame of slack prevents a small phase correction
            # from turning a previously retainable dense stroke into a merge.
            if abs(t - x) <= _MAX_ERROR + min(1.0, 0.01 / duration * 24) + 1e-9
        ]
        if not allowed:
            allowed = [min(ticks, key=lambda t: (abs(t - x), t))]

        nxt = {}
        for (last, start), (lost, cost, error, path) in dp.items():
            for t in allowed:
                if t < last:
                    continue
                mode = modes[0 if t < 12 else 1]
                margin = (
                    max(_MARGIN[mode], 0.004 / duration)
                    if t % 6 else 0.0
                )
                err = abs(t - x) / 24
                value = (
                    lost + int(t == last),
                    cost + err + margin,
                    error + err,
                    path + ((index, t),),
                )
                key = (t, start or t == 0)
                if key not in nxt or value < nxt[key]:
                    nxt[key] = value
        dp = nxt

    out = {}
    for (last, start), value in dp.items():
        key = (start, last == 24)
        if key not in out or value < out[key]:
            out[key] = value
    return out


def _choices(
    group: Sequence[tuple[Hit, float]], duration: float, triplets: bool, terminal: bool,
    *, uniform: bool = False, reading: bool = False,
) -> list[_Choice]:
    if terminal and reading and any(f >= 0.875 for _, f in group):
        preceding = [(hit, f) for hit, f in group if f < 0.875]
        supported = min(_choices(preceding, duration, triplets, True), key=lambda c: (c.lost, c.cost, c.error))
        if supported.modes == (4, 4):
            # The page ending alone must not introduce a finer grid in reading mode.
            slots = tuple(12 if f >= 0.875 else floor(f * 4 + 0.5) * 6 for _, f in group)
            kept = [(hit.inst, t) for (hit, _), t in zip(group, slots) if t % 12 == 0]
            start = sum(_BITS[inst] for inst in {inst for inst, t in kept if t == 0})
            error = sum(abs(f - t / 24) for (_, f), t in zip(group, slots))
            return [_Choice((4, 4), slots, start, 0, len(kept) - len(set(kept)), error, error)]
    grids = [4] + [
        g for g in (8, 6)
        if (triplets or g != 6) and duration / g >= _MIN_GAP
    ]
    patterns = [(g, g) for g in grids] if uniform else list(product(grids, repeat=2))
    if triplets:
        patterns.append((3, 3))  # Atomic whole-beat eighth triplet.

    result = {}
    by_inst = defaultdict(list)
    for j, (hit, fraction) in enumerate(group):
        by_inst[hit.inst].append((j, fraction))

    for modes in patterns:
        ticks = (
            [0, 8, 16, 24] if modes == (3, 3) else
            sorted(
                set(range(0, 13, 24 // modes[0]))
                | set(range(12, 25, 24 // modes[1]))
            )
        )
        if terminal:
            ticks = [t for t in ticks if t < 24]

        combined = [(0, 0, 0, 0.0, 0.0, ())]
        for inst, items in by_inst.items():
            options = _instrument_options(items, ticks, modes, duration)
            combined = [
                (
                    sm | (_BITS[inst] if start else 0),
                    em | (_BITS[inst] if end else 0),
                    lost + value[0],
                    cost + value[1],
                    err + value[2],
                    path + value[3],
                )
                for sm, em, lost, cost, err, path in combined
                for (start, end), value in options.items()
            ]

        for sm, em, lost, cost, err, path in combined:
            slots = tuple(t for _, t in sorted(path))

            # A fine grid with no distinctive occupied slots has no evidence.
            actual = tuple(modes)
            if modes != (3, 3):
                actual = tuple(
                    g if any(a <= t < a + 12 and t % 6 for t in slots)
                    else 4
                    for a, g in zip((0, 12), modes)
                )
            elif not any(t in (8, 16) for t in slots):
                actual = (4, 4)
            if triplets and any(t in (8, 16) for t in slots) and all(t % 8 == 0 for t in slots):
                actual = (3, 3)  # Sparse sextuplets spell as whole-beat eighth triplets.

            cost = _event_cost(group, slots, actual, duration)
            cost += _SWITCH * int(actual[0] != actual[1])
            cost += _notation_cost(tuple(sorted(set(slots)))) / duration
            choice = _Choice(actual, slots, sm, em, lost, cost, err)
            key = (actual, sm, em)
            if key not in result or (lost, cost, err, slots) < (
                result[key].lost,
                result[key].cost,
                result[key].error,
                result[key].slots,
            ):
                result[key] = choice

    return list(result.values())


def _solve_window(
    candidates: Sequence[Sequence[_Choice]], previous: tuple[int | None, int],
) -> tuple[_Choice, ...]:
    """Minimize merges first, then timing and grid costs, across the beat boundaries."""
    # Boundary state: preceding half's mode and occupied boundary instruments.
    states = {previous: ((0, 0.0, 0.0), ())}
    for choices in candidates:
        nxt = {}
        for c in choices:
            best = None
            for (mode, occupied), (value, path) in states.items():
                cross = (occupied & c.start).bit_count()
                transition = _SWITCH * int(
                    mode is not None and mode != c.modes[0]
                )
                score = (value[0] + c.lost + cross, value[1] + c.cost + transition, value[2] + c.error)
                option = (score, path + (c,))
                if best is None or score < best[0]:
                    best = option

            key = (c.modes[1], c.end)
            if key not in nxt or best[0] < nxt[key][0]:
                nxt[key] = best
        states = nxt

    return min(states.values(), key=lambda p: p[0])[1]


def _half_plan(
    per_beat: Sequence[Sequence[tuple[Hit, float]]], times: Sequence[float],
    lengths: Sequence[int], triplets: bool, reading: bool = False,
) -> list[_Choice]:
    """Choose half-beat grids and assignments, looking one bar beyond each commit."""
    candidates = [
        _choices(
            group, times[i + 1] - times[i], triplets,
            i == len(per_beat) - 1, reading=reading,
        )
        for i, group in enumerate(per_beat)
    ]

    # Uniform candidates give the neighbouring beats their weak evidence votes.
    alone = []
    for i, group in enumerate(per_beat):
        found = {}
        for c in _choices(group, times[i + 1] - times[i], triplets, i == len(per_beat) - 1, uniform=True, reading=reading):
            mode = next((g for g in c.modes if g != 4), 4)
            value = (c.lost, c.error, c.cost - _SWITCH * (c.modes[0] != c.modes[1]))
            if mode not in found or (value[0], value[2], value[1]) < (found[mode][0], found[mode][2], found[mode][1]):
                found[mode] = value
        fewest = min(v[0] for v in found.values())
        eligible = [g for g, v in found.items() if v[0] == fewest]
        if 8 in eligible and 6 in eligible:
            eligible.remove(max(6, 8, key=lambda g: found[g][1]))
        chosen = min(eligible, key=lambda g: found[g][2])
        informative = any(floor(f * 4 + 0.5) % 4 for _, f in group)
        alone.append(chosen if informative else None)

    for i, choices in enumerate(candidates):
        neighbours = [
            alone[j]
            for j in (i - 1, i + 1, i - lengths[i], i + lengths[i])
            if 0 <= j < len(alone)
        ]
        for c in choices:
            c.cost -= (
                _NEIGHBOUR
                * sum(neighbours.count(g) for g in c.modes) / 2
            )

    selected, previous, i = [], (None, 0), 0
    while i < len(per_beat):
        # Optimize two bars, commit one, and retain the actual boundary state.
        stop = min(len(per_beat), i + lengths[i])
        end = min(
            len(per_beat),
            stop + (lengths[stop] if stop < len(per_beat) else 0),
        )
        path = _solve_window(candidates[i:end], previous)
        committed = path[:stop - i]
        selected.extend(committed)
        previous = (committed[-1].modes[1], committed[-1].end)
        i = stop

    return selected


def bar_grid(
    hits: Sequence[Hit], beats: Sequence[float], downbeats: Sequence[float], *,
    meter: int | None = None, odd_bars: str = "none", beat_shift: float = 0, tempo_scale: float = 1,
) -> tuple[list[float], list[int]]:
    """The beats `quantize` counts and the beat indices where its bars start."""
    beats = _rescale(_recount(hits, _steady_beats(beats)), downbeats, tempo_scale)
    if fraction := beat_shift % 1:  # part of a beat later: those points between beats become the beats
        beats = [a + fraction * (b - a) for a, b in zip(beats, beats[1:])]
    if len(beats) < 2:
        raise ValueError("At least two beats are required")
    return beats, _boundaries(beats, downbeats, meter, odd_bars, floor(beat_shift), _snare_parity(hits, beats))


def quantize(
    hits: Sequence[Hit], beats: Sequence[float], downbeats: Sequence[float], *,
    title: str = "", subtitle: str = "", grid: int = 16,
    triplets: str = "auto", meter: int | None = None, odd_bars: str = "none",
    beat_shift: float = 0, tempo_scale: float = 1,
    attacks: Sequence[float] = (),
) -> Score:
    """Quantize hits onto a grid that follows the tracked beats.

    `odd_bars` says how far to trust the detected downbeats: "none" cuts
    regular bars of `meter` beats (the dominant detected length if None),
    "auto" allows an odd-length bar when the detections insist, "detected"
    follows every detected downbeat. `beat_shift` moves beat 1 later by that
    many beats, in steps of a quarter beat ("none" only); `tempo_scale` 2 or 0.5
    counts the beat twice or half as fast as the default.
    """
    beats = [float(time) for time in beats]
    if len(beats) < 2:
        raise ValueError("At least two beats are required")
    if not all(isfinite(t) for t in beats) or any(b <= a for a, b in zip(beats, beats[1:])):
        raise ValueError("Beats must be finite and strictly increasing")
    if grid not in (8, 16) or triplets not in ("off", "auto"):
        raise ValueError("grid must be 8 or 16 and triplets must be 'off' or 'auto'")
    if meter is not None and (not isinstance(meter, int) or not 1 <= meter <= 12):
        raise ValueError("meter must be an integer from 1 to 12")
    if not all(isfinite(t) for t in downbeats):
        raise ValueError("Downbeats must be finite")
    if any(h.inst not in INSTRUMENTS or not isfinite(h.time) or not isfinite(h.strength) for h in hits):
        raise ValueError("Hits must have a known instrument and finite time and strength")

    if odd_bars not in ODD_BAR_MODES or tempo_scale not in (0.5, 1, 2) or (4 * beat_shift) % 1:
        raise ValueError("odd_bars: none, auto or detected; tempo_scale: 0.5, 1 or 2; beat_shift: steps of 0.25")
    beats, boundaries = bar_grid(hits, beats, downbeats, meter=meter, odd_bars=odd_bars,
                                 beat_shift=beat_shift, tempo_scale=tempo_scale)
    first, last = boundaries[0], boundaries[-1]
    left_interval, right_interval = beats[1] - beats[0], beats[-1] - beats[-2]
    times = [
        beats[0] + i * left_interval if i < 0 else
        beats[-1] + (i - len(beats) + 1) * right_interval if i >= len(beats) else beats[i]
        for i in range(first, last + 1)
    ]
    # A hit just ahead of the first beat is that beat played early, not an uncovered one.
    early = (times[1] - times[0]) / 8
    strokes = _strokes([h for h in hits if times[0] - early <= h.time < times[-1]])
    from .pulse import refine_display_times

    times = refine_display_times(times, [h.time for h in strokes], attacks, triplets=triplets == "auto")
    per_beat: list[list[tuple[Hit, float]]] = [[] for _ in times[:-1]]
    for hit, onset in zip(strokes, _display_onsets(strokes, times)):
        time = max(times[0], onset)
        beat = min(bisect_right(times, time), len(per_beat)) - 1
        per_beat[beat].append((hit, (time - times[beat]) / (times[beat + 1] - times[beat])))
    bar_lengths = [end - start for start, end in zip(boundaries, boundaries[1:]) for _ in range(end - start)]
    plans = _half_plan(per_beat, times, bar_lengths, triplets == "auto", grid == 8)
    subdivisions = []
    for plan in plans:
        a, b = (2 if grid == 8 and g == 4 else g for g in plan.modes)
        subdivisions.append(a if a == b else [a, b])
    strengths = {
        inst: float(np.percentile([h.strength for h in hits if h.inst == inst], 90))
        for inst in INSTRUMENTS if any(h.inst == inst for h in hits)
    }
    from .dynamics import ghost_states

    ghosts = ghost_states(strokes)
    bars = []
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        a, b = start - first, end - first
        bars.append(Bar(
            index=index, beats=b - a, beat_times=times[a:b + 1],
            subdivisions=subdivisions[a:b],
            tempo=60 * (b - a) / (times[b] - times[a]),
        ))

    snapped: dict[tuple[int, str], Hit] = {}
    merged = 0
    for beat, (group, plan) in enumerate(zip(per_beat, plans)):
        for (hit, _), slot in zip(group, plan.slots):
            mode = plan.modes[0 if slot < 12 else 1]
            if grid == 8 and mode == 4 and slot % 12:
                continue
            tick = beat * DIVISIONS + slot
            key = tick, hit.inst
            merged += int(key in snapped)
            if key not in snapped or hit.strength > snapped[key].strength:
                snapped[key] = hit
    starts = [b - first for b in boundaries]
    for (tick, inst), hit in snapped.items():
        bar_index = bisect_right(starts, tick / DIVISIONS) - 1
        p90 = strengths[inst]
        ratio = min(1.0, hit.strength / p90) if p90 > 0 else 0.0
        velocity = max(1, min(127, round(40 + 70 * ratio)))
        state = ghosts.get((hit.time, inst), "normal")
        bars[bar_index].notes.append(Note(tick - starts[bar_index] * DIVISIONS, inst, velocity, state == "ghost", hit.time, state))
    order = {inst: i for i, inst in enumerate(INSTRUMENTS)}
    for bar in bars:
        bar.notes.sort(key=lambda n: (n.tick, order[n.inst]))
    return Score(title=title, subtitle=subtitle, bars=bars, grid=grid, merged_strokes=merged, tempo=round(median(b.tempo for b in bars), 1))
