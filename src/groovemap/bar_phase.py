"""First-pass, cache-independent drum grammar for ranking bar starts.

The input beats must already be recounted/rescaled and oriented so beats[0]
is a beat_shift=0 downbeat. Returned shifts are *physical* positive beat
offsets modulo a bar, not necessarily values for the UI's beat_shift option:
the legacy quantizer recomputes downbeat voting after a fractional shift.
Scores are normalized heuristic log-potentials, not probabilities.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from .score import Hit

DEFAULT_WEIGHTS = {"metrical": 1.0, "backbeat": 1.0, "crash": 1.0, "fill": 0.25}


def _grid(beats: Sequence[float], bar_beats: int) -> np.ndarray:
    grid = np.asarray(beats, dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
        raise ValueError("beats must contain at least two finite, increasing times")
    if not isinstance(bar_beats, int) or bar_beats < 1:
        raise ValueError("bar_beats must be a positive integer")
    return grid


def bar_start_times(beats: Sequence[float], shift: float, bar_beats: int = 4) -> list[float]:
    """Map a candidate back to seconds, without extrapolating past the grid."""
    grid = _grid(beats, bar_beats)
    positions = np.arange(shift % bar_beats, len(grid) - 1, bar_beats)
    return np.interp(positions, np.arange(len(grid)), grid).tolist()


def phase_features(
    hits: Sequence[Hit], beats: Sequence[float], bar_beats: int = 4,
) -> dict[float, dict[str, float]]:
    """Four fixed features for each quarter-beat candidate; no label input.

    Each instrument is weighted by strength / its own 90th percentile.
    Stronger-half non-ghost snares supply backbeats. Generic cymbals are
    only crash proxies: a preceding gap of two beats earns full weight,
    shorter gaps downweight repetitive riding. Fills use toms in the final
    beat, peaking half a beat before the proposed start.
    """
    grid = _grid(beats, bar_beats)
    selected = sorted((h for h in hits if grid[0] <= h.time < grid[-1]), key=lambda h: h.time)
    if any(not np.isfinite(h.strength) or h.strength < 0 for h in selected):
        raise ValueError("hit strengths must be finite and nonnegative")
    x = np.interp([h.time for h in selected], grid, np.arange(len(grid)))
    inst = np.array([h.inst for h in selected], dtype=str)
    strength = np.array([h.strength for h in selected])
    weight = np.zeros(len(selected))
    for name in set(inst):
        mask = inst == name
        weight[mask] = np.clip(strength[mask] / max(float(np.percentile(strength[mask], 90)), 1e-9), 0, 1)

    kick = inst == "kick"
    snare = np.isin(inst, ["snare", "sidestick"]) & np.array([not h.ghost for h in selected], dtype=bool)
    if np.any(snare):
        snare &= strength >= np.median(strength[snare])
    crash = inst == "cymbal"
    tom = np.isin(inst, ["tom", "tom_high", "tom_low"])
    crash_novelty = np.minimum(np.diff(x[crash], prepend=-2) / 2, 1)

    def mean(values: np.ndarray, mask: np.ndarray, extra: np.ndarray | None = None) -> float:
        w = weight[mask]
        # Divide by original weight, so repetitive cymbals really lose evidence.
        return float(np.sum(values * w * (1 if extra is None else extra)) / max(w.sum(), 1e-9))

    def near(position: np.ndarray, target: float, period: float, width: float = 0.10) -> np.ndarray:
        distance = (position - target + period / 2) % period - period / 2
        return np.exp(-0.5 * (distance / width) ** 2)

    result = {}
    for shift in np.arange(4 * bar_beats) / 4:
        phase = x - shift
        # Nested lattices: beats get 1.60, eighth offbeats .60, sixteenths .15.
        metrical = near(phase, 0, 1) + .45 * near(phase, 0, .5) + .15 * near(phase, 0, .25, .05)
        # The grammar's 2/4 snare prior assumes an even-numbered common-time bar.
        kick_value = .8 * near(phase[kick], 0, 2) + .2 * near(phase[kick], 0, bar_beats)
        snare_value = near(phase[snare], 1, 2)
        result[float(shift)] = {
            "metrical": mean(metrical, np.ones(len(selected), dtype=bool)),
            "backbeat": .5 * (mean(kick_value, kick) + mean(snare_value, snare)),
            "crash": mean(near(phase[crash], 0, bar_beats), crash, crash_novelty),
            "fill": mean(near(phase[tom], bar_beats - .5, bar_beats, .35), tom),
        }
    return result


def rank_bar_phases(
    hits: Sequence[Hit], beats: Sequence[float], bar_beats: int = 4, *,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[float, float]]:
    """Rank all 4 * bar_beats offsets, descending score; ties use smaller shift.

    Hand-set default weights were fixed before the first label evaluation.
    Intended for 4/4; other meters enumerate correctly but are unvalidated.
    An empty hit list yields ties and zero margin, not a confident prediction.
    """
    weights = DEFAULT_WEIGHTS if weights is None else weights
    unknown = set(weights) - set(DEFAULT_WEIGHTS)
    if unknown or any(not np.isfinite(v) for v in weights.values()):
        raise ValueError("weights must be finite and use known feature names")
    features = phase_features(hits, beats, bar_beats)
    return sorted(
        ((shift, sum(weights.get(name, 0) * value for name, value in row.items())) for shift, row in features.items()),
        key=lambda row: (-row[1], row[0]),
    )


# The arrows reach a quarter beat short of a whole 4/4 bar either way.
_UI_SHIFTS = [step / 4 for step in range(-15, 16)]


def bar_start_candidates(
    hits: Sequence[Hit], beats: Sequence[float], downbeats: Sequence[float], *,
    beat_shift: float = 0, tempo_scale: float = 1, drum_picks: int = 2,
) -> list[dict]:
    """Other bar starts worth one click, as `beat_shift` values, for plain 4/4 scores.

    On the songs users corrected the drum ranking alone picked the right start
    only 1 time in 6 but held it in its top three 4 times in 6, and it misses
    half-beat offsets: so its best picks are offered next to the half-beat
    moves, never applied (reports/bar_phase_first_pass.md).
    """
    from .quantize import bar_grid

    base, bounds = bar_grid(hits, beats, downbeats, tempo_scale=tempo_scale)
    if any(b - a != 4 for a, b in zip(bounds, bounds[1:])):
        return []
    first = next((b for b in bounds if b >= 0), None)
    if first is None or len(base) - first < 9:
        return []
    grid = np.asarray(base[first:])

    def position(shift: float) -> float | None:
        """Where the bars start with this `beat_shift`, in beats of `grid`, within a bar."""
        shifted, starts = bar_grid(hits, beats, downbeats, beat_shift=shift, tempo_scale=tempo_scale)
        inside = [shifted[b] for b in starts if 0 <= b < len(shifted) and grid[0] <= shifted[b] <= grid[-1]]
        if not inside:
            return None
        return round(4 * float(np.interp(inside[0], grid, np.arange(len(grid))))) / 4 % 4

    # Each reachable start, by the smallest move that reaches it.
    reach: dict[float, float] = {}
    for shift in sorted(_UI_SHIFTS, key=abs):
        spot = position(shift)
        if spot is not None:
            reach.setdefault(spot, shift)
    current = position(beat_shift)
    if current is None:
        return []
    wanted = [(spot, "drums") for spot, _ in rank_bar_phases(hits, grid)]
    picks: list[tuple[float, str]] = []
    for spot, reason in wanted:
        if len([p for p in picks if p[1] == "drums"]) == drum_picks:
            break
        if spot != current and spot in reach:
            picks.append((spot, reason))
    for spot in ((current - .5) % 4, (current + .5) % 4):
        if spot in reach and spot not in (p[0] for p in picks):
            picks.append((spot, "half_beat_earlier" if spot == (current - .5) % 4 else "half_beat_later"))
    return [{"beat_shift": reach[spot], "reason": reason} for spot, reason in picks]
