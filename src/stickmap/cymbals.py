"""Infer ride runs from timing alone; isolated generic cymbals stay crashes."""

from dataclasses import replace
from statistics import median

from .score import Hit

RIDE_MAX_GAP = 0.8
RIDE_IOI_TOLERANCE = 0.25
RIDE_MIN_RUN = 2
_MOMENT_WINDOW = 0.025
_EPSILON = 1e-12


def ride_runs(hits: list[Hit]) -> list[Hit]:
    """Relabel greedy, non-overlapping regular runs without changing hit order.

    Doubles within 25 ms of a moment's first hit share that moment and its
    label. A rejected extension starts the next run (the boundary is not
    shared). Thus gaps of 0.2 then 0.5 seconds yield a pair and an isolated
    hit, not one three-hit run. This is a timing heuristic, not timbre evidence.
    """
    relabel = {index for run in ride_run_indices(hits) for index in run}
    return [replace(hit, inst="ride") if i in relabel else hit for i, hit in enumerate(hits)]


def ride_run_indices(hits: list[Hit]) -> list[list[int]]:
    """Original hit indices for each eligible run, in stable time order."""
    moments: list[list[int]] = []
    for index in sorted((i for i, hit in enumerate(hits) if hit.inst == "cymbal"),
                        key=lambda i: hits[i].time):
        if moments and hits[index].time - hits[moments[-1][0]].time <= _MOMENT_WINDOW + _EPSILON:
            moments[-1].append(index)
        else:
            moments.append([index])

    times = [hits[moment[0]].time for moment in moments]
    runs: list[list[int]] = []
    start = 0
    while start < len(moments):
        end = start + 1
        intervals: list[float] = []
        while end < len(moments):
            gap = times[end] - times[end - 1]
            if gap > RIDE_MAX_GAP + _EPSILON:
                break
            candidate = intervals + [gap]
            center = median(candidate)
            if any(abs(ioi - center) > center * RIDE_IOI_TOLERANCE + _EPSILON for ioi in candidate):
                break
            intervals = candidate
            end += 1
        if end - start >= RIDE_MIN_RUN:
            runs.append([index for moment in moments[start:end] for index in moment])
        start = end

    return runs
