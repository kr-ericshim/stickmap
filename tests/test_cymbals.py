from dataclasses import replace

import pytest

from groovemap.cymbals import ride_runs
from groovemap.score import Hit


@pytest.mark.parametrize("times,expected", [
    ([], []),
    ([0], ["cymbal"]),
    ([0, 0.25, 0.5, 0.75], ["ride"] * 4),
    ([0, 0.5, 1, 1.5], ["ride"] * 4),
    ([0, 0.5], ["ride"] * 2),
    ([0, 0.9], ["cymbal"] * 2),
    ([0, 0.8, 1.6], ["ride"] * 3),
    ([0, 0.2, 0.7], ["ride", "ride", "cymbal"]),
    ([0, 1, 1.5, 2], ["cymbal", "ride", "ride", "ride"]),
    ([0, 0.5, 2, 3.5, 4], ["ride", "ride", "cymbal", "ride", "ride"]),
    ([0, 0.2, 0.7, 1.2], ["ride"] * 4),
    ([0, 0.3, 0.8], ["ride"] * 3),  # inclusive 25% around median 0.4
    ([0, 0.299, 0.8], ["ride", "ride", "cymbal"]),
    ([0, 0, 0.025], ["cymbal"] * 3),
    ([1, 1.025], ["cymbal"] * 2),
    ([0, 0.02, 0.5, 0.525, 1], ["ride"] * 5),
    ([0, 0.025, 0.05], ["ride"] * 3),  # anchored moments, not transitive doubles
])
def test_ride_runs(times, expected):
    hits = [Hit(time, "cymbal", 0.7) for time in times]
    assert [hit.inst for hit in ride_runs(hits)] == expected
    assert all(hit.inst == "cymbal" for hit in hits)


def test_preserves_order_fields_and_other_instruments():
    hits = [Hit(0.5, "cymbal", 0.2, True, 0.4), Hit(0, "snare", 0.3),
            Hit(0, "cymbal", 0.8, False, 0.9), Hit(0.2, "ride_bell", 0.6),
            Hit(0.25, "ride", 0.5), Hit(2, "cymbal", 0.9)]
    result = ride_runs(hits)
    assert result == [replace(hit, inst="ride") if i in (0, 2) else hit for i, hit in enumerate(hits)]
    assert all(result[i] is hits[i] for i in (1, 3, 4, 5))
    assert ride_runs(result) == result
