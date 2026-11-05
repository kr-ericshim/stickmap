"""Explicit signatures have a note unit; the legacy meter option does not."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Signature:
    numerator: int
    denominator: int

    @classmethod
    def parse(cls, value: str) -> Signature:
        if not isinstance(value, str) or not re.fullmatch(r"(?:[1-9]|[12][0-9]|3[0-2])/(?:4|8|16)", value):
            raise ValueError("time_signature must be N/4, N/8 or N/16, with N from 1 to 32")
        return cls(*map(int, value.split("/")))

    @property
    def groups(self) -> list[int]:
        """Compound beats use threes; asymmetric metres put the long group last."""
        n = self.numerator
        if self.denominator == 4 or n == 1:
            return [1] * n
        if n % 3 == 0:
            return [3] * (n // 3)
        return [2] * (n // 2) if n % 2 == 0 else [2] * ((n - 3) // 2) + [3]

    @property
    def compound(self) -> bool:
        return self.denominator == 8 and self.numerator >= 6 and self.numerator % 3 == 0


def parse_signatures(value: str) -> tuple[Signature, ...]:
    """Parse a set of up to four bar lengths on one denominator pulse."""
    if not isinstance(value, str):
        raise ValueError("time_signature must be a string")
    parts = value.split("+")
    if not 1 <= len(parts) <= 4:
        raise ValueError("time_signature accepts at most four signatures")
    signatures = tuple(Signature.parse(part) for part in parts)
    if len({s.denominator for s in signatures}) != 1:
        raise ValueError("Mixed signatures must share one denominator")
    return tuple(dict.fromkeys(signatures))


def parse_meter_changes(changes: object, time_signature: str) -> list[tuple[float, Signature]]:
    """User-placed changes: from each audio time on, bars use that signature."""
    base = parse_signatures(time_signature)[0]
    if not isinstance(changes, list) or len(changes) > 64:
        raise ValueError("meter_changes must be a list of at most 64 changes")
    parsed = []
    for change in changes:
        if not isinstance(change, dict) or set(change) != {"time", "time_signature"}:
            raise ValueError("each meter change needs exactly time and time_signature")
        time = change["time"]
        if isinstance(time, bool) or not isinstance(time, (int, float)) or not 0 <= time < 86400:
            raise ValueError("meter change time must be seconds from the start")
        signature = Signature.parse(change["time_signature"])
        if signature.denominator != base.denominator:
            raise ValueError("meter changes must keep the time signature's denominator")
        parsed.append((float(time), signature))
    times = [time for time, _ in sorted(parsed, key=lambda c: c[0])]
    if any(b - a < 0.05 for a, b in zip(times, times[1:])):
        raise ValueError("meter changes must be at different times")
    return sorted(parsed, key=lambda c: c[0])
