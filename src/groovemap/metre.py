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
        if not isinstance(value, str) or not re.fullmatch(r"(?:[1-9]|1[0-5])/(?:4|8|16)", value):
            raise ValueError("time_signature must be N/4, N/8 or N/16, with N from 1 to 15")
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
