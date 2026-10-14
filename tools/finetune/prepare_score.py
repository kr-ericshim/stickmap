"""Precompute this tag's derived snare responses on at most four CPU workers.

The sequential global scorer otherwise rebuilds every missing response serially.
Only existing student predictions/stems are read; outputs stay in the FT cache.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools" / "harvest"), str(Path(__file__).parent)]
import harvest
from score import load, songs
from solos import songs as solo_songs


def one(item: tuple[str, Path]) -> None:
    import ft_variants

    tag, song = item
    ft_variants.TAG = tag
    ft_variants.snare(song, ft_variants.student(song), f"student-{tag}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", required=True)
    p.add_argument("--only", nargs="+", required=True)
    p.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    a = p.parse_args()
    selected = []
    for row in songs(a.only):
        song = harvest.SONGS / row["slug"]
        try:
            truth, _ = load(song / "truth", "strict")
        except FileNotFoundError:
            continue
        if truth:
            selected.append(song)
    selected += solo_songs()
    with ProcessPoolExecutor(a.jobs) as pool:
        list(pool.map(one, [(a.tag, s) for s in selected]))
    print(len(selected), "snare responses prepared for", a.tag)


if __name__ == "__main__":
    main()
