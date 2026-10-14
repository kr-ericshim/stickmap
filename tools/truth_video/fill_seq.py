"""Score the detector on dense solo passages by stroke order, not exact time.

The solo transcription videos (Larnell Lewis, Tony Royster, ...) are mostly
fast fills, and nearly every system fails the strict timing gates of
align.py, so none of it reaches truth.json. Their fitted times still put
each notated moment within a stroke or two of where it was played. Here the
notated kick/snare/tom moments of every system that was read cleanly are
lined up, in order, with the detected ones (a monotone alignment that may
only pair moments within --tol seconds), and what is left over counts as
missed or extra.

    .venv/bin/python tools/truth_video/fill_seq.py [--videos larnell-lewis tony-royster] [--jd]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "harvest"))
import vault  # noqa: E402
from align import moment_instruments, moments  # noqa: E402
from evaluate import family  # noqa: E402

TRUTH = HERE.parents[1] / "truth"
DRUMS = ("kick", "snare", "tom")
# Only systems rejected for their timing alone are used; any doubt about the reading rules a system out.
TIMING = ("held-out agreement", "fitted attack agreement", "ambiguous alignment", "no-cursor alignment")
FAST = 0.09  # seconds: a notated moment this close to a neighbour is part of a very fast passage
SOLOS = ("benny-greb", "devon-taylor", "jrod", "ken-jezer", "larnell-lewis", "michael-alba",
         "tension", "todd-sucherman", "tony-royster", "when-a-drummer")


def notated(folder: Path, min_after: float) -> list[tuple[float, float, list[tuple[float, frozenset[str]]]]]:
    """(start, end, [(time, drums)]) for each cleanly read system."""
    strips = {s["system"]: s for s in json.loads((folder / "strip.json").read_text())}
    out = []
    for r in json.loads((folder / "alignment.json").read_text())["systems"]:
        if not r.get("times") or not all(q.startswith(TIMING) for q in r.get("reasons", [])):
            continue
        if not min_after <= (r.get("after") or {}).get("within_40ms", 0) <= 1:
            continue
        s = strips.get(r["system"])
        if s is None or s.get("staff", {}).get("reasons"):
            continue
        if any(n.get("exportable") is False and family(n.get("inst", "")) in DRUMS for n in s["notes"]):
            continue  # a drum head whose reading was not confirmed
        x, groups = moments(s)
        times = np.asarray(r["times"], dtype=float)
        if (len(groups) != len(times) or not np.isfinite(times).all() or
                times[0] < 0 or np.any(np.diff(times) <= 0)):
            continue
        seq = []
        for t, names in zip(r["times"], moment_instruments(s, groups)):
            drums = frozenset(family(n) for n in names) & set(DRUMS)
            if drums:
                seq.append((float(t), drums))
        if len(seq) >= 6:
            out.append((float(r["times"][0]), float(r["times"][-1]), seq))
    return out


def detected(hits, a: float, b: float) -> list[tuple[float, frozenset[str]]]:
    out: list[tuple[float, set[str]]] = []
    for h in sorted(hits, key=lambda h: h.time):
        fam = family(h.inst)
        if fam not in DRUMS or not a <= h.time <= b:
            continue
        if out and h.time - out[-1][0] <= 0.025:
            out[-1][1].add(fam)
        else:
            out.append((h.time, {fam}))
    return [(t, frozenset(d)) for t, d in out]


def align(ref, est, tol):
    """Monotone pairing of moments; unpaired ones cost 1, a pair costs its label mismatch and time gap."""
    n, m = len(ref), len(est)
    cost = np.full((n + 1, m + 1), np.inf)
    back = np.zeros((n + 1, m + 1), np.uint8)
    cost[0] = np.arange(m + 1)
    cost[:, 0] = np.arange(n + 1)
    for i in range(1, n + 1):
        ti, ai = ref[i - 1]
        for j in range(1, m + 1):
            options = [(cost[i - 1, j] + 1, 1), (cost[i, j - 1] + 1, 2)]
            tj, bj = est[j - 1]
            if abs(ti - tj) <= tol:
                mismatch = len(ai ^ bj) / max(len(ai), len(bj))
                options.append((cost[i - 1, j - 1] + 0.9 * mismatch + 0.1 * abs(ti - tj) / tol, 0))
            cost[i, j], back[i, j] = min(options)
    pairs, i, j = [], n, m
    while i and j:
        step = back[i, j]
        if step == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def fast_mask(ref) -> np.ndarray:
    """Nearest-neighbour gap < 90 ms, shared by loose scorers and vault stats."""
    times = np.array([t for t, _ in ref])
    if not len(times):
        return np.zeros(0, dtype=bool)
    gaps = np.diff(times)
    return np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf]) < FAST


def tally(ref, est, tol) -> Counter:
    c: Counter = Counter()
    pairs = align(ref, est, tol)
    times = np.array([t for t, _ in ref])
    fast = fast_mask(ref)
    paired_ref = {i: j for i, j in pairs}
    paired_est = {j for _, j in pairs}
    for i, (t, a) in enumerate(ref):
        speed = "fast" if fast[i] else "slow"
        b = est[paired_ref[i]][1] if i in paired_ref else frozenset()
        for key in ("all", speed):
            c[f"{key}.stroke.ref"] += 1
            c[f"{key}.stroke.tp"] += i in paired_ref
            for fam in DRUMS:
                c[f"{key}.{fam}.ref"] += fam in a
                c[f"{key}.{fam}.tp"] += fam in a and fam in b
        for fam in ("tom", "snare"):
            if a == {fam}:
                c[f"conf.{fam}->{'+'.join(sorted(b)) or 'nothing'}"] += 1
    for j, (t, b) in enumerate(est):
        near = np.abs(times - t).argmin() if len(times) else None
        speed = "fast" if near is not None and fast[near] else "slow"
        for key in ("all", speed):
            c[f"{key}.stroke.est"] += 1
            for fam in DRUMS:
                c[f"{key}.{fam}.est"] += fam in b
    return c


def one(args) -> tuple[str, Counter]:
    video, detector, tol, min_after, *mode = args
    folder = TRUTH / video
    song = Path((folder / "song.txt").read_text().strip())
    if vault.is_vault_song(song) != (mode[0] if mode else False):
        return video, Counter()
    systems = notated(folder, min_after)
    c: Counter = Counter()
    if not systems or not (song / "activations.drums-htdemucs.npy").exists():
        return video, c
    module, name = detector.split(":")
    detect = getattr(importlib.import_module(module), name)
    hits = detect(song)
    for a, b, ref in systems:
        c += tally(ref, detected(hits, a - 0.05, b + 0.05), tol)
        c["systems"] += 1
        c["seconds"] += b - a
    return video, c


def line(c: Counter, key: str) -> str:
    cells = []
    for fam in ("stroke", *DRUMS):
        tp, est, ref = c[f"{key}.{fam}.tp"], c[f"{key}.{fam}.est"], c[f"{key}.{fam}.ref"]
        cells.append(f"{fam} P{tp / max(1, est):.2f} R{tp / max(1, ref):.2f} F{2 * tp / max(1, est + ref):.2f} ({ref})")
    return " | ".join(cells)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--videos", nargs="*")
    p.add_argument("--jd", action="store_true", help="also the JD Beck videos (unusual kit)")
    p.add_argument("--tol", type=float, default=0.06)
    p.add_argument("--min-after", type=float, default=0.75)
    p.add_argument("--detect", default="evaluate:detect")
    p.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    p.add_argument("--json", type=Path)
    p.add_argument("--final", action="store_true", help="open and score ONLY vault video IDs (logged)")
    p.add_argument("--model-tag", help="model identity for the vault opening log (default: FT_TAG or detector)")
    a = p.parse_args()
    if not np.isfinite(a.tol) or a.tol <= 0:
        p.error("--tol must be positive and finite")
    if not np.isfinite(a.min_after) or not 0 <= a.min_after <= 1:
        p.error("--min-after must be in [0, 1]")
    if a.final:
        vault.open_final(vault.model_tag(a.detect, a.model_tag))
    videos = a.videos or [*SOLOS, *(sorted(d.name for d in TRUTH.glob("jd-*") if (d / "strip.json").exists()) if a.jd else [])]
    seal = vault.Vault()
    videos = [v for v in videos if seal.contains_song(Path((TRUTH / v / "song.txt").read_text().strip())) == a.final]
    total: Counter = Counter()
    with ProcessPoolExecutor(a.jobs) as pool:
        results = list(pool.map(one, [(v, a.detect, a.tol, a.min_after, a.final) for v in videos]))
    for video, c in results:
        if c["systems"]:
            print(f"{video:16s} {c['systems']:3d} sys {c['seconds']:5.0f}s | {line(c, 'all')}")
        total += c
    print(f"\n{'ALL':16s} {total['systems']:3d} sys {total['seconds']:5.0f}s | {line(total, 'all')}")
    print(f"{'very fast':27s} | {line(total, 'fast')}")
    print(f"{'rest':27s} | {line(total, 'slow')}")
    for fam in ("tom", "snare"):
        row = {k.split("->")[1]: v for k, v in total.items() if k.startswith(f"conf.{fam}->")}
        n = sum(row.values())
        print(f"notated {fam:5s} alone ({n}): " + ", ".join(f"{k} {v / max(1, n):.0%}" for k, v in sorted(row.items(), key=lambda kv: -kv[1])[:6]))
    if a.json:
        a.json.write_text(json.dumps(dict(total), indent=1))


if __name__ == "__main__":
    main()
