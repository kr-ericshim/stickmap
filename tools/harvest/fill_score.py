"""Score the detector on fast fills only: runs of drum strokes the transcriber wrote close together.

A fill is a run of at least four notated kick/snare/tom moments, each within
`--gap` seconds of the last, with a tom in it. Inside those runs (accepted
systems only) this reports, besides per-drum precision and recall:

- strokes: whether a notated moment has any kick/snare/tom detection near it,
  whatever the drum (a miss here is a stroke the detector never heard);
- what the detector has where a tom or snare was notated (right drum, the
  other drum, nothing);
- tom size against the notated staff position.

    .venv/bin/python tools/harvest/fill_score.py [--gap 0.14] [--only swartz] [--jobs 4]
    .venv/bin/python tools/harvest/fill_score.py --detect mymodule:detect   # try a variant
    .venv/bin/python tools/harvest/fill_score.py --loose [--final --model-tag frozen-v5]
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "truth_video"))
sys.path.insert(0, str(Path(__file__).parent))

import harvest  # noqa: E402
import vault  # noqa: E402
import fill_seq  # noqa: E402
from align import match  # noqa: E402
from evaluate import family  # noqa: E402
from score import load, songs  # noqa: E402

DRUMS = ("kick", "snare", "tom")
SIZE = {3: "tom_low", 6: "tom", 7: "tom_high", 8: "tom_high"}  # notated staff position -> our tom size
FAST = 0.09  # seconds: a fill whose median gap is under this counts as very fast


def fills(truth: list[dict], intervals, gap: float, min_len: int = 4) -> list[tuple[float, float, float]]:
    """(start, end, median gap) of each notated fill inside the accepted systems."""
    inside = lambda t: any(a <= t <= b for a, b in intervals)
    moments: dict[float, set[str]] = defaultdict(set)
    for e in truth:
        if family(e["inst"]) in DRUMS and inside(e["time"]):
            moments[round(e["time"], 3)].add(family(e["inst"]))
    times = sorted(moments)
    runs, run = [], []
    for t in times:
        if run and t - run[-1] > gap:
            runs.append(run)
            run = []
        run.append(t)
    runs.append(run)
    out = []
    for run in runs:
        if len(run) >= min_len and any("tom" in moments[t] for t in run):
            out.append((run[0], run[-1], float(np.median(np.diff(run)))))
    return out


def tally(truth: list[dict], hits, windows) -> Counter:
    c: Counter = Counter()
    for a, b, speed in windows:
        lo, hi = a - 0.03, b + 0.03
        ref = [e for e in truth if family(e["inst"]) in DRUMS and lo <= e["time"] <= hi]
        est = [h for h in hits if family(h.inst) in DRUMS and lo <= h.time <= hi]
        tag = "fast" if speed < FAST else "fill"
        for key in ("all", tag):
            c[f"{key}.n"] += 1
            c[f"{key}.dur"] += b - a
        for fam in DRUMS:
            r = sorted({round(e["time"], 5) for e in ref if family(e["inst"]) == fam})
            s = sorted(h.time for h in est if family(h.inst) == fam)
            tp = len(match(r, s))
            for key in ("all", tag):
                c[f"{key}.{fam}.tp"] += tp
                c[f"{key}.{fam}.est"] += len(s)
                c[f"{key}.{fam}.ref"] += len(r)
        # Strokes, whatever the drum.
        r = sorted({round(e["time"], 3) for e in ref})
        s = sorted({round(h.time, 2) for h in est})
        tp = len(match(r, s, 0.04))
        for key in ("all", tag):
            c[f"{key}.stroke.tp"] += tp
            c[f"{key}.stroke.est"] += len(s)
            c[f"{key}.stroke.ref"] += len(r)
        times = np.array([h.time for h in est]) if est else np.empty(0)
        for e in ref:
            fam = family(e["inst"])
            if fam == "kick":
                continue
            near = {family(est[i].inst) for i in np.where(np.abs(times - e["time"]) <= 0.04)[0]} if len(times) else set()
            got = fam if fam in near else ("tom" if "tom" in near else "snare" if "snare" in near else "kick" if near else "nothing")
            c[f"conf.{fam}->{got}"] += 1
            if fam == "tom" and got == "tom":
                size = next(est[i].inst for i in np.where(np.abs(times - e["time"]) <= 0.04)[0] if family(est[i].inst) == "tom")
                c[f"size.{e['pos']}->{size}"] += 1
                if e["pos"] in SIZE:
                    c["size.ok"] += size == SIZE[e["pos"]]
                    c["size.n"] += 1
    return c


def one(args: tuple[dict, str, float, str]) -> tuple[str, Counter]:
    row, tier, gap, detector = args
    module, name = detector.split(":")
    detect = getattr(importlib.import_module(module), name)
    song = harvest.SONGS / row["slug"]
    try:
        truth, intervals = load(song / "truth", tier)
    except FileNotFoundError:
        return row["channel"], Counter()
    windows = fills(truth, intervals, gap) if truth else []
    if not windows:
        return row["channel"], Counter()
    c = tally(truth, detect(song), windows)
    c["songs"] += 1
    return row["channel"], c


def one_loose(args) -> tuple[str, Counter]:
    """Read-only order-DP scoring; includes clean timing-only rejected systems.

    Strict and timing-only systems enter this separate metric once each. No
    rejected timestamp is exported as truth or consumed by the trainer.
    """
    row, detector, tol, min_after = args
    song = harvest.SONGS / row["slug"]
    try:
        systems = fill_seq.notated(song / "truth", min_after)
    except FileNotFoundError:
        return row["channel"], Counter()
    if not systems:
        return row["channel"], Counter()
    module, name = detector.split(":")
    hits = getattr(importlib.import_module(module), name)(song)
    c: Counter = Counter()
    for a, b, ref in systems:
        c += fill_seq.tally(ref, fill_seq.detected(hits, a - 0.05, b + 0.05), tol)
        c["systems"] += 1
        c["seconds"] += b - a
    c["songs"] += 1
    return row["channel"], c


def write_totals(args, totals, excluded) -> None:
    if args.json:
        result = dict(totals)
        if args.final:
            result['excluded'] = excluded
        args.json.write_text(json.dumps(result, indent=1))
        if args.final:
            lines = [f"{v}: {item['reason']}" for v, item in sorted(excluded.items())]
            args.json.with_suffix('.md').write_text(
                '\n'.join(['# Final vault fill scores', '', *lines, '',
                           '```json', json.dumps(result, indent=1), '```', '']))


def loose_scores(args, rows, excluded=None) -> None:
    totals: dict[str, Counter] = defaultdict(Counter)
    with ProcessPoolExecutor(args.jobs) as pool:
        for channel, c in pool.map(one_loose, [(r, args.detect, args.tol, args.min_after) for r in rows]):
            totals["all"] += c
            totals[channel] += c
    t = totals["all"]
    print(f"LOOSE ORDER-DP (+/-{args.tol * 1000:g} ms): {t['songs']} songs, "
          f"{t['systems']} clean systems, {t['seconds']:.0f}s")
    print(f"all         | {fill_seq.line(t, 'all')}")
    print(f"very fast   | {fill_seq.line(t, 'fast')}")
    print(f"rest        | {fill_seq.line(t, 'slow')}")
    print("Very fast: nearest notated kick/snare/tom neighbour gap < 90 ms; timing is approximate.")
    if args.per_channel:
        for channel in sorted(k for k in totals if k != "all"):
            c = totals[channel]
            if c["songs"]:
                print(f"{channel:12s} {c['songs']:3d} songs | {fill_seq.line(c, 'all')}")
    write_totals(args, t, excluded or {})


def line(c: Counter, key: str) -> str:
    cells = [f"{c[f'{key}.n']:4d} fills {c[f'{key}.dur']:6.0f}s"]
    for fam in (*DRUMS, "stroke"):
        tp, est, ref = c[f"{key}.{fam}.tp"], c[f"{key}.{fam}.est"], c[f"{key}.{fam}.ref"]
        f = 2 * tp / max(1, est + ref)
        cells.append(f"{fam} P{tp / max(1, est):.2f} R{tp / max(1, ref):.2f} F{f:.2f} ({ref})")
    return " | ".join(cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["strict", "scoring"], default="strict")
    parser.add_argument("--gap", type=float, default=0.14)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--detect", default="evaluate:detect", help="module:function taking a song folder, returning hits")
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--json", type=Path, help="write the totals here")
    parser.add_argument("--final", action="store_true", help="open and score ONLY the sealed test vault (logged)")
    parser.add_argument("--model-tag", help="model identity for the vault opening log (default: FT_TAG or detector)")
    parser.add_argument("--opening-id", help="final runner's existing durable vault opening ID")
    parser.add_argument("--loose", action="store_true", help="score clean systems with fill_seq's order DP, including timing-only rejections")
    parser.add_argument("--tol", type=float, default=0.06, help="loose order-DP tolerance in seconds (default: 0.06)")
    parser.add_argument("--min-after", type=float, default=0.75, help="loose minimum fitted attack agreement within 40 ms")
    args = parser.parse_args()
    if not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("--tol must be positive and finite")
    if not np.isfinite(args.min_after) or not 0 <= args.min_after <= 1:
        parser.error("--min-after must be in [0, 1]")
    if args.opening_id and not args.final:
        parser.error("--opening-id requires --final")
    if args.final:
        tag = vault.model_tag(args.detect, args.model_tag)
        if args.opening_id:
            vault.resume_final(args.opening_id, tag)
        else:
            vault.open_final(tag)
    rows = songs(args.only, final=args.final, loose=args.loose)
    excluded = {}
    if args.final:
        rows, excluded = vault.filter_final(rows, log=False)
    if args.loose:
        loose_scores(args, rows, excluded)
        return
    totals: dict[str, Counter] = defaultdict(Counter)
    with ProcessPoolExecutor(args.jobs) as pool:
        for channel, c in pool.map(one, [(r, args.tier, args.gap, args.detect) for r in rows]):
            totals["all"] += c
            totals[channel] += c
    t = totals["all"]
    print(f"{t['songs']} songs with fills (gap <= {args.gap}s)")
    print(f"all fills   {line(t, 'all')}")
    print(f"gap>={FAST}s  {line(t, 'fill')}")
    print(f"gap<{FAST}s   {line(t, 'fast')}")
    for fam in ("tom", "snare"):
        row = {k.split("->")[1]: v for k, v in t.items() if k.startswith(f"conf.{fam}->")}
        n = sum(row.values())
        print(f"notated {fam:5s} ({n}): " + ", ".join(f"{k} {v / max(1, n):.0%}" for k, v in sorted(row.items(), key=lambda kv: -kv[1])))
    print(f"tom size vs notated position: {t['size.ok'] / max(1, t['size.n']):.0%} of {t['size.n']}")
    for pos in sorted({int(k.split('.')[1].split('->')[0]) for k in t if k.startswith("size.") and "->" in k}):
        row = {k.split("->")[1]: v for k, v in t.items() if k.startswith(f"size.{pos}->")}
        print(f"  pos {pos}: {row}")
    if args.per_channel:
        for key in sorted(k for k in totals if k != "all"):
            if totals[key]["songs"]:
                print(f"{key:12s} {totals[key]['songs']:3d} songs | {line(totals[key], 'all')}")
    write_totals(args, t, excluded)


if __name__ == "__main__":
    main()
