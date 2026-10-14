"""Score the detector on every harvested song whose notation was read, pooled per channel.

Only notes inside accepted systems count, and a detection outside them is not a
false positive (see tools/truth_video/evaluate.py). The truth marks what the
transcriber wrote; unmarked ghost notes or articulations are not negatives.

    .venv/bin/python tools/harvest/score.py [--tier scoring] [--only atthedrum yeol]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "truth_video"))
sys.path.insert(0, str(Path(__file__).parent))

import harvest  # noqa: E402
import vault  # noqa: E402
from align import match  # noqa: E402
from evaluate import detect, family  # noqa: E402

FAMILIES = ("kick", "snare", "tom", "hihat", "cymbal")


def songs(only: list[str] | None, final: bool = False, loose: bool = False) -> list[dict]:
    seal = vault.Vault()
    statuses = ("read", "compacted", "unreadable", "fetched") if loose else ("read", "compacted")
    rows = [r for r in seal.rows.values() if r.get("status") in statuses
            and (not only or r["channel"] in only) and seal.contains_song(r) == final]
    return vault.filter_final(rows, seal=seal)[0] if final else rows


def load(folder: Path, tier: str) -> tuple[list[dict], list[list[float]]]:
    suffix = ".scoring" if tier == "scoring" else ""
    truth = json.loads((folder / f"truth{suffix}.json").read_text())
    report = json.loads((folder / f"alignment{suffix}.json").read_text())
    return truth, report.get("intervals") or ([[truth[0]["time"], truth[-1]["time"]]] if truth else [])


def tally(truth: list[dict], hits, intervals) -> dict:
    inside = lambda t: any(a - 0.05 <= t <= b + 0.05 for a, b in intervals)
    out = {}
    for c in FAMILIES:
        ref = list({round(e["time"], 5): e for e in truth if family(e["inst"]) == c}.values())
        est = [h for h in hits if family(h.inst) == c and inside(h.time)]
        pairs = match([e["time"] for e in ref], [h.time for h in est])
        out[c] = (len(pairs), len(est), len(ref))
        if c == "snare":
            out["ghost"] = (sum(1 for i, j in pairs if ref[i]["ghost"] and est[j].ghost),
                            sum(1 for i, j in pairs if est[j].ghost), sum(1 for e in ref if e["ghost"]))
        if c == "tom":
            out["tom_pos"] = Counter((ref[i]["pos"], est[j].inst) for i, j in pairs)
    return out


def main(model_tag: str | None = None, command: str | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tier", choices=["strict", "scoring"], default="strict")
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--final", action="store_true", help="open and score ONLY the sealed test vault (logged)")
    parser.add_argument("--model-tag", help="model identity for the vault opening log (default: FT_TAG or detector)")
    args = parser.parse_args()
    if args.final:
        vault.open_final(vault.model_tag("evaluate:detect", model_tag or args.model_tag), command)
    totals: dict[str, Counter] = defaultdict(Counter)
    positions: dict[str, Counter] = defaultdict(Counter)
    for row in songs(args.only, final=args.final):
        song = harvest.SONGS / row["slug"]
        try:
            truth, intervals = load(song / "truth", args.tier)
        except FileNotFoundError:
            continue
        if not truth:
            continue
        result = tally(truth, detect(song), intervals)
        for key in ("all", row["channel"]):
            for c in (*FAMILIES, "ghost"):
                tp, est, ref = result[c]
                totals[key][f"{c}.tp"] += tp
                totals[key][f"{c}.est"] += est
                totals[key][f"{c}.ref"] += ref
            totals[key]["songs"] += 1
            positions[key] += result["tom_pos"]
    for key in sorted(totals, key=lambda k: (k != "all", k)):
        t = totals[key]
        cells = []
        for c in FAMILIES:
            tp, est, ref = t[f"{c}.tp"], t[f"{c}.est"], t[f"{c}.ref"]
            cells.append(f"{c} P{tp / max(1, est):.2f} R{tp / max(1, ref):.2f} ({ref})")
        print(f"{key:12s} {t['songs']:3d} songs | " + " | ".join(cells))
    print("\ntom staff position (truth) -> our tom size, all songs:")
    for pos in sorted({p for p, _ in positions["all"]}):
        row = {inst: n for (p, inst), n in positions["all"].items() if p == pos}
        print(f"  pos {pos}: {row}")


if __name__ == "__main__":
    main()
