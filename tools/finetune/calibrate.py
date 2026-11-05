"""Cached, one-class-at-a-time threshold sweeps; no inference or training.

Uses ft_variants.full and the existing global/fill/sequence tally functions.
For family metrics only, tom sizing is replaced in memory by its exact stray
filter: pitch naming changes no family or time. Peak picking is memoized per
activation column/threshold; default integer totals must match retained reports.
All scratch/output goes to results/calibration; at most four CPU workers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools" / "finetune"),
               str(ROOT / "tools" / "harvest"), str(ROOT / "tools" / "truth_video")]
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import fill_score
import fill_seq
import ft_variants
import harvest
import score

ONLY = ("swartz", "allyeodrum", "drumclass", "mydrumsheet", "taktak", "drumshin",
        "drumplace", "nakamura", "chock", "yeol", "atthedrum", "jilju", "drummate", "drumscribe")
CLASSES = ("kick", "snare", "tom")
SCALES = (.25, .5, .75, .9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2., 2.25, 2.5, 3., 4., 6.)
GROUPS = (("global", None), ("covers", "all"), ("covers", "fast"),
          ("solos", "all"), ("solos", "fast"))


def items():
    out = []
    for row in score.songs(list(ONLY)):
        song = harvest.SONGS / row["slug"]
        try:
            truth, intervals = score.load(song / "truth", "strict")
        except FileNotFoundError:
            continue
        if truth:
            out.append({"kind": "harvest", "song": str(song), "channel": row["channel"],
                        "truth": truth, "intervals": intervals})
    for video in fill_seq.SOLOS:
        folder = fill_seq.TRUTH / video
        song = Path((folder / "song.txt").read_text().strip())
        systems = fill_seq.notated(folder, .75)
        if systems and (song / "activations.drums-htdemucs.npy").exists():
            out.append({"kind": "solo", "song": str(song), "channel": video, "systems": systems})
    return out


def integer_metrics(hits, item):
    if item["kind"] == "harvest":
        truth, intervals = item["truth"], item["intervals"]
        global_counts = score.tally(truth, hits, intervals)
        windows = fill_score.fills(truth, intervals, .14)
        c = fill_score.tally(truth, hits, windows)
        return {"global": {cls: list(global_counts[cls]) for cls in CLASSES},
                "covers": {f"{key}.{cls}.{field}": int(c[f"{key}.{cls}.{field}"])
                           for key in ("all", "fast") for cls in CLASSES for field in ("tp", "est", "ref")}}
    c = Counter()
    for a, b, ref in item["systems"]:
        # JSON/resume makes sets into lists; restore the scorer's set operations.
        ref = [(t, frozenset(drums)) for t, drums in ref]
        c += fill_seq.tally(ref, fill_seq.detected(hits, a - .05, b + .05), .06)
    return {"solos": {f"{key}.{cls}.{field}": int(c[f"{key}.{cls}.{field}"])
                      for key in ("all", "fast") for cls in CLASSES for field in ("tp", "est", "ref")}}


def configurations(scales):
    return {"default": (None, 1.),
            **{f"{cls}:{scale:g}": (cls, scale) for cls in CLASSES for scale in scales if scale != 1.}}


def one(task):
    from stickmap import models, toms
    originals = (np.load, ft_variants.snare, toms._headless, toms.tom_sizes, models.picked_frames)
    try:
        return _one(task)
    finally:
        np.load, ft_variants.snare, toms._headless, toms.tom_sizes, models.picked_frames = originals


def _one(task):
    item, tags, scales, output = task
    from stickmap import models, toms
    import adtof_pytorch as adtof
    import torch

    torch.set_num_threads(1)
    song = Path(item["song"])
    folder = ft_variants.CACHE / song.name
    if not folder.exists():
        folder = ft_variants.CACHE / "solos" / song.name
    saved = Path(output) / "songs" / f"{song.name}.json"
    data = json.loads(saved.read_text()) if saved.exists() else {"song": str(song), "channel": item["channel"], "tags": {}}
    configs = configurations(scales)
    missing = {tag: [key for key in configs if key not in data["tags"].get(tag, {})] for tag in tags}
    if not any(missing.values()):
        return data, 0

    # No snare/audio cache creation or neural inference is allowed here.
    mix = np.load(song / "activations.mix.npy")
    attacks = np.load(song / "attacks.htdemucs.npy")
    responses = {tag: np.load(folder / f"snare-student-{tag}.npy") for tag in tags if missing[tag]}
    old_load = np.load
    def cached_load(path, *args, **kw):
        if Path(path) == song / "activations.mix.npy":
            return mix
        if Path(path) == song / "attacks.htdemucs.npy":
            return attacks
        return old_load(path, *args, **kw)
    np.load = cached_load
    ft_variants.snare = lambda _song, _act, name: responses[name.removeprefix("student-")]

    # Keep tom_sizes' song-level kit exception and headless removal unchanged.
    # Naming is the expensive remainder, and cannot alter these family totals.
    stem = None
    headless = {}
    original_headless = toms._headless
    def cached_headless(stem, time):
        if time not in headless:
            headless[time] = original_headless(stem, time)
        return headless[time]
    toms._headless = cached_headless
    def family_toms(path, hits):
        nonlocal stem
        if not any(h.inst == "tom" for h in hits):
            return hits
        if stem is None:
            stem = toms._Stem(path)
        return toms._without_strays(stem, hits)
    toms.tom_sizes = family_toms

    # PeakPicker is independent per output column. Hash actual column bytes,
    # avoiding pointer reuse when full() constructs a fresh stem/mix blend.
    peak_cache = {}
    original_picked = models.picked_frames
    def cached_picked(activations, thresholds=None):
        thresholds = list(thresholds or models.DEFAULT_THRESHOLDS)
        result = {}
        for column, (label, cls) in enumerate(zip(adtof.LABELS_5, models.ADTOF_CLASSES)):
            vector = activations[:, column]
            threshold = thresholds[column]
            key = (hashlib.blake2b(vector.tobytes(), digest_size=16).digest(), float(threshold))
            if key not in peak_cache:
                picker = adtof.PeakPicker(thresholds=[threshold], fps=models.ADTOF_FPS)
                times = picker.pick(vector[None, :, None], labels=[label])[0][label]
                picked = [min(int(round(t * models.ADTOF_FPS)), len(vector) - 1) for t in times]
                peak_cache[key] = np.array(models._with_fast_strokes(vector, picked, threshold), dtype=int)
            result[cls] = peak_cache[key]
        return result
    models.picked_frames = cached_picked
    for inst in models.ADTOF_CLASSES:
        os.environ.pop(f"FT_{inst.upper()}_SCALE", None)

    count = 0
    for tag in tags:
        if not missing[tag]:
            continue
        ft_variants.TAG = tag
        act = ft_variants.student(song)
        if item["kind"] == "harvest":
            found = sorted(folder.glob(f"student-{tag}-fold?.npy"))
            if len(found) != 1:
                raise ValueError(f"{song.name}: expected one held-out {tag} prediction, found {found}")
            metadata = json.loads((ft_variants.CACHE / "models" / f"{found[0].stem.removeprefix('student-')}.json").read_text())
            if song.name not in metadata["held"] or song.name in metadata.get("train", []):
                raise ValueError(f"{song.name}: held-out provenance mismatch ({tag})")
        elif len(list(folder.glob(f"student-{tag}-fold?.npy"))) != 3:
            raise ValueError(f"{song.name}: solo needs all three {tag} folds")
        # Assert the memoized picker against the original on stem and blend.
        n = min(len(act), len(mix))
        for array in (act[:n], (act[:n] + mix[:n]) / 2):
            expected = original_picked(array)
            actual = cached_picked(array)
            for cls in models.ADTOF_CLASSES:
                np.testing.assert_array_equal(actual[cls], expected[cls])
        tag_data = data["tags"].setdefault(tag, {})
        for key in missing[tag]:
            cls, scale = configs[key]
            if cls:
                os.environ[f"FT_{cls.upper()}_SCALE"] = str(scale)
            hits = ft_variants.full(song, act, f"student-{tag}")
            tag_data[key] = integer_metrics(hits, item)
            if cls:
                os.environ.pop(f"FT_{cls.upper()}_SCALE")
            count += 1
        saved.write_text(json.dumps(data, indent=1))
    return data, count


def aggregate(results, tags, scales):
    totals = {tag: {key: {"global": {cls: [0, 0, 0] for cls in CLASSES},
                         "covers": Counter(), "solos": Counter()} for key in configurations(scales)} for tag in tags}
    for result in results:
        for tag in tags:
            for key in configurations(scales):
                for scorer, counts in result["tags"][tag][key].items():
                    if scorer == "global":
                        for cls in CLASSES:
                            totals[tag][key][scorer][cls] = [a + b for a, b in zip(totals[tag][key][scorer][cls], counts[cls])]
                    else:
                        totals[tag][key][scorer].update(counts)
    for tag in tags:
        reference_dir = ft_variants.CACHE / "results" / ("v3" if tag == "v3" else "v4")
        for scorer, key in GROUPS:
            old = json.loads((reference_dir / f"{tag}-{scorer}.json").read_text())
            new = totals[tag]["default"][scorer]
            for cls in CLASSES:
                if scorer == "global":
                    assert new[cls] == old[cls], (tag, scorer, cls, new[cls], old[cls])
                else:
                    for field in ("tp", "est", "ref"):
                        name = f"{key}.{cls}.{field}"
                        assert new[name] == old[name], (tag, scorer, name, new[name], old[name])
    return totals


def metrics(counts, scorer, key, cls):
    tp, est, ref = counts[cls] if scorer == "global" else [counts.get(f"{key}.{cls}.{field}", 0) for field in ("tp", "est", "ref")]
    return {"tp": tp, "est": est, "ref": ref, "p": tp / max(1, est), "r": tp / max(1, ref), "f1": 2 * tp / max(1, est + ref)}


def operating_points(totals, tags, scales):
    rows = []
    for scorer, key in GROUPS:
        for cls in CLASSES:
            points = {tag: [{"scale": scale, **metrics(totals[tag]["default" if scale == 1 else f"{cls}:{scale:g}"][scorer], scorer, key, cls)}
                            for scale in scales] for tag in tags}
            target = metrics(totals["v2"]["default"][scorer], scorer, key, cls)["p"]
            matched, best = {}, {}
            for tag in tags:
                eligible = [p for p in points[tag] if p["est"] and p["p"] >= target]
                matched[tag] = max(eligible, key=lambda p: (p["r"], p["p"], -abs(p["scale"] - 1))) if eligible else None
                best[tag] = max(points[tag], key=lambda p: (p["f1"], -abs(p["scale"] - 1)))
            rows.append({"scorer": scorer, "key": key, "class": cls, "target_p": target,
                         "matched": matched, "best_f1": best, "points": points})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    p.add_argument("--tags", nargs="+", default=["v2", "v3", "v4"])
    p.add_argument("--scales", nargs="+", type=float, default=list(SCALES))
    p.add_argument("--output", type=Path, default=ft_variants.CACHE / "results" / "calibration")
    a = p.parse_args()
    if "v2" not in a.tags or any(not np.isfinite(s) or s <= 0 for s in a.scales):
        p.error("v2 baseline and finite positive scales are required")
    scales = sorted(set([1., *a.scales]))
    selected = items()
    (a.output / "songs").mkdir(parents=True, exist_ok=True)
    print(f"{len(selected)} songs; {len(configurations(scales))} configurations x {a.tags}; {a.jobs} CPU workers", flush=True)
    start = time.monotonic()
    results = []
    with ProcessPoolExecutor(a.jobs) as pool:
        futures = [pool.submit(one, (item, a.tags, scales, str(a.output))) for item in selected]
        for index, future in enumerate(as_completed(futures), 1):
            result, count = future.result()
            results.append(result)
            print(f"{index}/{len(selected)} {Path(result['song']).name}: {count} new scores; {(time.monotonic() - start) / 60:.1f} min", flush=True)
    totals = aggregate(results, a.tags, scales)
    rows = operating_points(totals, a.tags, scales)
    payload = {"scales": scales, "tags": a.tags, "channels": list(ONLY), "jobs": a.jobs,
               "songs": {kind: sum(i["kind"] == kind for i in selected) for kind in ("harvest", "solo")},
               "baseline_exact": True, "seconds": time.monotonic() - start, "totals": totals, "rows": rows}
    (a.output / "sweep.json").write_text(json.dumps(payload, indent=1))
    print("Exact default integer totals reproduced for all three scorers/tags.", flush=True)
    for row in rows:
        print(row["scorer"], row["key"], row["class"], f"P>={row['target_p']:.4f}",
              {tag: None if cell is None else (cell["scale"], round(cell["p"], 4), round(cell["r"], 4)) for tag, cell in row["matched"].items()})


if __name__ == "__main__":
    main()
