"""Build detector-free, positive-only dense-fill supervision beside data.py's cache.

    .venv/bin/python -B tools/finetune/dense.py

Only enabled, already cached harvested songs are considered. No audio/model is
downloaded or evaluated, and source caches are read-only. dense.json is NOT a
truth.json replacement: a label means at least one stroke of its class in its
inclusive [round(time * fps) - tolerance_frames, ... + tolerance_frames] window.
Unlabelled frames/classes are unknown, never negative training examples.

Quality compares identical windows on cached strict labels and dense labels.
The band test uses bands.py's peak/ratio rules; toms have only class-agnostic
attack evidence. These caches were used to fit/filter strict timing already,
so this is an acoustic sanity check, not an independent hand-timed audit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "truth_video"))
from align import moment_instruments, moments  # noqa: E402
import bands  # noqa: E402
from evaluate import family  # noqa: E402
from fill_seq import TIMING  # noqa: E402
from splits import split_songs, vault  # noqa: E402

SONGS = ROOT / "truth" / "harvest" / "songs"
CACHE = ROOT / ".eval-cache" / "ft"
CLASSES = ("kick", "snare", "tom")
FPS = 100
MAX_GAP = 0.14
FAST_GAP = 0.09
MIN_MOMENTS = 4
MIN_AFTER = 0.75
MIN_ASSIGNED = 0.8
# A substantial 20-point deficit, with >2 excess missed labels, is needed.
# The count allowance avoids rejecting a tiny fill over one noisy band onset.
MAX_DEFICIT = 0.20
EXCESS_MISSES = 2
MIN_REFERENCE = 20  # prefer song timbre; otherwise channel, then cohort


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    """Replace only this tool's output, atomically, on repeat runs."""
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def dense_runs(seq, gap: float = MAX_GAP, *, exclusive: bool = False) -> list:
    """Maximal runs of distinct drum moments, >=4 long and containing a tom.

    Hat/cymbal-only moments are removed before calling this function. Unisons
    count once. Fast runs require every successive gap strictly below 90 ms.
    """
    out, run = [], []
    for item in seq:
        delta = item[0] - run[-1][0] if run else 0.0
        separate = delta >= gap - 1e-9 if exclusive else delta > gap + 1e-9
        if run and separate:
            if len(run) >= MIN_MOMENTS and any("tom" in q[1] for q in run):
                out.append(run)
            run = []
        run.append(item)
    if len(run) >= MIN_MOMENTS and any("tom" in q[1] for q in run):
        out.append(run)
    return out


def overlaps(span, intervals) -> bool:
    return any(span[0] <= b and a <= span[1] for a, b in intervals)


def candidates(strip: list, alignment: dict, strict_intervals: list,
               tolerance_frames: int = 3) -> tuple[list[dict], Counter]:
    """Select whole timing-only systems, then label just their dense drum runs."""
    strips = {s["system"]: s for s in strip}
    stats: Counter = Counter()
    selected = []
    # Include strict systems even if a report's top-level interval list is absent.
    strict_intervals = list(strict_intervals) + list(alignment.get("intervals", []))
    strict_intervals += [[r["times"][0], r["times"][-1]]
                         for r in alignment["systems"] if r.get("usable") and r.get("times")]
    for r in alignment["systems"]:
        if r.get("usable"):
            continue
        stats["rejected_systems"] += 1
        reasons = r.get("reasons", [])
        if not reasons or not all(q.startswith(TIMING) for q in reasons):
            stats["non_timing_rejection"] += 1
            continue
        stats["timing_only_systems"] += 1
        after = (r.get("after") or {}).get("within_40ms", 0)
        assigned = r.get("assigned_fraction", 0)
        if not (MIN_AFTER <= after <= 1 and MIN_ASSIGNED <= assigned <= 1):
            stats["insufficient_timing"] += 1
            continue
        s = strips.get(r["system"])
        if s is None:
            stats["missing_strip"] += 1
            continue
        if s.get("staff", {}).get("reasons") or any(
            n.get("exportable") is False and family(n.get("inst", "")) in CLASSES
            for n in s["notes"]
        ):
            stats["unexportable_reading"] += 1
            continue
        _, groups = moments(s)
        times = np.asarray(r.get("times", []), float)
        if (len(times) != len(groups) or not len(times) or
                not np.isfinite(times).all() or times[0] < 0 or
                np.any(np.diff(times) <= 0)):
            stats["invalid_moment_times"] += 1
            continue
        # Exclude the entire system, even when only its sparse portion overlaps.
        # Protect both the source interval and the fitted span plus label windows.
        pad = (tolerance_frames + 0.5) / FPS  # round-to-frame adds at most half a frame
        if (overlaps([times[0] - pad, times[-1] + pad], strict_intervals) or
                (r.get("interval") and overlaps(r["interval"], strict_intervals))):
            stats["strict_overlap"] += 1
            continue
        names = moment_instruments(s, groups)
        seq = [(float(t), set(map(family, ns)) & set(CLASSES), i)
               for i, (t, ns) in enumerate(zip(times, names))]
        seq = [q for q in seq if q[1]]
        runs = dense_runs(seq)
        if not runs:
            stats["no_dense_run"] += 1
            continue
        fast_runs = dense_runs(seq, FAST_GAP, exclusive=True)
        fast_moments = {q[2] for run in fast_runs for q in run}
        labels, spans = [], []
        for run_id, run in enumerate(runs):
            spans.append({"run": run_id, "start": run[0][0], "end": run[-1][0],
                          "moments": len(run),
                          "max_gap_seconds": max(b[0] - a[0] for a, b in zip(run, run[1:]))})
            for t, cs, moment in run:
                for cls in CLASSES:
                    if cls in cs:
                        labels.append({"time": t, "class": cls,
                                       "tolerance_frames": tolerance_frames,
                                       "system": r["system"], "moment": moment, "run": run_id,
                                       "fast_run": moment in fast_moments})
        selected.append({"system": r["system"], "source_interval": r.get("interval"),
                         "fitted_interval": [float(times[0]), float(times[-1])],
                         "reasons": reasons, "after_within_40ms": after,
                         "assigned_fraction": assigned, "runs": spans,
                         "fast_runs": len(fast_runs), "labels": labels})
    stats["candidate_systems"] = len(selected)
    return selected, stats


def audio_evidence(song: Path) -> tuple[np.ndarray, dict]:
    """Read existing caches only; bands.load could recompute a missing cache."""
    attacks = np.load(song / "attacks.htdemucs.npy", allow_pickle=False)
    data = np.load(song / bands.CACHE, allow_pickle=False)
    if (attacks.ndim != 1 or not len(attacks) or not np.isfinite(attacks).all() or
            np.any(np.diff(attacks) < 0)):
        raise ValueError("invalid or empty attack cache")
    if (data.ndim != 2 or data.shape[0] != len(bands.ORDER) + 2 or data.shape[1] < 2 or
            not np.isfinite(data).all() or not np.all(data[0] == bands.VERSION) or
            np.any(np.diff(data[1]) <= 0)):
        raise ValueError("invalid band cache/version")
    out = {"t": data[1], **{name: data[i + 2] for i, name in enumerate(bands.ORDER)}}
    for name in bands.ORDER:
        out[name + "_peaks"] = bands.peaks(out[name], out["t"])
    return attacks, out


def in_window(times, attacks, tolerance_frames: int) -> np.ndarray:
    """Any attack within the SAME quantized inclusive frame window as the loss."""
    frames = np.rint(np.asarray(times, float) * FPS)
    lo, hi = (frames - tolerance_frames) / FPS, (frames + tolerance_frames) / FPS
    idx = np.searchsorted(attacks, lo - 1e-9)
    valid = idx < len(attacks)
    hits = np.zeros(len(frames), bool)
    hits[valid] = attacks[idx[valid]] <= hi[valid] + 1e-9
    return hits


def band_window(audio: dict, times, cls: str, tolerance_frames: int) -> np.ndarray:
    """Exact window boundaries, with bands.RULES (peak and cross-band ratio).

    bands.window_max rounds the radius to hops and can include an extra hop;
    searching both edges keeps our check inside the trainer's declared window.
    """
    rule = bands.RULES[cls]
    out = []
    for frame in np.rint(np.asarray(times, float) * FPS):
        lo = np.searchsorted(audio["t"], (frame - tolerance_frames) / FPS, side="left")
        hi = np.searchsorted(audio["t"], (frame + tolerance_frames) / FPS, side="right")
        if hi <= lo:
            out.append(False)
            continue
        own = audio[rule["band"]][lo:hi].max()
        peak = audio[rule["band"] + "_peaks"][lo:hi].max()
        other = audio[rule["other"]][lo:hi].max()
        out.append(bool(peak > rule["peak"] and own >= rule["ratio"] * other))
    return np.asarray(out, bool)


def agreement(labels: list, attacks: np.ndarray, audio: dict,
              tolerance_frames: int) -> dict:
    out = {}
    for cls in CLASSES:
        times = [q["time"] for q in labels if q["class"] == cls]
        attack = in_window(times, attacks, tolerance_frames)
        band = band_window(audio, times, cls, tolerance_frames) if cls in bands.RULES else None
        out[cls] = {"labels": len(times), "attack_hits": int(attack.sum()),
                    "band_hits": int(band.sum()) if band is not None else None,
                    "both_hits": int((attack & band).sum()) if band is not None else None}
    return out


def pool(items) -> dict:
    out = {c: {"labels": 0, "attack_hits": 0,
               "band_hits": 0 if c in bands.RULES else None,
               "both_hits": 0 if c in bands.RULES else None} for c in CLASSES}
    for item in items:
        for cls in CLASSES:
            for key in out[cls]:
                if out[cls][key] is not None:
                    out[cls][key] += item[cls][key]
    return out


def rates(stats: dict) -> dict:
    return {cls: {**q, **{key.replace("_hits", "_rate"):
                         q[key] / q["labels"] if q[key] is not None and q["labels"] else None
                         for key in ("attack_hits", "band_hits", "both_hits")}}
            for cls, q in stats.items()}


def reference(song: dict, channel: dict, cohort: dict) -> dict:
    out = {}
    for cls in CLASSES:
        for name, stats in (("song", song), ("channel", channel), ("cohort", cohort)):
            if stats[cls]["labels"] >= MIN_REFERENCE or name == "cohort":
                out[cls] = {"source": name, **rates(stats)[cls]}
                break
    return out


def fold_quality(songs: list[dict], fold: str, deficit: float = MAX_DEFICIT) -> tuple[list[dict], dict]:
    """Select positives with strict evidence from eligible training songs only.

    Excluding same-song partners before pooling matters even when their channel
    is otherwise a training channel. Return fresh records for each fold.
    """
    training, held = split_songs(songs, fold)
    eligible = {s["slug"] for s in training}
    cohort = pool([s["strict"] for s in training])
    channels = {name: pool([s["strict"] for s in training if s["channel"] == name])
                for name in {s["channel"] for s in training}}
    selected = []
    for song in songs:
        retained, rejected = [], []
        baseline = None
        if song["slug"] in eligible:
            baseline = reference(song["strict"], channels[song["channel"]], cohort)
            for candidate in song["candidates"]:
                s = {**candidate, "quality_reasons": quality_reasons(candidate["agreement"], baseline, deficit)}
                (rejected if s["quality_reasons"] else retained).append(s)
        selected.append({**song, "retained": retained, "quality_rejected": rejected,
                         "quality_reference": baseline, "training_eligible": song["slug"] in eligible})
    return selected, {"quality_fold": fold, "reference_scope": "training-only",
                      "training_slugs": sorted(eligible), "held_slugs": sorted(s["slug"] for s in held),
                      "excluded_same_song_slugs": sorted(s["slug"] for s in songs
                          if s["slug"] not in eligible and s not in held),
                      "strict": rates(cohort)}


def save_labels(songs: list[dict], scope: dict, tolerance: int, deficit: float) -> None:
    for song in songs:
        retained = song["retained"]
        labels = sorted([q for s in retained for q in s["labels"]], key=lambda q: (q["time"], q["class"]))
        write_json(song["out"] / f"dense-fold{scope['quality_fold']}.json", {
            "version": 2, "tier": "dense", "positive_only": True,
            "slug": song["slug"], "channel": song["channel"], "fps": FPS,
            "quality_fold": scope["quality_fold"], "reference_scope": scope["reference_scope"],
            "training_eligible": song["training_eligible"],
            "tolerance_frames": tolerance, "window_center": "round(time * fps)",
            "status": song["status"], "labels": labels,
            "quality_policy": {"max_agreement_deficit": deficit,
                               "excess_misses_allowance": EXCESS_MISSES,
                               "reject_if": "rate deficit exceeds BOTH the fraction and count allowances"},
            # No intervals: these remain positive bags, not new negative regions.
            "systems": [{k: v for k, v in s.items() if k != "labels"} for s in retained],
            "strict_agreement": rates(song["strict"]), "quality_reference": song["quality_reference"],
            "quality_rejected": [{k: v for k, v in s.items() if k not in ("labels", "runs")}
                                 for s in song["quality_rejected"]],
        })


def quality_reasons(stats: dict, baseline: dict, deficit: float = MAX_DEFICIT) -> list[str]:
    """Reject a system on any class or class-mix-adjusted pooled evidence.

    A deficit must exceed both 20 percentage points and two expected hits.
    Expected pooled rates use the system's own class mix to avoid tom dilution
    or an inappropriate kick-vs-snare baseline. 'Both' checks the intersection
    as well as marginal attack and band agreement.
    """
    reasons = []
    for metric in ("attack", "band", "both"):
        total = hits = expected = 0
        for cls in CLASSES:
            q = stats[cls]
            if not q["labels"] or q[metric + "_hits"] is None:
                continue
            rate = baseline[cls][metric + "_rate"]
            if rate is None:
                reasons.append(f"no strict {cls} {metric} reference")
                continue
            n, h = q["labels"], q[metric + "_hits"]
            shortfall = rate * n - h
            if shortfall > deficit * n + 1e-9 and shortfall > EXCESS_MISSES + 1e-9:
                reasons.append(f"{cls} {metric}: {h}/{n} vs strict {rate:.4f}")
            total += n
            hits += h
            expected += rate * n
        shortfall = expected - hits
        if total and shortfall > deficit * total + 1e-9 and shortfall > EXCESS_MISSES + 1e-9:
            reasons.append(f"pooled {metric}: {hits}/{total} vs class-matched strict {expected / total:.4f}")
    return reasons


def union_seconds(spans: list) -> float:
    end, total = -math.inf, 0.0
    for a, b in sorted(spans):
        total += max(0.0, b - max(a, end))
        end = max(end, b)
    return total


def counts(songs: list[dict], key: str) -> dict:
    systems = runs = fast_runs = 0
    seconds = system_seconds = 0.0
    labels, fast_labels = Counter(), Counter()
    song_count = 0
    for song in songs:
        chosen = song[key]
        song_count += bool(chosen)
        systems += len(chosen)
        runs += sum(len(s["runs"]) for s in chosen)
        fast_runs += sum(s["fast_runs"] for s in chosen)
        seconds += union_seconds([[r["start"], r["end"]] for s in chosen for r in s["runs"]])
        system_seconds += union_seconds([s["fitted_interval"] for s in chosen])
        for s in chosen:
            for q in s["labels"]:
                labels[q["class"]] += 1
                fast_labels[q["class"]] += q["fast_run"]
    return {"songs": song_count, "systems": systems, "runs": runs,
            "seconds": round(seconds, 6), "system_span_seconds": round(system_seconds, 6),
            "labels": {c: labels[c] for c in CLASSES}, "fast_runs": fast_runs,
            "fast_labels": {c: fast_labels[c] for c in CLASSES}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance-frames", type=int, default=3)
    parser.add_argument("--max-agreement-deficit", type=float, default=MAX_DEFICIT)
    parser.add_argument("--fold", choices=("0", "1", "2", "all"),
                        help="default regenerates 0, 1, 2 and all; legacy dense.json is preserved")
    args = parser.parse_args()
    if args.tolerance_frames < 1 or not 0 < args.max_agreement_deficit < 1:
        parser.error("tolerance must be >=1 frame and deficit between 0 and 1")
    songs, skipped = [], Counter()
    seal = vault.Vault()
    # Every existing training-song output is replaced, including empty outputs,
    # so a new exclusion or disabled legend cannot leave stale training labels.
    for song in sorted(SONGS.iterdir()):
        if not song.is_dir():
            continue
        out = CACHE / song.name
        if not out.is_dir() or not (out / "truth.json").is_file() or not (out / "feat.npy").is_file():
            skipped["not_cached"] += 1
            continue
        meta = read_json(out / "truth.json")
        row = {"slug": song.name, "video": meta.get("video"), "video_id": meta.get("video_id"),
               "channel": meta["channel"], "out": out,
               "candidates": [], "retained": [], "selection": {}, "strict": pool([]),
               "status": "ok"}
        songs.append(row)
        # Check both identities before reading notation or acoustic evidence.
        # Keep an empty sealed row to clear stale fold labels on regeneration.
        canonical = {**(read_json(song / "meta.json") if (song / "meta.json").exists() else {}),
                     "slug": song.name}
        if seal.contains_song(row) or seal.contains_song(canonical):
            row["status"] = "sealed_vault"
            # Canonical metadata may be the only source of the sealed identity.
            # Preserve it for split_songs(), which uses the shared membership API.
            if not seal.contains_song(row):
                row.update(video=canonical.get("video"), video_id=canonical.get("video_id"))
            skipped[row["status"]] += 1
            continue
        try:
            legend = read_json(song / "truth" / "legend.json")
            if legend.get("enabled") is not True:
                row["status"] = "disabled_legend"
                skipped[row["status"]] += 1
                continue
            strip = read_json(song / "truth" / "strip.json")
            alignment = read_json(song / "truth" / "alignment.json")
            attacks, audio = audio_evidence(song)
            strict_labels = [{"time": t, "class": c} for c in CLASSES for t in meta["events"][c]]
            row["strict"] = agreement(strict_labels, attacks, audio, args.tolerance_frames)
            row["strict_seconds"] = union_seconds(meta["intervals"])
            chosen, stats = candidates(strip, alignment, meta["intervals"], args.tolerance_frames)
            row["selection"] = dict(stats)
            n_frames = len(np.load(out / "feat.npy", mmap_mode="r", allow_pickle=False))
            for s in chosen:
                # Never clip a supervision window into the audio or a strict interval.
                if any(round(q["time"] * FPS) - args.tolerance_frames < 0 or
                       round(q["time"] * FPS) + args.tolerance_frames >= n_frames for q in s["labels"]):
                    row["selection"]["window_outside_features"] = row["selection"].get("window_outside_features", 0) + 1
                    continue
                s["agreement"] = agreement(s["labels"], attacks, audio, args.tolerance_frames)
                row["candidates"].append(s)
        except (OSError, ValueError, KeyError) as exc:
            row["status"] = f"unavailable: {exc}"
            row["candidates"] = []
            row["strict"] = pool([])
            skipped["missing_or_invalid_inputs"] += 1
            print(f"skip {song.name}: {exc}", file=sys.stderr)

    folds = {}
    original_songs = songs
    for fold in (args.fold,) if args.fold else ("0", "1", "2", "all"):
        songs, scope = fold_quality(original_songs, fold, args.max_agreement_deficit)
        save_labels(songs, scope, args.tolerance_frames, args.max_agreement_deficit)
        eligible_songs = [s for s in songs if s["training_eligible"]]
        folds[fold] = {**scope, "before_quality": counts(eligible_songs, "candidates"),
                       "retained": counts(eligible_songs, "retained"),
                       "quality_rejected_systems": sum(len(s["quality_rejected"]) for s in songs)}
        print(f"fold {fold}: {len(scope['training_slugs'])} training songs; "
              f"dense {folds[fold]['retained']}")
    # Detailed aggregate diagnostics describe the last generated scope (all by default).
    cohort = pool([s["strict"] for s in songs if s["training_eligible"]])
    candidate_stats = pool([s["agreement"] for song in songs if song["training_eligible"] for s in song["candidates"]])
    retained_stats = pool([s["agreement"] for song in songs for s in song["retained"]])
    summary = {
        "version": 2, "cached_songs": len(songs), "skipped": dict(skipped), "folds": folds,
        "aggregate_fold": scope["quality_fold"],
        "policy": {"fps": FPS, "tolerance_frames": args.tolerance_frames,
                   "min_after_within_40ms": MIN_AFTER, "min_assigned_fraction": MIN_ASSIGNED,
                   "timing_reason_prefixes": TIMING, "max_gap_seconds": MAX_GAP,
                   "min_moments": MIN_MOMENTS, "requires_tom": True,
                   "fast_run": "at least 4 drum moments, with a tom; every gap < 0.09 s",
                   "seconds": "union of dense-run first-to-last moment spans, without tolerance padding",
                   "overlap": "whole source interval OR fitted system span including rounded label windows",
                   "max_agreement_deficit": args.max_agreement_deficit,
                   "excess_misses_allowance": EXCESS_MISSES,
                   "reference": f"same song with >= {MIN_REFERENCE} strict class labels, else channel, else cohort",
                   "quality": "reject whole system if any class or class-matched pool exceeds BOTH deficit and excess-miss allowance, for attack, band, or their intersection",
                   "evidence": "existing stem attack cache and bands.py peak/ratio rules; no detector labels",
                   "limitation": "reuse of alignment evidence; band rules are not physical class truth; no tom-specific band test"},
        "selection": dict(sum((Counter(s["selection"]) for s in eligible_songs), Counter())),
        "before_quality": counts(eligible_songs, "candidates"), "retained": counts(eligible_songs, "retained"),
        "quality_rejected_systems": sum(len(s["quality_rejected"]) for s in songs),
        "agreement": {"strict": rates(cohort), "dense_before_quality": rates(candidate_stats),
                      "dense": rates(retained_stats),
                      "strict_on_dense_songs": rates(pool([s["strict"] for s in songs if s["retained"]]))},
        "strict_seconds": round(sum(s.get("strict_seconds", 0) for s in eligible_songs), 6),
        "songs": [{"slug": s["slug"], "channel": s["channel"], "status": s["status"],
                   "training_eligible": s["training_eligible"],
                   "selection": s["selection"], "before_quality": counts([s], "candidates"),
                   "retained": counts([s], "retained")} for s in songs],
    }
    summary_path = CACHE / (f"dense-fold{args.fold}.summary.json" if args.fold else "dense.folds.summary.json")
    write_json(summary_path, summary)
    print(json.dumps({k: summary[k] for k in ("cached_songs", "skipped", "selection", "before_quality", "retained", "quality_rejected_systems")}, indent=2))
    print("\nclass  tier                 labels   attacks     band       both")
    for cls in CLASSES:
        for tier in ("strict", "dense_before_quality", "dense"):
            q = summary["agreement"][tier][cls]
            fmt = lambda k: f"{q[k]:.2%}" if q[k] is not None else "n/a"
            print(f"{cls:6s} {tier:20s} {q['labels']:6d}   {fmt('attack_rate'):9s}  {fmt('band_rate'):9s}  {fmt('both_rate')}")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
