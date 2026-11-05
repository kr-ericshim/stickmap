"""One cached-only first-pass evaluation; no inference, audio loading or writes to songs.

Run: .venv/bin/python tools/bar_phase_eval.py
The output JSON includes candidate times and feature contributions for inspection.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

import numpy as np

from stickmap.bar_phase import DEFAULT_WEIGHTS, bar_start_times, phase_features, rank_bar_phases
from stickmap.pipeline import Options, detect_hits
from stickmap.quantize import _boundaries, _recount, _rescale, _snare_parity, _steady_beats

TOLERANCE_BEATS = .10
REQUIRED_AGREEMENT = .90
MARGINS = (.02, .05, .10)
TRUTH_NOTE = (
    "Bounded inspection of truth/*/{alignment.scoring,strip,truth.scoring}.json, "
    "tools/truth_video/{README.md,heads.py,strip.py,align.py}: timed note heads, "
    "but no directly usable validated bar/downbeat timestamps. README warns "
    "bar-line candidates can be stems. cj-thompson truth.scoring.json is empty. "
    "No new video decoding or bar-line extraction; zero extra truth labels."
)


def baseline_grid(hits, cached, opts):
    """Exactly the legacy pre-display grid construction at beat_shift=0.

    Uses detect_hits outputs only; deliberately does not run pipeline.run or
    sound renaming, waveform tom sizing, fill_snares or display refinement.
    Saved-score compatibility is measured rather than assumed.
    """
    beats = np.asarray(_rescale(_recount(hits, _steady_beats(cached["beats"])), cached["downbeats"], opts.tempo_scale))
    boundaries = _boundaries(beats, cached["downbeats"], opts.meter, opts.odd_bars, 0, _snare_parity(hits, beats))
    if any(b - a != 4 for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("baseline does not have constant four-beat bars")
    index = np.arange(boundaries[0], boundaries[-1] + 1)
    grid = np.interp(index, np.arange(len(beats)), beats)
    grid[index < 0] = beats[0] + index[index < 0] * (beats[1] - beats[0])
    beyond = index >= len(beats)
    grid[beyond] = beats[-1] + (index[beyond] - len(beats) + 1) * (beats[-1] - beats[-2])
    return grid, (float(beats[4]), float(beats[-5]))


def timing_match(predicted, reference, grid, window):
    """Symmetric bar-time agreement on the shared interior (edge bars excluded).

    Distances are measured in local beat units, not fixed seconds. A song is
    correct only if >=90% of both predicted and reference starts are within
    0.10 beat. This allows bounded saved-score display refinement, while
    remaining below half the 0.25-beat candidate spacing.
    """
    predicted = np.asarray(predicted)
    reference = np.asarray(reference)
    low, high = window
    p = predicted[(predicted >= low) & (predicted <= high)]
    r = reference[(reference >= low) & (reference <= high)]
    if not len(p) or not len(r):
        raise ValueError("no interior bar starts to compare")
    px = np.interp(p, grid, np.arange(len(grid)))
    rx = np.interp(r, grid, np.arange(len(grid)))
    # Match against the full opposing set: display refinement may move the
    # same bar a few milliseconds across the evaluation-window boundary.
    all_px = np.interp(predicted, grid, np.arange(len(grid)))
    all_rx = np.interp(reference, grid, np.arange(len(grid)))
    dp = np.abs(px[:, None] - all_rx[None, :]).min(axis=1)
    dr = np.abs(rx[:, None] - all_px[None, :]).min(axis=1)
    precision, recall = float(np.mean(dp <= TOLERANCE_BEATS)), float(np.mean(dr <= TOLERANCE_BEATS))
    return {
        "correct": min(precision, recall) >= REQUIRED_AGREEMENT,
        "precision": precision, "recall": recall,
        "mean_error_beats": float((dp.mean() + dr.mean()) / 2),
        "reference_bars": len(r), "predicted_bars": len(p),
    }


def summary(rows):
    return {
        "n": len(rows),
        "default": sum(r["default_correct"] for r in rows),
        **{f"top{k}": sum(r[f"top{k}"] for r in rows) for k in (1, 2, 3)},
    }


def evaluate(song_dir):
    raw = json.loads((song_dir / "options.json").read_text())
    if raw.get("time_signature") is not None:
        return None, {"song": song_dir.name, "reason": f"time_signature={raw['time_signature']}"}
    opts = Options(**{k: v for k, v in raw.items() if k in {f.name for f in fields(Options)}})
    saved = json.loads((song_dir / "score.json").read_text())
    if any(b["beats"] != 4 or b.get("denominator", 4) != 4 for b in saved["bars"]) or opts.odd_bars != "none":
        return None, {"song": song_dir.name, "reason": "not plain constant 4/4"}
    hits, _, _ = detect_hits(song_dir, opts)
    cached = json.loads((song_dir / "beats.json").read_text())
    grid, window = baseline_grid(hits, cached, opts)
    ranking = rank_bar_phases(hits, grid)
    features = phase_features(hits, grid)
    reference = [b["beat_times"][0] for b in saved["bars"]]
    starts = {shift: bar_start_times(grid, shift) for shift, _ in ranking}
    matches = {shift: timing_match(times, reference, grid, window) for shift, times in starts.items()}
    label = min(matches, key=lambda s: matches[s]["mean_error_beats"])
    compatible = matches[label]["correct"]
    row = {
        "song": song_dir.name, "title": saved["title"],
        "label_group": "strong" if opts.beat_shift != 0 else "weak",
        "option_beat_shift": opts.beat_shift, "tempo_scale": opts.tempo_scale,
        "label_shift": label if compatible else None, "closest_label_shift": label,
        "label_compatible": compatible, "label_match": matches[label],
        "default_shift": 0, "default_correct": matches[0]["correct"],
        "prediction": ranking[0][0], "margin": ranking[0][1] - ranking[1][1],
        **{f"top{k}": any(matches[s]["correct"] for s, _ in ranking[:k]) for k in (1, 2, 3)},
        "label_rank": next((i + 1 for i, (s, _) in enumerate(ranking) if matches[s]["correct"]), None),
        "ranking": [{"shift": s, "score": value, "features": features[s], "match": matches[s]} for s, value in ranking],
        "reference_bar_starts": reference,
        "predicted_bar_starts": starts[ranking[0][0]], "default_bar_starts": starts[0],
        "evaluation_window_seconds": window, "hits": len(hits),
        "sound_names_present": bool(opts.sound_names),
        "ablation_predictions": {}, "ablation_correct": {},
    }
    for removed in DEFAULT_WEIGHTS:
        alternative = sorted(features, key=lambda s: (-sum(w * features[s][name] for name, w in DEFAULT_WEIGHTS.items() if name != removed), s))[0]
        row["ablation_predictions"][removed] = alternative
        row["ablation_correct"][removed] = matches[alternative]["correct"]
    return row, None


def rate(n, total):
    return f"{n}/{total} ({100*n/total:.1f}%)" if total else "0/0 (N/A)"


def markdown(report):
    lines = [
        "# Bar phase first pass", "",
        "Fixed hand-set weights; no label fitting or sweeps. Offsets below are physical beats modulo 4 relative to the reconstructed zero-shift baseline, not UI option values.", "",
        "| Song | Label group | UI shift | Label offset | Default correct | Top 3 offsets | Top-1 / 2 / 3 | Margin |",
        "|---|---|---:|---:|---|---|---|---:|",
    ]
    for r in report["songs"]:
        top3 = ", ".join(f"{x['shift']:g}" for x in r["ranking"][:3])
        success = "/".join("Y" if r[f"top{k}"] else "N" for k in (1, 2, 3))
        lines.append(f"| {r['title'].replace('|', '/')} | {r['label_group']} | {r['option_beat_shift']:g} | {r['label_shift']} | {'Y' if r['default_correct'] else 'N'} | {top3} | {success} | {r['margin']:.4f} |")
    lines += ["", "| Group | Default | Top-1 | Top-2 | Top-3 |", "|---|---|---|---|---|"]
    for group, s in report["summary"].items():
        lines.append(f"| {group} | " + " | ".join(rate(s[k], s['n']) for k in ('default', 'top1', 'top2', 'top3')) + " |")
    lines += ["", "| Margin > | Group | Coverage | Retained top-1 | Correct among actual changes |", "|---:|---|---|---|---|"]
    for r in report["selective"]:
        lines.append(f"| {r['threshold']:.2f} | {r['group']} | {rate(r['n'], r['total'])} | {rate(r['top1'], r['n'])} | {rate(r['changed_correct'], r['changed_n'])} |")
    lines += ["", "| Removed cue | Strong top-1 | Weak top-1 |", "|---|---|---|"]
    for cue, values in report["ablation"].items():
        lines.append(f"| {cue} | {rate(values['strong'], report['summary']['strong']['n'])} | {rate(values['weak'], report['summary']['weak']['n'])} |")
    lines += ["", "Evaluation: symmetric >=90% bar-start agreement within 0.10 local beat; exclude one bar near both tracker endpoints. All eligible songs remain in denominators, including any incompatible labels.", "",
              "Baseline: current quantizer's steady/recount/rescale/downbeat-vote path using cached detect_hits output. No sound-map renaming, waveform tom sizing, fill_snares, cymbal veto, or display-time refinement. This reconstructs the current default, not a historical software snapshot.", "",
              "Strong/weak status comes only from options.beat_shift != 0 / == 0. Label offsets and accuracy come from score.json bar-start times. No score notes enter the scorer. Margins are heuristic units, not calibrated probabilities. Thresholds are descriptive, not validated deployment cutoffs.", "",
              "Truth: " + report['truth_note'], "", "Skipped:"]
    lines += [f"- {r['song']}: {r['reason']}" for r in report['skipped']]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--songs", type=Path, default=Path("songs"))
    parser.add_argument("--output", type=Path, default=Path("reports/bar_phase_first_pass.json"))
    args = parser.parse_args()
    rows, skipped = [], []
    for path in sorted(args.songs.iterdir()):
        if not path.is_dir():
            continue
        row, skip = evaluate(path)
        if row is not None:
            rows.append(row)
            print(f"{path.name}: label={row['label_shift']} top1={row['prediction']:g} rank={row['label_rank']} margin={row['margin']:.4f}", flush=True)
        else:
            skipped.append(skip)
    groups = {"strong": [r for r in rows if r['label_group'] == 'strong'], "weak": [r for r in rows if r['label_group'] == 'weak'], "all": rows, "truth": []}
    report = {
        "weights": DEFAULT_WEIGHTS, "fitted": False,
        "tolerance_beats": TOLERANCE_BEATS, "required_agreement": REQUIRED_AGREEMENT,
        "songs": rows, "skipped": skipped, "truth_note": TRUTH_NOTE,
        "summary": {name: summary(group) for name, group in groups.items()},
        "selective": [{
            "threshold": t, "group": name, "total": len(group),
            **summary([r for r in group if r['margin'] > t]),
            "changed_n": sum(r['margin'] > t and r['prediction'] != 0 for r in group),
            "changed_correct": sum(r['margin'] > t and r['prediction'] != 0 and r['top1'] for r in group),
        } for t in MARGINS for name, group in groups.items() if name != 'truth'],
        "ablation": {cue: {name: sum(r['ablation_correct'][cue] for r in group) for name, group in groups.items() if name != 'truth'} for cue in DEFAULT_WEIGHTS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown(report))
    print(json.dumps(report['summary'], indent=2))
    print(f"Wrote {args.output} and {args.output.with_suffix('.md')}")


if __name__ == "__main__":
    main()
