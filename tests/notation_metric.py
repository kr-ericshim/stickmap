"""Cached-only notation audit. Writes exclusively to the requested scratch directory.

Run: .venv/bin/python tests/notation_metric.py --out /private/tmp/notation-before
Repeat with --inputs /private/tmp/notation-before to reuse the identical raw hits.
No pipeline.run, inference, sound-map learning, or writes to the song library.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linear_sum_assignment

from groovemap import kit, library
from groovemap.fills import fill_snares
from groovemap.musicxml import to_musicxml
from groovemap.pipeline import Options, cache_paths, detect_hits
from groovemap.quantize import _strokes, quantize
from groovemap.score import Hit, Score
from groovemap.toms import tom_sizes

COUNTS = ("notes", "events", "notes_32nd", "short_rests", "rests", "unbeamed_flags", "dots", "tuplet_switches", "ghosts", "snares", "snare_ghosts")


def raw_matches(score: Score, hits: list[Hit]) -> dict[tuple[int, int], float]:
    """One-to-one same-instrument minimum-error matching for legacy scores.

    New scores carry raw_time directly. Legacy matching is limited to 200 ms;
    unmatched notes are reported, never assigned zero error.
    """
    result = {}
    for inst in {h.inst for h in hits}:
        source = [h.time for h in _strokes(hits) if h.inst == inst]
        notes = [(b.index, i, b.tick_time(n.tick)) for b in score.bars for i, n in enumerate(b.notes) if n.inst == inst]
        if not source or not notes:
            continue
        error = np.abs(np.asarray(source)[:, None] - np.asarray([n[2] for n in notes])[None, :])
        a, b = linear_sum_assignment(error)
        for row, col in zip(a, b):
            if error[row, col] < 0.2:
                result[notes[col][:2]] = source[row]
    return result


def measure(score: Score, xml: str, hits: list[Hit]) -> dict:
    root = ET.fromstring(xml)
    matched = raw_matches(score, hits) if any(getattr(n, "raw_time", None) is None for b in score.bars for n in b.notes) else {}
    rows, errors = [], []
    for bar, node in zip(score.bars, root.findall("part/measure"), strict=True):
        row = dict.fromkeys(COUNTS, 0)
        row.update(bar=bar.index + 1, subdivisions=bar.subdivisions, beat_times=bar.beat_times)
        families = {}
        for note in node.findall("note"):
            if note.find("chord") is not None:
                continue
            rest = note.find("rest") is not None
            kind = note.findtext("type")
            row["rests" if rest else "events"] += 1
            row["notes_32nd"] += int(not rest and kind == "32nd")
            row["short_rests"] += int(rest and int(note.findtext("duration")) < 6)
            row["unbeamed_flags"] += int(not rest and kind in ("eighth", "16th", "32nd") and note.find("beam[@number='1']") is None)
            row["dots"] += len(note.findall("dot"))
            voice = note.findtext("voice", "1")
            family = note.findtext("time-modification/actual-notes", "straight")
            row["tuplet_switches"] += int(voice in families and families[voice] != family)
            families[voice] = family
        local_errors = []
        for i, note in enumerate(bar.notes):
            row["notes"] += 1
            row["ghosts"] += int(note.ghost)
            row["snares"] += int(note.inst == "snare")
            row["snare_ghosts"] += int(note.inst == "snare" and note.ghost)
            raw = getattr(note, "raw_time", None)
            if raw is None:
                raw = matched.get((bar.index, i))
            if raw is not None:
                local_errors.append(abs(raw - bar.tick_time(note.tick)) * 1000)
        errors += local_errors
        row.update(timing_samples=len(local_errors), timing_mae_ms=float(np.mean(local_errors)) if local_errors else None,
                   timing_error_sum_ms=sum(local_errors), ghost_ratio=row["ghosts"] / max(1, row["notes"]),
                   snare_ghost_ratio=row["snare_ghosts"] / max(1, row["snares"]))
        rows.append(row)
    total = {key: sum(row[key] for row in rows) for key in COUNTS}
    total.update(bars=len(rows), merged_strokes=score.merged_strokes, input_strokes=len(_strokes(hits)),
                 timing_samples=len(errors), timing_unmatched=total["notes"] - len(errors),
                 timing_mae_ms=float(np.mean(errors)) if errors else None,
                 timing_p95_ms=float(np.percentile(errors, 95)) if errors else None,
                 ghost_ratio=total["ghosts"] / max(1, total["notes"]),
                 snare_ghost_ratio=total["snare_ghosts"] / max(1, total["snares"]))
    return {"total": total, "bars": rows}


def cached_inputs(song: Path, opts: Options) -> tuple[list[Hit], np.ndarray, np.ndarray]:
    paths = cache_paths(song, opts)
    hits, activations, attacks = detect_hits(song, opts)
    if opts.onset_input != "mix":
        if opts.sound_names:
            saved = kit.SoundMap.load(song)
            if saved is None:
                raise ValueError("Saved sound names need an existing soundmap.json; audit cannot learn one")
            hits = kit.rename(hits, saved, opts.sound_names, paths["drums"], attacks)
        else:
            hits = fill_snares(hits, activations, np.load(paths["snare"]), attacks)
        hits = tom_sizes(paths["drums"], hits)
    return hits, activations, attacks


def rescore(hits: list[Hit], beats: dict, opts: Options, activations: np.ndarray, attacks: np.ndarray, title: str) -> Score:
    common = dict(title=title, grid=opts.grid, triplets=opts.triplets, beat_shift=opts.beat_shift, tempo_scale=opts.tempo_scale)
    if opts.time_signature:
        from groovemap.signature_quantize import quantize_signature

        return quantize_signature(hits, beats["beats"], beats["downbeats"], activations=activations, attacks=attacks,
                                  time_signature=opts.time_signature, **common)
    # Accept the older API when taking the initial baseline.
    import inspect

    if "attacks" in inspect.signature(quantize).parameters:
        common["attacks"] = attacks
    return quantize(hits, beats["beats"], beats["downbeats"], meter=opts.meter, odd_bars=opts.odd_bars, **common)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--songs", type=Path, default=Path("songs"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--slug")
    parser.add_argument("--triplets", choices=("auto", "off"))
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(Path("/private/tmp")):
        parser.error("--out must be under /private/tmp; the library is read-only")
    out.mkdir(parents=True, exist_ok=True)
    report = {"songs": {}, "errors": {}}
    for song in sorted(args.songs.iterdir()):
        if not song.is_dir() or (args.slug and args.slug != song.name):
            continue
        try:
            saved = library.load_options(song)
            opts = Options(**saved, pdf=False, preview=False)
            if args.triplets:
                opts.triplets = args.triplets
            beats = json.loads((song / "beats.json").read_text())
            if args.inputs:
                frozen = json.loads((args.inputs / song.name / "inputs.json").read_text())
                hits = [Hit(**h) for h in frozen["hits"]]
                paths = cache_paths(song, opts)
                activations = np.load(paths["activations"])
                attacks = np.load(paths["attacks"]) if paths["attacks"].exists() else np.empty(0)
            else:
                hits, activations, attacks = cached_inputs(song, opts)
            from groovemap.dynamics import annotate_confidence

            paths = cache_paths(song, opts)
            hits = annotate_confidence(hits, activations, np.load(paths["snare"]) if opts.onset_input != "mix" and paths["snare"].exists() else None)
            dest = out / song.name
            dest.mkdir(exist_ok=True)
            (dest / "inputs.json").write_text(json.dumps({"hits": [asdict(h) for h in hits], "options": saved}))
            score = rescore(hits, beats, opts, activations, attacks, song.name)
            xml = to_musicxml(score, voices=opts.voices)
            score.save(dest / "score.json")
            (dest / "score.musicxml").write_text(xml)
            metrics = measure(score, xml, hits)
            metrics["options"] = asdict(opts)
            report["songs"][song.name] = metrics
            print(song.name, json.dumps(metrics["total"]), flush=True)
        except (OSError, ValueError) as exc:
            report["errors"][song.name] = str(exc)
            print(song.name, "ERROR", str(exc), flush=True)
    totals = [s["total"] for s in report["songs"].values()]
    report["total"] = {key: sum(t[key] for t in totals) for key in (*COUNTS, "bars", "merged_strokes", "input_strokes", "timing_samples", "timing_unmatched")}
    t = report["total"]
    t["timing_mae_ms"] = sum(s["timing_mae_ms"] * s["timing_samples"] for s in totals if s["timing_mae_ms"] is not None) / max(1, t["timing_samples"])
    t["ghost_ratio"] = t["ghosts"] / max(1, t["notes"])
    t["snare_ghost_ratio"] = t["snare_ghosts"] / max(1, t["snares"])
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("TOTAL", json.dumps(t), flush=True)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
