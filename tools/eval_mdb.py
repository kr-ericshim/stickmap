"""Measure detection accuracy on MDB Drums (23 full mixes with onset + beat annotations).

    uv run python tools/eval_mdb.py "<path>/MDBDrums/MDB Drums" [--cache DIR]

Compares running ADTOF on the Demucs drum stem, on the full mix, and on the
element-wise max / mean of both, then reports beat and downbeat tracking.
"""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np

from stickmap import models

CLASS_MAP = {"KD": "kick", "SD": "snare", "HH": "hihat", "TT": "tom", "CY": "cymbal"}
CLASSES = ("kick", "snare", "hihat", "tom", "cymbal")


def match(reference: list[float], estimate: list[float], tolerance: float) -> int:
    """Count one-to-one matches within `tolerance` seconds (both lists sorted)."""
    hits, i, j = 0, 0, 0
    while i < len(reference) and j < len(estimate):
        delta = estimate[j] - reference[i]
        if abs(delta) <= tolerance:
            hits, i, j = hits + 1, i + 1, j + 1
        elif delta < 0:
            j += 1
        else:
            i += 1
    return hits


def f_measure(tp: int, n_ref: int, n_est: int) -> tuple[float, float, float]:
    precision = tp / n_est if n_est else 0.0
    recall = tp / n_ref if n_ref else 0.0
    f = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f


def cached(path: Path, compute):
    if path.exists():
        return np.load(path)
    value = np.asarray(compute())
    np.save(path, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".eval-cache"))
    parser.add_argument("--demucs-model", default="htdemucs")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    device = models.pick_device()

    # totals[variant][class] = [tp, n_ref, n_est]
    totals = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    beat_totals = {"beats": [0, 0, 0], "downbeats": [0, 0, 0]}

    mixes = sorted((args.dataset / "audio" / "full_mix").glob("*_MIX.wav"))
    for mix in mixes:
        name = mix.stem.removesuffix("_MIX")
        reference = defaultdict(list)
        for line in (args.dataset / "annotations" / "class" / f"{name}_class.txt").read_text().splitlines():
            if line.strip():
                onset, label = line.split()
                if label in CLASS_MAP:
                    reference[CLASS_MAP[label]].append(float(onset))

        drums = args.cache / f"{name}.drums.{args.demucs_model}.wav"
        if not drums.exists():
            models.separate_drums(mix, drums, device=device, model=args.demucs_model)
        stem = cached(args.cache / f"{name}.act-stem.{args.demucs_model}.npy", lambda: models.drum_activations(drums, device=device))
        full = cached(args.cache / f"{name}.act-mix.npy", lambda: models.drum_activations(mix, device=device))
        frames = min(len(stem), len(full))
        variants = {
            "stem": stem,
            "mix": full,
            "max": np.maximum(stem[:frames], full[:frames]),
            "mean": (stem[:frames] + full[:frames]) / 2,
        }
        for variant, activations in variants.items():
            estimate = defaultdict(list)
            for hit in models.pick_hits(activations):
                estimate[hit.inst].append(hit.time)
            for inst in CLASSES:
                tp = match(sorted(reference[inst]), estimate[inst], 0.05)
                row = totals[variant][inst]
                row[0], row[1], row[2] = row[0] + tp, row[1] + len(reference[inst]), row[2] + len(estimate[inst])

        beats_path = args.cache / f"{name}.beats.npz"
        if not beats_path.exists():
            beats, downbeats = models.track_beats(mix, device=device)
            np.savez(beats_path, beats=beats, downbeats=downbeats)
        tracked = np.load(beats_path)
        rows = [l.split() for l in (args.dataset / "annotations" / "beats" / f"{name}_MIX.beats").read_text().splitlines() if l.strip()]
        ref_beats = [float(r[0]) for r in rows]
        ref_down = [float(r[0]) for r in rows if r[1] == "1"]
        for key, ref, est in (("beats", ref_beats, tracked["beats"]), ("downbeats", ref_down, tracked["downbeats"])):
            tp = match(ref, list(est), 0.07)
            beat_totals[key] = [a + b for a, b in zip(beat_totals[key], (tp, len(ref), len(est)))]
            if key == "downbeats":
                print(f"{name:14} downbeat F {f_measure(tp, len(ref), len(est))[2]:.2f}  ({len(est)} est / {len(ref)} ref)")

    print(f"\nOnset F-measure over {len(mixes)} tracks, ±50 ms (precision / recall / F)")
    print(f"{'':8}" + "".join(f"{c:>20}" for c in (*CLASSES, "ALL")))
    for variant, per_class in totals.items():
        cells = [f_measure(*per_class[c]) for c in CLASSES]
        cells.append(f_measure(*[sum(v) for v in zip(*(per_class[c] for c in CLASSES))]))
        print(f"{variant:8}" + "".join(f"{p:>7.2f}/{r:.2f}/{f:.3f}" for p, r, f in cells))
    print("reference counts: " + ", ".join(f"{c} {totals['stem'][c][1]}" for c in CLASSES))
    for key, row in beat_totals.items():
        p, r, f = f_measure(*row)
        print(f"{key}: P {p:.3f} R {r:.3f} F {f:.3f} (±70 ms)")


if __name__ == "__main__":
    main()
