"""Detectors for the scorers (--detect ft_variants:...): the full pipeline on a fine-tuned model's stem activations.

The mix activations stay the original model's. A harvested song is scored with the fold model that held it
out (student-<tag>-fold<k>.npy); a solo video, which no fold trained on, with the mean of the fold models.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".eval-cache" / "ft"
TAG = os.environ.get("FT_TAG", "v1")


def student(song: Path) -> np.ndarray:
    folder = CACHE / song.name
    if not folder.exists():
        folder = CACHE / "solos" / song.name
    found = sorted(folder.glob(f"student-{TAG}-fold?.npy"))
    if not found:
        raise FileNotFoundError(f"no student-{TAG} activations for {song.name}")
    return np.mean([np.load(f) for f in found], axis=0)


def snare(song: Path, act: np.ndarray, name: str) -> np.ndarray:
    """dynamics.snare_response from these activations, as the pipeline remakes it; cached as snare-<name>.npy."""
    from groovemap.dynamics import snare_response

    folder = CACHE / song.name if (CACHE / song.name).exists() else CACHE / "solos" / song.name
    cached = folder / f"snare-{name}.npy"
    if not cached.exists():
        stem = song / "drums.htdemucs.wav"
        np.save(cached, snare_response(stem if stem.exists() else stem.with_suffix(".opus"), act))
    return np.load(cached)


def full(song: Path, act: np.ndarray, name: str, sensitivity: float = 1.0):
    from groovemap import models
    from groovemap.fills import fill_snares, tom_runs
    from groovemap.toms import tom_sizes

    mix = np.load(song / "activations.mix.npy")
    n = min(len(act), len(mix))
    response = snare(song, act, name)
    act, mix = act[:n], mix[:n]
    thr = [t / sensitivity for t in models.DEFAULT_THRESHOLDS]
    # Threshold multipliers (larger = stricter), without changing activation,
    # mix, snare-response or rescue calibration. Defaults preserve old scores.
    for column, inst in enumerate(models.ADTOF_CLASSES):
        thr[column] *= float(os.environ.get(f"FT_{inst.upper()}_SCALE", "1"))
    attacks = np.load(song / "attacks.htdemucs.npy")
    hits = models.pick_hits(act, thr, mix=mix, snare=response)
    hits = fill_snares(tom_runs(hits, act, mix, attacks, thr), act, response, attacks)
    stem = song / "drums.htdemucs.wav"
    return tom_sizes(stem if stem.exists() else stem.with_suffix(".opus"), hits)


def tuned(song: Path):
    return full(song, student(song), f"student-{TAG}")


def teacher(song: Path):
    """The same chain on the original model's activations from the cached features (the fair baseline)."""
    folder = CACHE / song.name
    if not folder.exists():
        folder = CACHE / "solos" / song.name
    return full(song, np.load(folder / "teacher0.npy").astype(np.float32), "teacher0")
