"""Features and teacher activations for the drum-solo videos (external test, never trained on), and fold-model predictions.

    uv run python tools/finetune/solos.py                 # features + teacher0
    uv run python tools/finetune/solos.py v1              # + student-v1-fold{0,1,2}.npy
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).resolve().parents[1] / "truth_video")]
from train import CACHE, MODELS, original_model, run  # noqa: E402

import fill_seq  # noqa: E402

OUT = CACHE / "solos"


def songs() -> list[Path]:
    out = []
    for video in fill_seq.SOLOS:
        song = Path((fill_seq.TRUTH / video / "song.txt").read_text().strip())
        if (song / "activations.drums-htdemucs.npy").exists():
            out.append(song)
    return out


def prepare(song: Path) -> None:
    import torch
    from adtof_pytorch.audio import process_audio_file

    torch.set_num_threads(1)
    folder = OUT / song.name
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / "feat.npy").exists():
        stem = song / "drums.htdemucs.wav"
        feat, _ = process_audio_file(str(stem if stem.exists() else stem.with_suffix(".opus")))
        np.save(folder / "feat.npy", feat[:, :, 0].astype(np.float16))
    if not (folder / "teacher0.npy").exists():
        np.save(folder / "teacher0.npy", run(original_model("cpu").eval(), np.load(folder / "feat.npy"), "cpu").astype(np.float16))


def predict(args: tuple[str, Path]) -> None:
    import torch

    torch.set_num_threads(1)
    name, song = args
    model = original_model("cpu")
    model.load_state_dict(torch.load(MODELS / f"{name}.pt", map_location="cpu"))
    folder = OUT / song.name
    np.save(folder / f"student-{name}.npy", run(model.eval(), np.load(folder / "feat.npy"), "cpu").astype(np.float32))


if __name__ == "__main__":
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(prepare, songs()))
        if len(sys.argv) > 1:
            tag = sys.argv[1]
            list(pool.map(predict, [(f"{tag}-fold{k}", s) for k in range(3) for s in songs()]))
    print(len(songs()), "solo songs ready in", OUT)
