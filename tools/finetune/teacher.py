"""The original ADTOF model's activations on each cached song's features (teacher0.npy), in parallel on the CPU."""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train import CACHE, original_model, run  # noqa: E402


def one(folder: Path) -> None:
    import torch

    torch.set_num_threads(1)
    out = folder / "teacher0.npy"
    if not out.exists():
        np.save(out, run(original_model("cpu").eval(), np.load(folder / "feat.npy"), "cpu").astype(np.float16))


if __name__ == "__main__":
    folders = [d for d in sorted(CACHE.iterdir()) if (d / "feat.npy").exists()]
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(one, folders))
    print(sum((d / "teacher0.npy").exists() for d in folders), "of", len(folders))
