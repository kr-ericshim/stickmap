"""The original ADTOF model's activations on each cached song's features (teacher0.npy), in parallel on the CPU."""

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--only", nargs="+", help="teacher inference only for these channels")
    args = parser.parse_args()
    folders = [d for d in sorted(CACHE.iterdir())
               if (d / "feat.npy").exists() and (d / "truth.json").exists()
               and (not args.only or json.loads((d / "truth.json").read_text())["channel"] in args.only)]
    with ProcessPoolExecutor(args.jobs) as pool:
        list(pool.map(one, folders))
    print(sum((d / "teacher0.npy").exists() for d in folders), "of", len(folders))
