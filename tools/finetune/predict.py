"""A fine-tuned model's activations for whole songs (student-<name>.npy next to the cached features), on the CPU in parallel.

    uv run python tools/finetune/predict.py v1-fold0            # the songs that fold held out
    uv run python tools/finetune/predict.py v1-fold0-1500       # an intermediate checkpoint
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train import CACHE, MODELS, original_model, run  # noqa: E402


def one(args: tuple[str, str]) -> None:
    import torch

    torch.set_num_threads(1)
    name, slug = args
    model = original_model("cpu")
    model.load_state_dict(torch.load(MODELS / f"{name}.pt", map_location="cpu"))
    folder = CACHE / slug
    np.save(folder / f"student-{name}.npy", run(model.eval(), np.load(folder / "feat.npy"), "cpu").astype(np.float32))


if __name__ == "__main__":
    name = sys.argv[1]
    run_name = name if (MODELS / f"{name}.json").exists() else name.rsplit("-", 1)[0]
    slugs = json.loads((MODELS / f"{run_name}.json").read_text())["held"]
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(one, [(name, slug) for slug in slugs]))
    print(len(slugs), "songs predicted with", name)
