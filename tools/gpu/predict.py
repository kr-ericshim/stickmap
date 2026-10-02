"""Whole-song CUDA inference from cached features, without audio or scoring.

Usage: .venv/bin/python tools/gpu/predict.py TAG [TAG ...] [--folds 0 1 2] [--no-solos]
For each fold: predict the checkpoint JSON's held songs and all cached solos.
Save float32 student-TAG-foldK.npy beside feat.npy. Tags/folds run sequentially.
Missing checkpoints/features are errors; there is no CPU fallback.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tags", nargs="+")
    parser.add_argument("--folds", nargs="+", type=int, choices=(0, 1, 2), default=[0, 1, 2])
    parser.add_argument("--no-solos", action="store_true", help="predict held-out harvested songs only")
    args = parser.parse_args()
    for tag in args.tags:
        if not re.fullmatch(r"[A-Za-z0-9_]+", tag):
            parser.error(f"invalid tag {tag!r}: use letters/digits/underscore, no '-'")

    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT / "tools" / "finetune"))
    from train import CACHE, MODELS, original_model, run

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run this in the WSL cu128 venv")
    torch.set_num_threads(1)
    solos = []
    if not args.no_solos:
        if not (CACHE / "solos").is_dir():
            parser.error("missing .eval-cache/ft/solos; prepare on the Mac and push, or use --no-solos")
        solos = sorted(p for p in (CACHE / "solos").iterdir() if p.is_dir())
        if not solos:
            parser.error("no cached solos; prepare on the Mac and push, or use --no-solos")

    # Check all requested inputs before overwriting any predictions.
    plan = []
    for tag in args.tags:
        for fold in dict.fromkeys(args.folds):
            name = f"{tag}-fold{fold}"
            checkpoint = MODELS / f"{name}.pt"
            if not checkpoint.is_file():
                parser.error(f"missing {checkpoint}")
            metadata = json.loads((MODELS / f"{name}.json").read_text())
            held = metadata["held"]
            if any(not isinstance(slug, str) or Path(slug).name != slug or slug in (".", "..") for slug in held):
                parser.error(f"invalid held slug in {name}.json")
            folders = list(dict.fromkeys([CACHE / slug for slug in held] + solos))
            for folder in folders:
                if not (folder / "feat.npy").is_file():
                    parser.error(f"missing {folder / 'feat.npy'}")
            plan.append((name, checkpoint, folders))

    for name, checkpoint, folders in plan:
        model = original_model("cuda")
        model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
        model.eval()
        for folder in folders:
            feat = np.load(folder / "feat.npy", allow_pickle=False)
            act = run(model, feat, "cuda").astype(np.float32)
            output = folder / f"student-{name}.npy"
            # Leave a complete previous output in place if writing is interrupted.
            temporary = output.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, act)
            temporary.replace(output)
        del model
        torch.cuda.empty_cache()
        print(f"{name}: {len(folders)} songs predicted", flush=True)


if __name__ == "__main__":
    main()
