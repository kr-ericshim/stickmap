"""Run a text file of 'TAG <train.py args>' experiments on CUDA in parallel.

Usage: .venv/bin/python tools/gpu/train_tags.py [--jobs 3] [--no-predict] experiments.txt
Each non-comment line is parsed by shlex, never a shell. Folds 0, 1, 2 run
sequentially within each tag; up to --jobs tags run together. No SSH is used.
All train.py defaults are retained except device=cuda, tag, and fold.
Logs: logs/gpu/TAG-foldK.log and TAG-predict.log. Failures return a nonzero exit.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def experiments(path: Path) -> list[tuple[str, list[str]]]:
    items = []
    seen = set()
    for number, line in enumerate(path.read_text().splitlines(), 1):
        words = shlex.split(line, comments=True)
        if not words:
            continue
        tag, *args = words
        if not re.fullmatch(r"[A-Za-z0-9_]+", tag):
            raise ValueError(f"{path}:{number}: tag must use letters/digits/underscore, no '-'")
        if tag in seen:
            raise ValueError(f"{path}:{number}: duplicate tag {tag}")
        # These are owned by the launcher. Require full option names to prevent
        # argparse abbreviations such as --dev or --fo from overriding them.
        for arg in args:
            option = arg.split("=", 1)[0]
            if option.startswith("--") and any(
                reserved.startswith(option) for reserved in ("--tag", "--fold", "--device", "--help")
            ):
                raise ValueError(f"{path}:{number}: {option} is reserved by the launcher")
            if option == "-h":
                raise ValueError(f"{path}:{number}: -h is not a training argument")
        seen.add(tag)
        items.append((tag, args))
    if not items:
        raise ValueError(f"{path}: no experiments")
    return items


def experiment(item: tuple[str, list[str]], predict: bool) -> None:
    tag, args = item
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    logs = ROOT / "logs" / "gpu"

    def execute(script: str, arguments: list[str], name: str) -> None:
        command = [sys.executable, str(ROOT / "tools" / script), *arguments]
        with (logs / f"{name}.log").open("w") as log:
            log.write(shlex.join(command) + "\n")
            log.flush()
            subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True)

    for fold in range(3):
        execute("finetune/train.py", ["--device", "cuda", "--tag", tag,
                                     "--fold", str(fold), *args], f"{tag}-fold{fold}")
    if predict:
        execute("gpu/predict.py", [tag], f"{tag}-predict")
    print(f"{tag} done", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", type=Path)
    parser.add_argument("--jobs", type=int, choices=range(1, 4), default=3,
                        help="parallel tags, each with sequential folds (default: 3)")
    parser.add_argument("--no-predict", action="store_true", help="train only; predict separately later")
    args = parser.parse_args()
    try:
        items = experiments(args.experiments)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    import torch

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; run this in the WSL cu128 venv")
    # train.py creates missing teacher0.npy. Concurrent writers can corrupt
    # that shared cache, so populate it with a sequential run first.
    if args.jobs > 1:
        missing = [p.parent for p in (ROOT / ".eval-cache" / "ft").glob("*/truth.json")
                   if not (p.parent / "teacher0.npy").is_file()]
        if missing:
            parser.error(f"{len(missing)} song caches lack teacher0.npy; use --jobs 1 first")
    (ROOT / "logs" / "gpu").mkdir(parents=True, exist_ok=True)
    failed = False
    with ThreadPoolExecutor(args.jobs) as pool:
        pending = {pool.submit(experiment, item, not args.no_predict): item[0] for item in items}
        for future in as_completed(pending):
            try:
                future.result()
            except (OSError, subprocess.CalledProcessError) as exc:
                failed = True
                print(f"{pending[future]} failed: {exc}; see logs/gpu/", file=sys.stderr)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
