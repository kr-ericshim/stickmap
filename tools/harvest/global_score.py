"""Run the original score.py with only its detect callable replaced in memory."""
from __future__ import annotations
import argparse
import importlib
import json
import runpy
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "truth_video"))
sys.path.insert(0, str(ROOT / "tools" / "harvest"))
import evaluate
import vault

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detect", required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--model-tag", help="model identity for the vault opening log")
    args, rest = parser.parse_known_args()
    command = shlex.join([sys.executable, *sys.argv])
    module, function = args.detect.split(":")
    evaluate.detect = getattr(importlib.import_module(module), function)
    ns = runpy.run_path(str(ROOT / "tools" / "harvest" / "score.py"))
    # Wrap tally to retain the exact integer counters behind the printed line.
    original = ns["main"].__globals__["tally"]
    totals = {}
    def tally(*a, **kw):
        result = original(*a, **kw)
        for fam in ns["FAMILIES"]:
            cell = totals.setdefault(fam, [0, 0, 0])
            for i, value in enumerate(result[fam]):
                cell[i] += value
        totals["songs"] = totals.get("songs", 0) + 1
        return result
    ns["main"].__globals__["tally"] = tally
    sys.argv = [str(ROOT / "tools" / "harvest" / "score.py"), *rest]
    ns["main"](model_tag=vault.model_tag(args.detect, args.model_tag), command=command)
    if args.json:
        args.json.write_text(json.dumps(totals, indent=2))

if __name__ == "__main__":
    main()
