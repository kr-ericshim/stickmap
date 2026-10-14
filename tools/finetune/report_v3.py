"""Make a v2/v3 Markdown comparison from the three scorers' integer totals."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def report(inputs: Path, output: Path, candidate: str = "v3") -> None:
    totals = {tag: {scorer: json.loads((inputs / f"{tag}-{scorer}.json").read_text())
                    for scorer in ("global", "covers", "solos")} for tag in ("v2", candidate)}
    rows = []
    groups = (("All harvested songs", "global", None),
              ("Cover fills, all", "covers", "all"),
              ("Cover fills, gap < 90 ms", "covers", "fast"),
              ("Solo passages, all", "solos", "all"),
              ("Solo passages, gap < 90 ms", "solos", "fast"))
    for name, scorer, key in groups:
        for cls in ("kick", "snare", "tom"):
            values = []
            for tag in ("v2", candidate):
                data = totals[tag][scorer]
                if scorer == "global":
                    tp, est, ref = data[cls]
                else:
                    tp, est, ref = (data.get(f"{key}.{cls}.{field}", 0) for field in ("tp", "est", "ref"))
                if ref <= 0:
                    raise ValueError(f"{tag} {scorer} {key} {cls}: no reference strokes")
                values.append((tp / max(1, est), tp / ref, 2 * tp / max(1, est + ref), ref))
            a, b = values
            if a[3] != b[3]:
                raise ValueError(f"{name} {cls}: v2/{candidate} reference counts differ")
            rows.append(f"| {name} | {cls} | {a[3]} | {a[0]:.4f} / {a[1]:.4f} | "
                        f"{b[0]:.4f} / {b[1]:.4f} | {b[2] - a[2]:+.4f} |")
    title = "v3: v2 + dense positives" if candidate == "v3" else f"{candidate}: peak-local dense positives vs v2"
    text = [f"# {title}", "",
            "Channel-held-out harvested predictions; solo predictions averaged across folds 0, 1 and 2.",
            "Default scorer settings and `ft_variants:tuned`; strict harvested truth. "
            "Dense data supplies training positives only. Scores use the existing evaluation truth.", "",
            f"| Evaluation | Class | Reference strokes | v2 P / R | {candidate} P / R | F1 change |",
            "|---|---|---:|---:|---:|---:|", *rows, "", "Integer totals and scorer logs:", ""]
    if candidate != "v3":
        text[2:2] = ["Harvested evaluation is restricted to the original 14 channels; solos remain an external test.",
                     "Dense weight 0.3; only the detached student argmax +/-1 frame (clipped to its bag) "
                     "masks KD and negative truth. The remaining frames keep their existing supervision.",
                     "Fold-local quality, same-song exclusions, partial-legend masks and new training channels "
                     "also changed; score differences cannot be attributed to the loss alone.", ""]
    for tag in ("v2", candidate):
        links = []
        for scorer in ("global", "covers", "solos"):
            for suffix in ("json", "log"):
                file = inputs / f"{tag}-{scorer}.{suffix}"
                links.append(f"[{scorer}.{suffix}]({os.path.relpath(file, output.parent)})")
        text.append(f"- {tag}: " + ", ".join(links))
    text += ["", "Precision against written truth is a lower bound: unwritten strokes can count as extras.",
             "Cover 'fast' uses each fill's median gap; solo 'fast' uses each stroke's nearest neighbour gap.", ""]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path(".eval-cache/ft/results/v3"))
    parser.add_argument("--output", type=Path, default=Path(".eval-cache/ft/results/v3.md"))
    args = parser.parse_args()
    report(args.inputs, args.output)
    print(args.output)
