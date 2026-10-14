"""Generate the original-14-channel v4/v2 report from actual scorer totals."""

from __future__ import annotations

import argparse
from pathlib import Path

from report_v3 import report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="v4")
    p.add_argument("--inputs", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    inputs = a.inputs or Path(f".eval-cache/ft/results/{a.tag}")
    output = a.output or Path(f".eval-cache/ft/results/{a.tag}.md")
    report(inputs, output, candidate=a.tag)
    if a.tag.startswith("v4nd"):
        output.write_text(output.read_text().replace(
            f"# {a.tag}: peak-local dense positives vs v2",
            f"# {a.tag}: v4 without dense positives vs v2").replace(
            "Dense weight 0.3; only the detached student argmax +/-1 frame (clipped to its bag) "
            "masks KD and negative truth. The remaining frames keep their existing supervision.",
            "Dense weight 0: no dense positives, dense crop sampling or dense-local masks. "
            "New channels, fold assignments and withheld-head masks match v4.").replace(
            "Dense data supplies training positives only.", "Dense supervision is disabled.").replace(
            "Fold-local quality, same-song exclusions, partial-legend masks and new training channels "
            "also changed; score differences cannot be attributed to the loss alone.",
            "Same-song exclusions, withheld-head masks and new training channels differ from v2; "
            "compare v4nd with v4 to isolate the dense package (loss, crop sampling and local masks)."))
    if a.tag.endswith("-smoke"):
        output.write_text(output.read_text().replace("\n\n", f"\n\n**20-step smoke only; these scores are not a 1500-step {a.tag.removesuffix('-smoke')} result.**\n\n", 1))
    print(output)


if __name__ == "__main__":
    main()
