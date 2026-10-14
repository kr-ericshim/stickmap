#!/usr/bin/env bash
# Full experiment; run from any directory. All generated files stay under .eval-cache/ft/.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="tools/finetune${PYTHONPATH:+:$PYTHONPATH}"
# Compare the default pipeline thresholds even if the caller tested a tom scale earlier.
unset FT_TOM_SCALE
PYTHON=.venv/bin/python
RESULTS=.eval-cache/ft/results/v3
mkdir -p "$RESULTS"
test -f tools/harvest/global_score.py
ONLY=(swartz allyeodrum drumclass mydrumsheet taktak drumshin drumplace nakamura chock yeol atthedrum jilju drummate drumscribe)

for fold in 0 1 2; do
    device=mps
    threads=0
    if [[ "$fold" != 0 ]]; then
        device=cpu
        threads=4
    fi
    "$PYTHON" tools/finetune/train.py --fold "$fold" --tag v3 --doubt 0.3 --dense 1.0 \
        --dense-mode window --dense-labels legacy --no-withheld-mask --only "${ONLY[@]}" \
        --device "$device" --threads "$threads" --steps 1500 | tee "$RESULTS/train-fold$fold.log"
done
for fold in 0 1 2; do
    "$PYTHON" tools/finetune/predict.py "v3-fold$fold" | tee "$RESULTS/predict-fold$fold.log"
done
"$PYTHON" tools/finetune/solos.py v3 | tee "$RESULTS/solos-predict.log"

# Predictions may have changed on a rerun; rebuild only v3's derived snare responses.
"$PYTHON" - <<'PY'
from pathlib import Path
cache = Path('.eval-cache/ft')
for pattern in ('*/snare-student-v3.npy', 'solos/*/snare-student-v3.npy'):
    for file in cache.glob(pattern):
        file.unlink()
PY

for tag in v3 v2; do
    FT_TAG="$tag" "$PYTHON" tools/harvest/fill_score.py --detect ft_variants:tuned \
        --only "${ONLY[@]}" --jobs 4 --json "$RESULTS/$tag-covers.json" | tee "$RESULTS/$tag-covers.log"
    FT_TAG="$tag" "$PYTHON" tools/harvest/global_score.py --detect ft_variants:tuned \
        --only "${ONLY[@]}" --json "$RESULTS/$tag-global.json" | tee "$RESULTS/$tag-global.log"
    FT_TAG="$tag" "$PYTHON" tools/truth_video/fill_seq.py --detect ft_variants:tuned \
        --jobs 4 --json "$RESULTS/$tag-solos.json" | tee "$RESULTS/$tag-solos.log"
done
"$PYTHON" tools/finetune/report_v3.py
