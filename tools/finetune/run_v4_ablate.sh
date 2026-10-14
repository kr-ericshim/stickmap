#!/usr/bin/env bash
# v4 without dense positives; separate tags for full and 20-step smoke runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="tools/finetune${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
unset FT_KICK_SCALE FT_SNARE_SCALE FT_TOM_SCALE FT_HIHAT_SCALE FT_CYMBAL_SCALE
PYTHON=.venv/bin/python
TAG=v4nd
STEPS=1500
SMOKE_CPU=0
if [[ ( "${1:-}" == --smoke || "${1:-}" == --smoke-cpu ) && $# == 1 ]]; then
    TAG=v4nd-smoke
    STEPS=20
    if [[ "$1" == --smoke-cpu ]]; then
        SMOKE_CPU=1
    fi
elif [[ $# != 0 ]]; then
    echo "Usage: bash tools/finetune/run_v4_ablate.sh [--smoke|--smoke-cpu]" >&2
    exit 2
fi
RESULTS=".eval-cache/ft/results/$TAG"
mkdir -p "$RESULTS"
ONLY=(swartz allyeodrum drumclass mydrumsheet taktak drumshin drumplace nakamura chock yeol atthedrum jilju drummate drumscribe)
test -f tools/harvest/global_score.py

for fold in 0 1 2; do
    device=mps
    threads=0
    if [[ "$fold" != 0 || "$SMOKE_CPU" == 1 ]]; then
        device=cpu
        threads=4
    fi
    "$PYTHON" tools/finetune/train.py --fold "$fold" --tag "$TAG" --doubt 0.3 --dense 0 \
        --dense-mode peak --dense-labels fold --device "$device" --threads "$threads" \
        --steps "$STEPS" | tee "$RESULTS/train-fold$fold.log"
done
for fold in 0 1 2; do
    "$PYTHON" tools/finetune/predict.py "$TAG-fold$fold" | tee "$RESULTS/predict-fold$fold.log"
done
"$PYTHON" tools/finetune/solos.py "$TAG" | tee "$RESULTS/solos-predict.log"

# Invalidate only this run's derived snare cache after replacing its predictions.
FT_RUN_TAG="$TAG" "$PYTHON" - <<'PY'
import os
from pathlib import Path
tag = os.environ['FT_RUN_TAG']
cache = Path('.eval-cache/ft')
for pattern in (f'*/snare-student-{tag}.npy', f'solos/*/snare-student-{tag}.npy'):
    for file in cache.glob(pattern):
        file.unlink()
PY

"$PYTHON" tools/finetune/prepare_score.py --tag "$TAG" --only "${ONLY[@]}" --jobs 4 \
    | tee "$RESULTS/prepare-score.log"

for tag in "$TAG" v2; do
    FT_TAG="$tag" "$PYTHON" tools/harvest/fill_score.py --detect ft_variants:tuned \
        --only "${ONLY[@]}" --jobs 4 --json "$RESULTS/$tag-covers.json" | tee "$RESULTS/$tag-covers.log"
    FT_TAG="$tag" "$PYTHON" tools/harvest/global_score.py --detect ft_variants:tuned \
        --only "${ONLY[@]}" --json "$RESULTS/$tag-global.json" | tee "$RESULTS/$tag-global.log"
    FT_TAG="$tag" "$PYTHON" tools/truth_video/fill_seq.py --detect ft_variants:tuned \
        --jobs 4 --json "$RESULTS/$tag-solos.json" | tee "$RESULTS/$tag-solos.log"
done
"$PYTHON" tools/finetune/report_v4.py --tag "$TAG"
