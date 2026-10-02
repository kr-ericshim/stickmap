#!/usr/bin/env bash
# Mac fallback scoring, sequential pools with at most four CPU workers.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash tools/gpu/score_tags.sh --only CHANNEL[,CHANNEL...] [--jobs 4] [--pull] TAG [TAG ...]
Mac fallback when WSL scoring is unavailable; normally use score_wsl.sh and pull JSONs.
Run with prepared local audio/truth caches. No training or model inference.
--only is required: use the same comparison cohort for every tag.
--pull first runs sync.sh pull --predictions; otherwise use local predictions only.
--jobs accepts 1..4. Scorers run sequentially; default FT_*_SCALE thresholds are used.
Results/logs: .eval-cache/ft/results/gpu/TAG/{prepare,covers,solos,global}.log
              .eval-cache/ft/results/gpu/TAG/{covers,solos,global}.json
EOF
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
JOBS=4
PULL=0
ONLY=()
TAGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --jobs|--only)
            [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }
            if [[ "$1" == --jobs ]]; then
                JOBS=$2
            else
                [[ "$2" =~ ^[a-zA-Z0-9_]+(,[a-zA-Z0-9_]+)*$ ]] || {
                    echo '--only needs comma-separated channel names.' >&2; exit 2;
                }
                IFS=',' read -r -a ONLY <<< "$2"
            fi
            shift 2 ;;
        --pull) PULL=1; shift ;;
        --) shift; TAGS+=("$@"); break ;;
        -*) usage >&2; exit 2 ;;
        *) TAGS+=("$1"); shift ;;
    esac
done
[[ "$JOBS" =~ ^[1-4]$ && ${#ONLY[@]} -gt 0 && ${#TAGS[@]} -gt 0 ]] || { usage >&2; exit 2; }
for tag in "${TAGS[@]}"; do
    [[ "$tag" =~ ^[a-zA-Z0-9_]+$ ]] || { echo "Invalid tag: $tag (no '-')." >&2; exit 2; }
done
[[ "$(uname -s)" == Darwin ]] || { echo 'Run score_tags.sh on the Mac with its audio/truth caches.' >&2; exit 1; }
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/tools/finetune:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset FT_KICK_SCALE FT_SNARE_SCALE FT_TOM_SCALE FT_HIHAT_SCALE FT_CYMBAL_SCALE
PYTHON="$ROOT/.venv/bin/python"
if [[ "$PULL" == 1 ]]; then
    bash "$ROOT/tools/gpu/sync.sh" pull --predictions "${TAGS[@]}"
fi
for tag in "${TAGS[@]}"; do
    RESULTS="$ROOT/.eval-cache/ft/results/gpu/$tag"
    mkdir -p "$RESULTS"
    # ft_variants averages whatever folds it finds; require complete inputs
    # so a partial pull cannot silently change the solo ensemble.
    "$PYTHON" - "$tag" <<'PY'
import json
import sys
from pathlib import Path

tag = sys.argv[1]
cache = Path('.eval-cache/ft')
solos = sorted(p for p in (cache / 'solos').iterdir() if p.is_dir())
if not solos:
    raise SystemExit('No local solo feature caches; prepare/pull before scoring.')
for fold in range(3):
    name = f'{tag}-fold{fold}'
    checkpoint = cache / 'models' / f'{name}.pt'
    if not checkpoint.is_file():
        raise SystemExit(f'Missing {checkpoint}')
    metadata = json.loads((cache / 'models' / f'{name}.json').read_text())
    for folder in [cache / slug for slug in metadata['held']] + solos:
        prediction = folder / f'student-{name}.npy'
        if not prediction.is_file():
            raise SystemExit(f'Missing {prediction}; complete prediction and pull first.')
PY
    # A rerun may have replaced predictions under this tag. Rebuild only this
    # tag's derived snare cache locally; no audio or other tag's cache is removed.
    find "$ROOT/.eval-cache/ft" -type f -name "snare-student-$tag.npy" -delete
    "$PYTHON" tools/finetune/prepare_score.py --tag "$tag" --only "${ONLY[@]}" --jobs "$JOBS" \
        > "$RESULTS/prepare.log" 2>&1
    FT_TAG="$tag" "$PYTHON" tools/harvest/fill_score.py --detect ft_variants:tuned \
        --only "${ONLY[@]}" --jobs "$JOBS" --json "$RESULTS/covers.json" > "$RESULTS/covers.log" 2>&1
    FT_TAG="$tag" "$PYTHON" tools/truth_video/fill_seq.py --detect ft_variants:tuned \
        --jobs "$JOBS" --json "$RESULTS/solos.json" > "$RESULTS/solos.log" 2>&1
    FT_TAG="$tag" "$PYTHON" tools/harvest/global_score.py --detect ft_variants:tuned \
        --only "${ONLY[@]}" --json "$RESULTS/global.json" > "$RESULTS/global.log" 2>&1
    echo "$tag scored: $RESULTS"
done
