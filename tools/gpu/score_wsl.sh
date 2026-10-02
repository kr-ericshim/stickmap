#!/usr/bin/env bash
# WSL CPU scoring: launch up to two ready tags while CUDA jobs continue.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash tools/gpu/score_wsl.sh --only CHANNEL[,CHANNEL...] [options] TAG [TAG ...]
Run in the WSL venv with truth/audio caches and complete CUDA predictions.
--only is required: choose the same comparison cohort for every tag.
--jobs N          workers per tag, 1..12 (default 6; preparation uses at most 4).
--parallel N      tags scored at once, 1..2 (default 2). Scorers within a tag are sequential.
--queue-log FILE  wait for "TAG done" from train_tags.py (default logs_run.out).
--no-wait         score existing predictions without watching a queue log.
Results/logs: scores/TAG/{covers,solos,global}.json and {prepare,covers,solos,global}.log.
Successful tags append "TAG scored <timestamp>" to scores/done.txt; failures exit nonzero.
EOF
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
JOBS=6
PARALLEL=2
QUEUE_LOG=logs_run.out
WAIT_QUEUE=1
ONLY=()
TAGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --jobs|--parallel|--only|--queue-log)
            [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }
            case "$1" in
                --jobs) JOBS=$2 ;;
                --parallel) PARALLEL=$2 ;;
                --queue-log) QUEUE_LOG=$2 ;;
                --only)
                    [[ "$2" =~ ^[a-zA-Z0-9_]+(,[a-zA-Z0-9_]+)*$ ]] || {
                        echo '--only needs comma-separated channel names.' >&2; exit 2;
                    }
                    IFS=',' read -r -a ONLY <<< "$2" ;;
            esac
            shift 2 ;;
        --no-wait) WAIT_QUEUE=0; shift ;;
        --) shift; TAGS+=("$@"); break ;;
        -*) usage >&2; exit 2 ;;
        *) TAGS+=("$1"); shift ;;
    esac
done
[[ "$JOBS" =~ ^([1-9]|1[0-2])$ && "$PARALLEL" =~ ^[12]$ &&
   ${#ONLY[@]} -gt 0 && ${#TAGS[@]} -gt 0 ]] || { usage >&2; exit 2; }
SEEN=' '
for tag in "${TAGS[@]}"; do
    [[ "$tag" =~ ^[a-zA-Z0-9_]+$ ]] || { echo "Invalid tag: $tag (no '-')." >&2; exit 2; }
    [[ "$SEEN" != *" $tag "* ]] || { echo "Duplicate tag: $tag" >&2; exit 2; }
    SEEN+="$tag "
done
[[ "$(uname -s)" == Linux ]] || { echo 'Run score_wsl.sh on WSL; use score_tags.sh on the Mac.' >&2; exit 1; }
cd "$ROOT"
[[ -d truth && -x .venv/bin/python ]] || { echo 'Missing truth/ or WSL venv; use sync.sh push --truth first.' >&2; exit 1; }
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/tools/finetune:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset FT_KICK_SCALE FT_SNARE_SCALE FT_TOM_SCALE FT_HIHAT_SCALE FT_CYMBAL_SCALE
PYTHON="$ROOT/.venv/bin/python"
PREP_JOBS=$JOBS
(( PREP_JOBS <= 4 )) || PREP_JOBS=4
mkdir -p scores
# One scheduler owns these tag caches/results at a time. The lock is released
# automatically on exit, and prevents overlapping reruns of the same tag.
exec 9> scores/.score.lock
flock -n 9 || { echo 'Another WSL scoring scheduler is already running.' >&2; exit 1; }

score_one() {
    local tag=$1 results="$ROOT/scores/$1"
    mkdir -p "$results"
    # Require all held-out predictions and all three solo folds, so averaging
    # cannot silently accept a partially completed prediction job.
    "$PYTHON" - "$tag" <<'PY'
import json
import sys
from pathlib import Path

tag = sys.argv[1]
cache = Path('.eval-cache/ft')
solos = sorted(p for p in (cache / 'solos').iterdir() if p.is_dir())
if not solos:
    raise SystemExit('No solo feature caches; prepare on the Mac and push first.')
for fold in range(3):
    name = f'{tag}-fold{fold}'
    checkpoint = cache / 'models' / f'{name}.pt'
    if not checkpoint.is_file():
        raise SystemExit(f'Missing {checkpoint}')
    metadata = json.loads((cache / 'models' / f'{name}.json').read_text())
    for folder in [cache / slug for slug in metadata['held']] + solos:
        prediction = folder / f'student-{name}.npy'
        if not prediction.is_file():
            raise SystemExit(f'Missing {prediction}; complete CUDA prediction first.')
PY
    find "$ROOT/.eval-cache/ft" -type f -name "snare-student-$tag.npy" -delete
    "$PYTHON" tools/finetune/prepare_score.py --tag "$tag" --only "${ONLY[@]}" --jobs "$PREP_JOBS" \
        > "$results/prepare.log" 2>&1
    FT_TAG="$tag" "$PYTHON" tools/harvest/fill_score.py --detect ft_variants:tuned \
        --only "${ONLY[@]}" --jobs "$JOBS" --json "$results/covers.json.tmp" > "$results/covers.log" 2>&1
    FT_TAG="$tag" "$PYTHON" tools/truth_video/fill_seq.py --detect ft_variants:tuned \
        --jobs "$JOBS" --json "$results/solos.json.tmp" > "$results/solos.log" 2>&1
    FT_TAG="$tag" "$PYTHON" tools/harvest/global_score.py --detect ft_variants:tuned \
        --only "${ONLY[@]}" --json "$results/global.json.tmp" > "$results/global.log" 2>&1
    for scorer in covers solos global; do
        mv "$results/$scorer.json.tmp" "$results/$scorer.json"
    done
    printf '%s scored %s\n' "$tag" "$(date -Is)" >> scores/done.txt
    echo "$tag scored: $results"
}

PIDS=()
ACTIVE_TAGS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        # Stop the per-tag shell and its current scorer if interrupted.
        pkill -TERM -P "$pid" 2>/dev/null || true
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
PENDING=("${TAGS[@]}")
FAILED=0
echo "Waiting/scoring ${#TAGS[@]} tags; parallel=$PARALLEL, workers/tag=$JOBS, queue=$QUEUE_LOG"
while (( ${#PENDING[@]} > 0 || ${#PIDS[@]} > 0 )); do
    next_pids=()
    next_active=()
    for i in "${!PIDS[@]}"; do
        if kill -0 "${PIDS[$i]}" 2>/dev/null; then
            next_pids+=("${PIDS[$i]}")
            next_active+=("${ACTIVE_TAGS[$i]}")
        elif wait "${PIDS[$i]}"; then
            :
        else
            echo "${ACTIVE_TAGS[$i]} scoring failed; see scores/${ACTIVE_TAGS[$i]}/*.log" >&2
            FAILED=1
        fi
    done
    PIDS=("${next_pids[@]}")
    ACTIVE_TAGS=("${next_active[@]}")
    next_pending=()
    for tag in "${PENDING[@]}"; do
        if [[ "$WAIT_QUEUE" == 1 ]] && [[ -f "$QUEUE_LOG" ]] && grep -qE "^$tag failed:" "$QUEUE_LOG"; then
            echo "$tag training/prediction failed; skipping scoring (see $QUEUE_LOG)." >&2
            FAILED=1
        elif (( ${#PIDS[@]} < PARALLEL )) && {
            [[ "$WAIT_QUEUE" == 0 ]] || { [[ -f "$QUEUE_LOG" ]] && grep -qx "$tag done" "$QUEUE_LOG"; };
        }; then
            echo "$tag scoring started"
            score_one "$tag" < /dev/null &
            PIDS+=("$!")
            ACTIVE_TAGS+=("$tag")
        else
            next_pending+=("$tag")
        fi
    done
    PENDING=("${next_pending[@]}")
    if (( ${#PENDING[@]} > 0 || ${#PIDS[@]} > 0 )); then sleep 5; fi
done
exit "$FAILED"
