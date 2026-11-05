#!/usr/bin/env bash
# Mac -> WSL code/cache and optional truth; WSL -> Mac score JSONs by default.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash tools/gpu/sync.sh push [--truth]
       bash tools/gpu/sync.sh pull [--predictions] TAG [TAG ...]
push: code and training/feature caches; install status.py as ~/.local/bin/gm-status.
--truth: also copy truth/audio, excluding *.mp4 *.webm *.mkv *.m4a *.png *.jpg
         frames/ video* truth-history/. Rebase remote solo song.txt paths.
pull: scores/TAG/{covers,solos,global}.json and scores/done.txt only.
--predictions: instead pull student-TAG-fold{0,1,2}.npy and final checkpoints
               for the Mac score_tags.sh fallback (no scores required).
Environment: GPU_HOST=gpu-host (an SSH alias), GPU_REMOTE_DIR=groovemap (relative to remote home).
No deletion, venv sync, or shipped-weight replacement. Audio transfer is opt-in.
EOF
}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HOST=${GPU_HOST:-gpu-host}
REMOTE=${GPU_REMOTE_DIR:-groovemap}
ACTION=${1:-}
case "$ACTION" in
    -h|--help) usage; exit 0 ;;
    push|pull) shift ;;
    *) usage >&2; exit 2 ;;
esac
# Keep paths usable by rsync's remote shell without extra shell quoting.
if [[ ! "$HOST" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.@-]*$ ||
      ! "$REMOTE" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_./-]*$ ]]; then
    echo 'GPU_HOST/GPU_REMOTE_DIR must have no spaces or shell metacharacters.' >&2
    exit 2
fi
REMOTE=${REMOTE%/}
PUSH_TRUTH=0
PULL_PREDICTIONS=0
TAGS=()
TAG_COUNT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --truth) [[ "$ACTION" == push ]] || { usage >&2; exit 2; }; PUSH_TRUTH=1; shift ;;
        --predictions) [[ "$ACTION" == pull ]] || { usage >&2; exit 2; }; PULL_PREDICTIONS=1; shift ;;
        -*) usage >&2; exit 2 ;;
        *) TAGS+=("$1"); TAG_COUNT=$((TAG_COUNT + 1)); shift ;;
    esac
done
if [[ "$ACTION" == push && "$TAG_COUNT" -ne 0 ]] || [[ "$ACTION" == pull && "$TAG_COUNT" -eq 0 ]]; then
    usage >&2; exit 2
fi
if [[ "$ACTION" == pull ]]; then
    # Mac's Bash 3.2 treats an empty array expansion as unset with set -u.
    for tag in "${TAGS[@]}"; do
        if [[ ! "$tag" =~ ^[a-zA-Z0-9_]+$ ]]; then
            echo "Invalid tag: $tag (use letters, digits, underscore; no '-')." >&2
            exit 2
        fi
    done
fi

transfer() {
    # awk succeeds even when there is no output; pipefail preserves rsync errors.
    rsync -a -e 'ssh -o BatchMode=yes' "$@" < /dev/null 2>&1 | awk '!/^#/'
}

remote_command() {
    ssh -o BatchMode=yes "$HOST" "$1" < /dev/null 2>&1 | awk '!/^#/'
}

if [[ "$ACTION" == push ]]; then
    [[ -d "$ROOT/.eval-cache/ft" ]] || { echo 'Missing .eval-cache/ft; prepare caches on the Mac first.' >&2; exit 1; }
    external_songs=''
    if [[ "$PUSH_TRUTH" == 1 ]]; then
        [[ -d "$ROOT/truth" ]] || { echo 'Missing local truth/.' >&2; exit 1; }
        external_songs=$(PYTHONDONTWRITEBYTECODE=1 "$ROOT/.venv/bin/python" "$ROOT/tools/gpu/truth_paths.py" --external)
    fi
    remote_command "mkdir -p '$REMOTE/.eval-cache/ft'"
    transfer --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' \
        "$ROOT/pyproject.toml" "$ROOT/uv.lock" "$ROOT/src" "$ROOT/tools" "$HOST:$REMOTE/"
    transfer --exclude 'student-*' --exclude 'snare-*' \
        --exclude 'models/' --exclude 'results/' --exclude 'smoke/' \
        --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' \
        "$ROOT/.eval-cache/ft/" "$HOST:$REMOTE/.eval-cache/ft/"
    if [[ "$PUSH_TRUTH" == 1 ]]; then
        truth_excludes=(--exclude '*.mp4' --exclude '*.webm' --exclude '*.mkv' --exclude '*.m4a'
                        --exclude '*.png' --exclude '*.jpg' --exclude 'frames/'
                        --exclude 'video*' --exclude 'truth-history/'
                        --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store')
        transfer "${truth_excludes[@]}" "$ROOT/truth" "$HOST:$REMOTE/"
        # Some solo song.txt entries point to Mac songs/ rather than truth/.
        # Copy only those referenced audio/cache folders into the WSL truth tree.
        if [[ -n "$external_songs" ]]; then
            remote_command "mkdir -p '$REMOTE/truth/songs'"
            while IFS= read -r song_dir; do
                transfer "${truth_excludes[@]}" "$song_dir" "$HOST:$REMOTE/truth/songs/"
            done <<< "$external_songs"
        fi
        remote_command "cd '$REMOTE' && PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/gpu/truth_paths.py --rebase"
    fi
    # Install a copy, not a symlink, so gm-status remains a standalone command.
    remote_command "mkdir -p \"\$HOME/.local/bin\" && install -m 755 '$REMOTE/tools/gpu/status.py' \"\$HOME/.local/bin/gm-status\""
    echo 'Code/cache pushed; gm-status installed (truth/audio copied only with --truth).'
elif [[ "$PULL_PREDICTIONS" == 1 ]]; then
    mkdir -p "$ROOT/.eval-cache/ft/models"
    for tag in "${TAGS[@]}"; do
        predictions=()
        models=()
        for fold in 0 1 2; do
            predictions+=(--include "student-$tag-fold$fold.npy")
            models+=(--include "$tag-fold$fold.pt" --include "$tag-fold$fold.json")
        done
        transfer --prune-empty-dirs --include '*/' "${predictions[@]}" --exclude '*' \
            "$HOST:$REMOTE/.eval-cache/ft/" "$ROOT/.eval-cache/ft/"
        transfer "${models[@]}" --exclude '*' \
            "$HOST:$REMOTE/.eval-cache/ft/models/" "$ROOT/.eval-cache/ft/models/"
        for fold in 0 1 2; do
            for suffix in pt json; do
                [[ -f "$ROOT/.eval-cache/ft/models/$tag-fold$fold.$suffix" ]] || {
                    echo "Missing $tag-fold$fold.$suffix after pull; check the completed WSL run." >&2
                    exit 1
                }
            done
        done
        echo "$tag pulled (final checkpoints and predictions only)"
    done
else
    # Check remote inputs first; include rules alone would succeed even if a
    # score file were missing, potentially leaving an old local JSON in place.
    checks="test -f '$REMOTE/scores/done.txt'"
    score_includes=(--include '/done.txt')
    for tag in "${TAGS[@]}"; do
        score_includes+=(--include "/$tag/")
        for scorer in covers solos global; do
            checks+=" && test -f '$REMOTE/scores/$tag/$scorer.json'"
            score_includes+=(--include "/$tag/$scorer.json")
        done
    done
    remote_command "$checks" || {
        echo 'Missing WSL score JSONs/done.txt; finish scoring the requested tags first.' >&2
        exit 1
    }
    mkdir -p "$ROOT/scores"
    transfer "${score_includes[@]}" --exclude '*' "$HOST:$REMOTE/scores/" "$ROOT/scores/"
    for tag in "${TAGS[@]}"; do
        echo "$tag pulled (score JSONs only)"
    done
fi
