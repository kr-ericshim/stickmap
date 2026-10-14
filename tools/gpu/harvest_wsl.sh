#!/usr/bin/env bash
# Separate entry point: does not modify/use sync.sh's harvest journal handling.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "$TASK_ROOT/.venv/bin/python" "$TASK_ROOT/tools/gpu/harvest_wsl.py" "$@"
