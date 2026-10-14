"""Detached job exit evidence; reader jobs share the harvest writer lock."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from harvest_transfer import atomic, lock, vault

ROOT = Path(__file__).resolve().parents[2]

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path)
    p.add_argument('--lock', action='store_true')
    p.add_argument('--status', action='store_true')
    p.add_argument('argv', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.status:
        rows = vault.Vault().rows
        for path in sorted((ROOT / 'logs/harvest').glob('*.json')):
            state = json.loads(path.read_text())
            if 'pid' not in state:
                continue
            if state['exit'] is None:
                try:
                    os.kill(state['pid'], 0)
                    outcome = 'running (PID present; completion not established)'
                except ProcessLookupError:
                    outcome = 'stopped without exit evidence; inspect log'
            else:
                outcome = 'success' if state['exit'] == 0 else f'FAILED exit={state["exit"]}'
            print(path.name, outcome, 'videos=', state.get('videos', []))
            for video in state.get('videos', []):
                row = rows.get(video, {})
                result = vault.SONGS / (row.get('slug') or '') / 'truth/result.json'
                stats = json.loads(result.read_text()) if result.exists() else {}
                print(' ', video, row.get('status'),
                      f'strict={stats.get("strict_notes", 0)} notes/{stats.get("strict_seconds", 0):.2f}s',
                      f'scoring={stats.get("scoring_notes", 0)} notes/{stats.get("scoring_seconds", 0):.2f}s')
            log = path.with_suffix('.log')
            if log.exists():
                print('\n'.join(log.read_text(errors='replace').splitlines()[-8:]))
        sys.exit(0)
    if not a.job:
        p.error('--job is required without --status')
    command = a.argv[1:] if a.argv[:1] == ['--'] else a.argv
    state = {'pid': os.getpid(), 'started': time.time(), 'command': command, 'exit': None}
    atomic(a.job, json.dumps(state).encode())
    code = 1
    before = vault.Vault().rows
    try:
        with lock() if a.lock else nullcontext():
            # The wrapper already has the detached log as stdout/stderr. Its
            # child gets no interactive stdin or other inherited SSH FDs.
            code = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                  close_fds=True).returncode
    finally:
        state.update(exit=code, finished=time.time())
        state['videos'] = [v for v, row in vault.Vault().rows.items() if row != before.get(v)]
        atomic(a.job, json.dumps(state).encode())
        if a.job.name.startswith('fetch_'):
            atomic(ROOT / 'logs/harvest/latest_fetch.txt', (str(a.job) + '\n').encode())
    sys.exit(code)
