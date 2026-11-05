"""Mac controller for WSL harvest; setup prints commands, never installs.

push --init [--inputs], verify, fetch <harvest flags>, read <batch flags>,
compact (no --force), pull [--only CHANNEL ...] [--overwrite-local], status. Jobs are detached.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid

import harvest_transfer as transfer

ROOT = Path(__file__).resolve().parents[2]
HOST = os.environ.get('GPU_HOST', 'gpu-host')
REMOTE = os.environ.get('GPU_REMOTE_DIR', 'groovemap')
SSH = ['ssh', '-o', 'BatchMode=yes', HOST]


def call(argv, **kw):
    print(shlex.join(map(str, argv)), flush=True)
    return subprocess.run(argv, check=True, **kw)


def ssh(command: str, capture: bool = False):
    return call([*SSH, f'cd "$HOME/{REMOTE}" && export PATH="$PWD/.venv/bin:$HOME/.deno/bin:$PATH" && {command}'],
                stdin=subprocess.DEVNULL, capture_output=capture, text=True)


def rsync(source, destination, *options):
    call(['rsync', '-a', '--exclude=__pycache__/', '-e', 'ssh -o BatchMode=yes',
          *options, str(source), str(destination)], stdin=subprocess.DEVNULL)


def setup() -> None:
    print('''# Run in WSL; sudo is needed ONLY for these two apt commands.
sudo apt-get update
sudo apt-get install -y ffmpeg rsync util-linux curl unzip git build-essential python3-dev
cd ~/groovemap
# Existing venv and CUDA torch are required. Print deps and pin installed CUDA packages.
.venv/bin/python tools/gpu/harvest_deps.py freeze
# Matching torchaudio CUDA wheel; resolver is constrained to the existing torch.
~/.local/bin/uv pip install --python .venv/bin/python \\
  -c logs/harvest/deps/cuda-constraints.txt \\
  "torchaudio==$(.venv/bin/python -c 'import torch; print(torch.__version__)')" \\
  --index-url https://download.pytorch.org/whl/cu128
# All pyproject runtime deps, pinned ADTOF source, OpenCV and yt-dlp[default].
~/.local/bin/uv pip install --python .venv/bin/python \\
  -c logs/harvest/deps/cuda-constraints.txt -r logs/harvest/deps/harvest-requirements.txt
# Deno (yt-dlp's YouTube JS runtime) from PyPI into the venv; ssh() puts .venv/bin on PATH.
~/.local/bin/uv pip install --python .venv/bin/python deno
# torchcodec (imported by demucs.api): PyPI's build wants CUDA 13 libs and the cu128 build wants NPP;
# the CPU build is enough (audio reaches demucs as tensors) and uses the apt FFmpeg's shared libs.
~/.local/bin/uv pip install --python .venv/bin/python -c logs/harvest/deps/cuda-constraints.txt \\
  --reinstall-package torchcodec "torchcodec==0.11.1" --index-url https://download.pytorch.org/whl/cpu
~/.local/bin/uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python tools/gpu/harvest_deps.py verify
# No uv sync / uv run, no model downloads during verify.''')


def push(initialize: bool, inputs: bool, tools_only: bool = False) -> None:
    transfer.vault.config()  # Fail before sending any files if the seal is missing.
    transfer.vault.performance_table_file()  # reject malformed owner-authored groups before transfer
    token = '.harvest-transfer/' + uuid.uuid4().hex
    ssh('mkdir -p ' + shlex.quote(token + '/code'))
    try:
        # Nothing is sent to live code/templates/inputs before the writer lock.
        # --tools-only leaves WSL's src/ (what `stickmap transcribe` runs) untouched, e.g. while
        # src/ is mid-edit on the Mac.
        parts = ('tools',) if tools_only else ('pyproject.toml', 'uv.lock', 'src', 'tools')
        for name in parts:
            rsync(ROOT / name, f'{HOST}:{REMOTE}/{token}/code/')
        with tempfile.TemporaryDirectory(prefix='gm-harvest-push-') as temp:
            bundle = Path(temp)
            (bundle / 'code-parts.txt').write_text('\n'.join(parts) + '\n')
            for name in ('manifest.jsonl', 'vault.json', 'vault_log.jsonl'):
                source = transfer.vault.OUT / name
                (bundle / name).write_bytes(source.read_bytes() if source.exists() else b'')
            groups = transfer.vault.OUT / 'same_performance.json'
            if groups.exists():
                (bundle / groups.name).write_bytes(groups.read_bytes())
            rsync(str(bundle) + '/', f'{HOST}:{REMOTE}/{token}/')
        if inputs:
            excluded = ['--exclude=' + value for value in (
                '*.mp4', '*.webm', '*.mkv', '*.m4a', '*.png', '*.jpg', 'frames/',
                'video*', 'truth-history/', '.truth-work-*', '*.lock', '.wsl-*',
                'harvest/manifest.jsonl', 'harvest/vault.json', 'harvest/vault_log.jsonl',
                'harvest/same_performance.json')]
            ssh('mkdir -p ' + shlex.quote(token + '/inputs/truth') + ' ' +
                shlex.quote(token + '/inputs/.eval-cache/ft'))
            rsync(str(ROOT / 'truth') + '/', f'{HOST}:{REMOTE}/{token}/inputs/truth/', *excluded)
            rsync(str(ROOT / '.eval-cache/ft') + '/', f'{HOST}:{REMOTE}/{token}/inputs/.eval-cache/ft/',
                  '--exclude=student-*', '--exclude=snare-*', '--exclude=models/',
                  '--exclude=results/', '--exclude=smoke/')
            result = subprocess.check_output([sys.executable, str(ROOT / 'tools/gpu/truth_paths.py'), '--external'], text=True)
            for name in result.splitlines():
                path = Path(name)
                destination = token + '/inputs/truth/songs/' + path.name
                ssh('mkdir -p ' + shlex.quote(destination))
                rsync(str(path) + '/', f'{HOST}:{REMOTE}/{destination}/', *excluded)
        # Bootstrap with staged helpers, including on the very first push.
        ssh_cmd = ['.venv/bin/python', token + '/code/tools/gpu/harvest_transfer.py',
                   'publish', token, '--root', '.']
        if initialize:
            ssh_cmd.append('--init')
        if inputs:
            ssh_cmd.append('--inputs')
        ssh(shlex.join(ssh_cmd))
    finally:
        ssh('rm -rf -- ' + shlex.quote(token))
    transfer.atomic(transfer.vault.OUT / '.wsl-mirror', b'WSL is the sole manifest writer; do not run Mac batch.py.\n')


def detached(kind: str, args: list[str]) -> None:
    token = kind + '_' + uuid.uuid4().hex[:12]
    log = f'logs/harvest/{token}.log'
    job = ['.venv/bin/python', '-u', 'tools/gpu/harvest_job.py', '--job', f'logs/harvest/{token}.json']
    if kind == 'read':
        job += ['--lock']
    command = ['.venv/bin/python', '-u', 'tools/harvest/harvest.py', kind] if kind in ('fetch', 'compact') else [
        '.venv/bin/python', '-u', 'tools/truth_video/batch.py']
    if kind == 'fetch':
        command += ['--device', 'cuda', '--min-free-gb', '20']
    elif kind == 'read':
        command += ['--min-free-gb', '10']
        if '--latest-fetch' in args or '--reread' in args:
            command = ['.venv/bin/python', '-u', 'tools/gpu/harvest_read.py', '--min-free-gb', '10']
    job += ['--', *command, *args]
    # A trailing & backgrounds ssh()'s entire cd/export/mkdir && chain. That
    # shell can retain the SSH channel FDs while waiting for the job. setsid -f
    # always forks; only its fully redirected child survives the foreground
    # launcher. No shell background chain and no --wait.
    ssh('mkdir -p logs/harvest && setsid -f nohup ' + shlex.join(job) +
        f' > {shlex.quote(log)} 2>&1 < /dev/null')
    print(f'Job {token}; log: ~/{REMOTE}/{log}; '
          'use status to inspect exit. Background launch is not completion.')


def pull(only: list[str] | None, overwrite_local: bool = False) -> None:
    token = '.harvest-transfer/' + uuid.uuid4().hex
    args = ['.venv/bin/python', 'tools/gpu/harvest_transfer.py', 'snapshot', token]
    if only:
        args += ['--only', *only]
    ssh('mkdir -p .harvest-transfer && ' + shlex.join(args))
    try:
        with transfer.lock():
            # Same filesystem as SONGS is required for atomic directory swaps.
            with tempfile.TemporaryDirectory(prefix='.wsl-pull-', dir=transfer.vault.OUT) as temp:
                bundle = Path(temp)
                rsync(f'{HOST}:{REMOTE}/{token}/', str(bundle) + '/', '--exclude=songs/')
                transfer.validate_mirror(bundle)
                index = json.loads((bundle / 'snapshot.json').read_text())
                files = transfer.snapshot_files(bundle)
                transfer.check_local(files, overwrite_local)
                sizes = [row[1] for row in index['files']]
                slugs = {Path(name).parts[0] for name in files}
                extra = sum(path.stat().st_size for slug in slugs
                            for path in (transfer.vault.SONGS / slug).rglob('*')
                            if path.is_file() and str(path.relative_to(transfer.vault.SONGS)) not in files)
                needed = sum(sizes) + extra + max(sizes, default=0)
                free = shutil.disk_usage(transfer.vault.OUT).free
                if free - needed < 8e9:
                    raise RuntimeError(f'pull staging needs up to {needed / 1e9:.2f} GB plus 8 GB reserve; free {free / 1e9:.2f} GB')
                if files:
                    rsync(f'{HOST}:{REMOTE}/{token}/songs/', str(bundle / 'songs') + '/')
                transfer.publish_songs(bundle, overwrite_local)
                transfer.publish_mirror(bundle)
                print(f'Published verified songs and exact journal snapshot; compact output total {index["bytes"] / 1e9:.2f} GB.')
    finally:
        ssh('rm -rf -- ' + shlex.quote(token))


def main() -> None:
    if not re.fullmatch(r'[A-Za-z0-9_.@-]+', HOST) or HOST.startswith('-'):
        raise ValueError('unsafe GPU_HOST')
    if not re.fullmatch(r'[A-Za-z0-9_./-]+', REMOTE) or REMOTE.startswith('/') or '..' in Path(REMOTE).parts:
        raise ValueError('GPU_REMOTE_DIR must be a safe home-relative path')
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('setup')
    sub.add_parser('verify')
    push_parser = sub.add_parser('push')
    push_parser.add_argument('--init', action='store_true')
    push_parser.add_argument('--inputs', action='store_true')
    push_parser.add_argument('--tools-only', action='store_true', help='send tools/ only; keep WSL src/, pyproject, uv.lock')
    sub.add_parser('fetch')
    sub.add_parser('read')
    compact_parser = sub.add_parser('compact')
    compact_parser.add_argument('--only', nargs='+')
    sub.add_parser('status')
    pull_parser = sub.add_parser('pull')
    pull_parser.add_argument('--only', nargs='+')
    pull_parser.add_argument('--overwrite-local', action='store_true',
                             help='explicitly replace locally changed song files after inspecting/preserving them')
    a, rest = p.parse_known_args()
    if a.command not in ('fetch', 'read') and rest:
        p.error('unrecognized arguments: ' + shlex.join(rest))
    if a.command == 'setup':
        setup()
    elif a.command == 'push':
        push(a.init, a.inputs, a.tools_only)
    elif a.command == 'verify':
        ssh('.venv/bin/python tools/gpu/harvest_deps.py verify')
    elif a.command in ('fetch', 'read'):
        detached(a.command, rest)
    elif a.command == 'compact':
        flags = []
        if a.only:
            flags += ['--only', *a.only]
        detached('compact', flags)
    elif a.command == 'pull':
        pull(a.only, a.overwrite_local)
    else:
        ssh('.venv/bin/python tools/harvest/harvest.py status && '
            '.venv/bin/python tools/gpu/harvest_job.py --status')


if __name__ == '__main__':
    main()
