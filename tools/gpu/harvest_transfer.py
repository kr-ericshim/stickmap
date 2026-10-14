"""Byte-preserving, fail-closed journal migration and compact output snapshots.

Invoked by harvest_wsl.py on either host. No JSONL reconstruction or merge.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/harvest'))
import manifest_io
import vault


def journal(path: Path) -> bytes:
    data = path.read_bytes() if path.exists() else b''
    if data and not data.endswith(b'\n'):
        raise ValueError(f'unfinished journal: {path}; preserve and repair it before transferring')
    for line in data.splitlines():
        json.loads(line)
    return data


def require_prefix(older: bytes, newer: bytes, label: str) -> None:
    if not newer.startswith(older):
        raise ValueError(f'{label} diverged; refusing to merge, reorder, duplicate or truncate rows')


def atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('wb') as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    temp.replace(path)


@contextmanager
def lock():
    vault.OUT.mkdir(parents=True, exist_ok=True)
    with (vault.OUT / '.harvest-writer.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('another harvest writer/snapshot/comparison is running; refusing to publish') from exc
        yield


def validate_import(bundle: Path, initialize: bool = False) -> bool:
    """Validate the complete migration before any code/input publication."""
    incoming = journal(bundle / 'manifest.jsonl')
    # Validate video keys as well as JSON before publishing an initial journal.
    list(manifest_io.records(bundle / 'manifest.jsonl'))
    current = journal(vault.MANIFEST)
    if not vault.MANIFEST.exists():
        if not initialize:
            raise ValueError('first migration requires push --init')
    elif initialize and not (vault.OUT / '.wsl-authority').exists() and incoming.startswith(current):
        # One explicit migration of an old rsync mirror, before any WSL writer.
        pass
    else:
        require_prefix(incoming, current, 'manifest (Mac must be a prefix of WSL)')
    local_log = journal(bundle / 'vault_log.jsonl')
    remote_log = journal(vault.LOG)
    if vault.LOG.exists():
        require_prefix(local_log, remote_log, 'vault log')
    elif local_log and not initialize:
        raise ValueError('initial vault log migration requires --init')
    data = json.loads((bundle / 'vault.json').read_text())
    groups = bundle / 'same_performance.json'
    vault.performance_table_file(groups)
    if not groups.exists() and (vault.OUT / groups.name).exists():
        raise ValueError('same_performance.json missing from push; refusing to discard remote groups')
    vault.timestamp(data['cutoff'])
    if not isinstance(data['video_ids'], list) or any(not isinstance(v, str) or not v for v in data['video_ids']):
        raise ValueError('invalid vault video_ids')
    if vault.CONFIG.exists():
        old = vault.config()
        if vault.timestamp(old['cutoff']) != vault.timestamp(data['cutoff']):
            raise ValueError('refusing to move the established vault cutoff')
        if not set(old['video_ids']).issubset(data['video_ids']):
            raise ValueError('refusing to unseal explicit vault videos')
    return not vault.MANIFEST.exists() or (
        initialize and not (vault.OUT / '.wsl-authority').exists() and incoming.startswith(current))


def import_bundle(bundle: Path, initialize: bool = False) -> None:
    """WSL is the manifest/log writer; Mac owns the vault configuration."""
    migrate = validate_import(bundle, initialize)
    local_log = journal(bundle / 'vault_log.jsonl')
    if migrate:
        atomic(vault.MANIFEST, journal(bundle / 'manifest.jsonl'))
    if not vault.LOG.exists() and local_log:
        atomic(vault.LOG, local_log)
    atomic(vault.CONFIG, (bundle / 'vault.json').read_bytes())
    groups = bundle / 'same_performance.json'
    if groups.exists():
        atomic(vault.OUT / groups.name, groups.read_bytes())
    # This flag is informational on WSL; .wsl-mirror is enforced on the Mac.
    atomic(vault.OUT / '.wsl-authority', b'WSL is the sole manifest and vault_log writer.\n')


def publish_push(bundle: Path, initialize: bool = False, inputs: bool = False) -> None:
    """Run from staged code, holding the live writer lock through publication."""
    with lock():
        if (vault.OUT / '.wsl-mirror').exists():
            raise RuntimeError('push publication belongs to WSL, not the Mac mirror')
        validate_import(bundle, initialize)
        listed = bundle / 'code-parts.txt'
        parts = listed.read_text().split() if listed.exists() else ['pyproject.toml', 'uv.lock', 'src', 'tools']
        if not set(parts) <= {'pyproject.toml', 'uv.lock', 'src', 'tools'} or 'tools' not in parts:
            raise ValueError(f'unexpected staged code parts: {parts}')
        for name in parts:
            if not (bundle / 'code' / name).exists():
                raise FileNotFoundError(f'incomplete staged code: {name}')
        if inputs and not all((bundle / 'inputs' / name).is_dir() for name in ('truth', '.eval-cache/ft')):
            raise FileNotFoundError('incomplete staged inputs')
        subprocess.run(['rsync', '-a', str(bundle / 'code') + '/', str(ROOT) + '/'], check=True)
        if inputs:
            subprocess.run(['rsync', '-a', str(bundle / 'inputs') + '/', str(ROOT) + '/'], check=True)
            subprocess.run([sys.executable, str(ROOT / 'tools/gpu/truth_paths.py'), '--rebase'],
                           cwd=ROOT, check=True)
        import_bundle(bundle, initialize)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def snapshot(destination: Path, only: list[str] | None = None) -> None:
    """Hard-link immutable compact outputs; copy journals while holding writer lock.

    Canonical truth only: histories/frames/QA bitmaps stay on WSL. Every NPY,
    JSON, Opus and textual rejection artifact in canonical truth is retained.
    """
    seal = vault.Vault()
    destination.mkdir(parents=True, exist_ok=False)
    atomic(destination / 'manifest.jsonl', journal(vault.MANIFEST))
    atomic(destination / 'vault.json', vault.CONFIG.read_bytes())
    atomic(destination / 'vault_log.jsonl', journal(vault.LOG))
    groups = vault.OUT / 'same_performance.json'
    if groups.exists():
        atomic(destination / groups.name, groups.read_bytes())
    total = 0
    files = []
    for row in seal.rows.values():
        if row.get('status') != 'compacted' or (only and row.get('channel') not in only):
            continue
        slug = row.get('slug')
        if not slug or Path(slug).name != slug or slug in ('.', '..'):
            raise ValueError('unsafe compacted slug')
        song = vault.SONGS / slug
        if not song.is_dir():
            raise FileNotFoundError(f'compacted manifest folder missing: {song}; push --inputs first')
        if song.resolve().parent != vault.SONGS.resolve():
            raise ValueError('song symlink escapes harvest root')
        if (song / 'truth').is_symlink():
            raise ValueError('canonical truth must not be a symlink')
        for path in [*song.iterdir(), *(song / 'truth').rglob('*')]:
            if not path.is_file() or path.is_symlink():
                continue
            if path.suffix not in ('.npy', '.opus', '.json', '.txt', '.md', '.log'):
                continue
            target = destination / 'songs' / slug / path.relative_to(song)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)
            total += path.stat().st_size
            files.append([str(target.relative_to(destination / 'songs')), path.stat().st_size,
                          int(path.stat().st_mtime), digest(target)])
    atomic(destination / 'snapshot.json', json.dumps({'bytes': total, 'files': files}).encode())


def validate_mirror(bundle: Path) -> None:
    require_prefix(journal(vault.MANIFEST), journal(bundle / 'manifest.jsonl'), 'manifest')
    require_prefix(journal(vault.LOG), journal(bundle / 'vault_log.jsonl'), 'vault log')
    if vault.CONFIG.read_bytes() != (bundle / 'vault.json').read_bytes():
        raise ValueError('vault config changed locally; push it before pulling')
    groups = vault.OUT / 'same_performance.json'
    remote_groups = bundle / groups.name
    if ((groups.read_bytes() if groups.exists() else None) !=
            (remote_groups.read_bytes() if remote_groups.exists() else None)):
        raise ValueError('same_performance.json differs; reconcile the Mac-owned table and push it before pulling')


def snapshot_files(bundle: Path) -> dict[str, str]:
    files = {}
    for record in json.loads((bundle / 'snapshot.json').read_text())['files']:
        if len(record) != 4:
            raise ValueError('snapshot lacks SHA-256 hashes; run harvest_wsl.sh push after the WSL job finishes, then retry pull')
        name, size, mtime, sha = record
        path = Path(name)
        if (path.is_absolute() or len(path.parts) < 2 or '..' in path.parts or
                path.parts[0].startswith('.') or str(path) != name or name in files):
            raise ValueError(f'unsafe or duplicate snapshot path: {name}')
        if not isinstance(sha, str) or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError(f'invalid snapshot hash: {name}')
        files[name] = sha
    return files


def pulled_files() -> dict[str, str]:
    path = vault.OUT / '.wsl-pulled.json'
    return json.loads(path.read_text())['files'] if path.exists() else {}


def local_hash(name: str) -> str | None:
    path = vault.SONGS / name
    if any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(vault.SONGS)):
        raise ValueError(f'local song symlink; refusing to publish: {name}')
    if path.exists() and not path.is_file():
        raise ValueError(f'local song path is not a file: {name}')
    return digest(path) if path.exists() else None


def check_local(files: dict[str, str], overwrite_local: bool = False) -> None:
    previous = pulled_files()
    slugs = {Path(name).parts[0] for name in files}
    managed = {name: sha for name, sha in previous.items() if Path(name).parts[0] in slugs}
    conflicts = []
    for name in sorted(managed.keys() | files.keys()):
        current = local_hash(name)
        if name in managed:
            conflict = current != managed[name]
        else:
            # First pull/bootstrap: existing files must match the remote copy.
            conflict = current is not None and current != files[name]
        if conflict:
            conflicts.append(name)
    if conflicts and not overwrite_local:
        raise ValueError('local song files changed; nothing overwritten. Inspect/preserve them, then '
                         'use pull --overwrite-local only to replace them explicitly:\n' + '\n'.join(conflicts))


def atomic_song(stage: Path, target: Path) -> None:
    """Replace an entire song in one rename, including existing nonempty dirs."""
    if not target.exists():
        os.replace(stage, target)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    source, destination = os.fsencode(stage), os.fsencode(target)
    if sys.platform == 'darwin':
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        result = rename(source, destination, 2)  # RENAME_SWAP
    elif sys.platform.startswith('linux'):
        rename = libc.renameat2
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        result = rename(-100, source, -100, destination, 2)  # RENAME_EXCHANGE
    else:
        raise RuntimeError('atomic song replacement requires macOS or Linux')
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(target))


def publish_songs(bundle: Path, overwrite_local: bool = False) -> None:
    files = snapshot_files(bundle)
    validate_mirror(bundle)
    check_local(files, overwrite_local)
    # Validate every received file before publishing any song.
    for name, sha in files.items():
        path = bundle / 'songs' / name
        if path.is_symlink() or not path.is_file() or digest(path) != sha:
            raise ValueError(f'staged song hash mismatch: {name}; nothing overwritten')
    previous = pulled_files()
    slugs = sorted({Path(name).parts[0] for name in files})
    # Preserve local-only artifacts (videos/history/etc.) in the staged song.
    for slug in slugs:
        target, stage = vault.SONGS / slug, bundle / 'songs' / slug
        if target.exists():
            for path in target.rglob('*'):
                relative = path.relative_to(target)
                destination = stage / relative
                if path.is_symlink():
                    raise ValueError(f'local song symlink; refusing to publish: {path}')
                if path.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                elif not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
    # Catch local edits during the transfer/staging before the first exchange.
    check_local(files, overwrite_local)
    vault.SONGS.mkdir(parents=True, exist_ok=True)
    for slug in slugs:
        atomic_song(bundle / 'songs' / slug, vault.SONGS / slug)
        previous = {name: sha for name, sha in previous.items() if Path(name).parts[0] != slug}
        previous.update({name: sha for name, sha in files.items() if Path(name).parts[0] == slug})
        atomic(vault.OUT / '.wsl-pulled.json', json.dumps({'files': previous}, sort_keys=True).encode())


def publish_mirror(bundle: Path) -> None:
    """Called only after staged songs are published; preserve exact journal bytes."""
    validate_mirror(bundle)
    for name in ('manifest.jsonl', 'vault_log.jsonl'):
        atomic(vault.OUT / name, journal(bundle / name))
    atomic(vault.OUT / '.wsl-mirror', b'Use tools/gpu/harvest_wsl.sh; do not run Mac batch.py.\n')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('import', 'snapshot', 'publish'))
    p.add_argument('bundle', type=Path)
    p.add_argument('--init', action='store_true')
    p.add_argument('--only', nargs='+')
    p.add_argument('--inputs', action='store_true')
    p.add_argument('--root', type=Path, help='live repo root when invoked from staged code')
    a = p.parse_args()
    if a.root:
        global ROOT
        ROOT = a.root.resolve()
        vault.ROOT = ROOT
        vault.OUT = ROOT / 'truth/harvest'
        vault.CONFIG = vault.OUT / 'vault.json'
        vault.MANIFEST = vault.OUT / 'manifest.jsonl'
        vault.LOG = vault.OUT / 'vault_log.jsonl'
        vault.SONGS = vault.OUT / 'songs'
    if a.command == 'publish':
        publish_push(a.bundle, a.init, a.inputs)
        return
    with lock():
        if a.command == 'import':
            import_bundle(a.bundle, a.init)
        else:
            snapshot(a.bundle, a.only)


if __name__ == '__main__':
    main()
