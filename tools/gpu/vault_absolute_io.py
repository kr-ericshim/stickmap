"""Local filesystem and offline guards shared by absolute-vault children."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys


def safe(path, root):
    root = Path(root).resolve()
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(root):
        raise ValueError(f'path escapes repository: {path}')
    for item in (path, *path.parents):
        if item == root:
            break
        if item.is_symlink():
            raise ValueError(f'symlink forbidden: {item}')
    return path


def tree(path, root):
    path = safe(path, root)
    if path.exists() and any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError(f'symlink in tree: {path}')
    return path


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Durable atomic replacement; destinations are guarded by the caller."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temp.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    sync_directory(path.parent)


def sync_directory(folder):
    fd = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def hashes(paths, root):
    return {safe(p, root).relative_to(root).as_posix(): sha(p) for p in sorted(set(paths))}


def verify(values, root):
    for relative, expected in values.items():
        if sha(safe(root / relative, root)) != expected:
            raise ValueError(f'frozen input changed: {relative}')


@contextmanager
def guard(*, writes=(), reads_forbidden=(), offline=True, subprocesses=True):
    """Python audit enforcement; OpenCV/ffmpeg get explicit checked paths too.

    A scoped hook remains registered but becomes inert on exit. Directory
    creation may traverse ancestors; file writes/removal/rename may not.
    Network use is forbidden, including implicit weight downloads.
    """
    active = True
    destinations = tuple(Path(p).resolve() for p in writes)
    forbidden = tuple(Path(p).resolve() for p in reads_forbidden)

    def path_of(value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            raise PermissionError('unverifiable filesystem destination')
        return Path(os.fsdecode(value)).resolve()

    def writable(value, mkdir=False):
        path = path_of(value)
        if not any(path.is_relative_to(p) or (mkdir and p.is_relative_to(path)) for p in destinations):
            raise PermissionError(f'write outside vault/final destinations: {path}')

    def audit(event, args):
        if not active:
            return
        if event == 'open':
            value, mode, flags = args
            writing = ((isinstance(mode, str) and any(c in mode for c in 'wax+')) or
                       bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if writing:
                # Logging libraries use /dev/null; it is not a file output.
                if not (isinstance(value, (str, bytes, os.PathLike)) and path_of(value) == Path(os.devnull)):
                    writable(value)
            elif isinstance(value, (str, bytes, os.PathLike)):
                path = path_of(value)
                if any(path.is_relative_to(p) for p in forbidden):
                    raise PermissionError(f'sealed read before durable opening: {path}')
        elif event in ('os.rename', 'os.link'):
            writable(args[0]); writable(args[1])
        elif event == 'os.symlink':
            raise PermissionError('symlink creation forbidden')
        elif event == 'os.mkdir':
            writable(args[0], mkdir=True)
        elif event in ('os.remove', 'os.rmdir', 'os.chmod', 'os.chown', 'os.utime',
                       'os.setxattr', 'os.removexattr', 'os.truncate'):
            writable(args[0])
        elif offline and event in ('socket.connect', 'socket.getaddrinfo'):
            raise PermissionError('network forbidden in absolute vault evaluation')
        elif not subprocesses and event == 'subprocess.Popen':
            raise PermissionError('subprocess forbidden in canonical CV read')

    sys.addaudithook(audit)
    try:
        yield
    finally:
        active = False
