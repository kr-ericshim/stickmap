"""Append-only shared journal. A partial final write is retried by the next poll."""
import fcntl
import json
import os
import time
from pathlib import Path


def records(path):
    """Yield committed rows in journal order, retaining first-seen timestamps."""
    path = Path(path)
    if not path.exists():
        return
    data = path.read_bytes()
    for number, line in enumerate(data.splitlines(keepends=True), 1):
        if not line.endswith(b'\n'):
            break
        try:
            row = json.loads(line)
            row['video']
        except (ValueError, KeyError) as e:
            raise ValueError(f'{path}:{number}: malformed committed manifest record') from e
        yield row


def read(path):
    rows = {}
    for row in records(path):
        key = row['video']
        rows[key] = {**rows.get(key, {}), **row}
    return rows


def append(path, **row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps({**row, 'at': time.time()}, ensure_ascii=False) + '\n').encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        # Do not concatenate onto a crashed writer's incomplete JSON record.
        # Recovery must preserve/repair that record explicitly, never truncate.
        with path.open('rb') as check:
            check.seek(0,os.SEEK_END)
            if check.tell():
                check.seek(-1,os.SEEK_END)
                if check.read(1)!=b'\n':raise ValueError('manifest has an unfinished final record; retry after writer completes')
        # One append syscall also coexists with an already-running old fetcher.
        if os.write(fd, payload) != len(payload):
            raise OSError('short manifest append')
        os.fsync(fd)
    finally:
        os.close(fd)
