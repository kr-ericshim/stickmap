"""One local agy batch at a time, with a shared quota stop signal."""
import contextlib
import fcntl
import os
import re
import signal
import subprocess
import time
from common import DATA, confined, dump


def quota_text(text):
    return bool(re.search(r'RESOURCE_EXHAUSTED|\b(?:code\s+|HTTP\s+|error_code["\s:]+)429\b|429\s+Too Many|Individual quota reached', text, re.I))


@contextlib.contextmanager
def batch_lock(owner):
    path = confined(DATA/'agy.lock')
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another vision_read agy batch is active; defer this batch')
        dump(DATA/'agy-owner.json', {'pid': os.getpid(), 'owner': owner, 'started_unix': time.time()})
        try:
            yield
        finally:
            dump(DATA/'agy-owner.json', {'pid': os.getpid(), 'owner': owner, 'finished_unix': time.time()})
            fcntl.flock(handle, fcntl.LOCK_UN)


def terminate(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def invoke(argv, folder, out_path, err_path, seconds, stop):
    """Poll local logs; first quota error stops all in-flight and pending jobs."""
    if stop.is_set():
        return 'quota_cancelled', None
    tick = time.monotonic()
    with out_path.open('w') as out, err_path.open('w') as err:
        proc = subprocess.Popen(argv, cwd=folder, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                start_new_session=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        while True:
            logs = err_path.read_text(errors='replace') + out_path.read_text(errors='replace')
            if quota_text(logs):
                stop.set()
                terminate(proc)
                return 'quota_exhausted', proc.returncode
            if stop.is_set():
                terminate(proc)
                return 'quota_cancelled', proc.returncode
            code = proc.poll()
            if code is not None:
                return ('ok' if code == 0 else 'cli_error'), code
            if time.monotonic()-tick >= seconds:
                terminate(proc)
                return 'timeout', proc.returncode
            time.sleep(.25)
