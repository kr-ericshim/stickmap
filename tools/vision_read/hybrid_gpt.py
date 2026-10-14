"""Codex image backend for the frozen hybrid rhythm contract (no gold/audio)."""
import argparse
import concurrent.futures
import contextlib
import datetime
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
sys.dont_write_bytecode = True
from common import DATA, confined, dump, load, sha, validate
from coordination import terminate
from hybrid import HERE, HYBRID, VERSION, instructions, merge, unsealed
from run import parse_response

MODEL = 'gpt-6-luna'
MODELS = ('gpt-6-luna', 'gpt-6.1-sol')


def attached_prompt(meta, cv):
    # Only adapt image delivery: the musical instructions and schema are exact.
    prompt = instructions(meta, cv)
    start = prompt.index('LOOK MODE:')
    end = prompt.index('\nThe clean original', start)
    return (prompt[:start] + 'LOOK MODE: The two attached images are original.png first and annotated.png second. '
            'Inspect each attached image exactly once. No image-view, terminal, file-reading, search, crop, code, '
            'agent, browser, write or other tools. The Python harness saves your final response. Return JSON only.' + prompt[end:])


def rate_limited(text):
    # A bare 429 also appears in coordinates (x=429.25); require an HTTP/status context for it.
    return bool(re.search(r'(?:status|http|error|code)\W{0,3}429\b|\b429\W{0,3}too many|rate[_ -]?limit(?:ed|_exceeded| reached)?|too many requests|usage limit|quota exceeded', text, re.I))


def retry_seconds(text, fallback):
    matches = re.findall(r'(?:retry[-_ ]after[\"\s:=]+|try again in\s+)(\d+(?:\.\d+)?)\s*(ms|milliseconds|s|seconds|m|minutes)?', text, re.I)
    delays = [float(n) * (.001 if unit.lower() in ('ms', 'milliseconds') else 60 if unit.lower() in ('m', 'minutes') else 1)
              for n, unit in matches]
    return max([fallback, *delays])


class Backoff:
    """All workers honor any worker's cooldown; no quota hammering."""
    def __init__(self):
        self.lock = threading.Lock()
        self.until = 0

    def defer(self, seconds):
        with self.lock:
            self.until = max(self.until, time.monotonic() + seconds)

    def wait(self, stop):
        while not stop.is_set():
            with self.lock:
                remaining = self.until - time.monotonic()
            if remaining <= 0:
                return True
            stop.wait(min(15, remaining))
        return False


@contextlib.contextmanager
def run_lock(owner, name='gpt'):
    with confined(DATA/(name+'.lock')).open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another hybrid GPT batch is active')
        dump(DATA/(name+'-owner.json'), dict(pid=os.getpid(), owner=owner, started_unix=time.time()))
        try:
            yield
        finally:
            dump(DATA/(name+'-owner.json'), dict(pid=os.getpid(), owner=owner, finished_unix=time.time()))


def argv_for(args, output, prompt):
    return [args.codex, 'exec', '-m', args.model, '-c', 'model_reasoning_effort='+args.effort,
            '--skip-git-repo-check', '-s', 'read-only', '-i', 'original.png', '-i', 'annotated.png',
            '-o', str(output), '--ignore-user-config', '--ephemeral', '--json',
            '-c', 'project_doc_max_bytes=0', prompt]


def invoke(argv, job, out_path, err_path, seconds, stop):
    if stop.is_set():
        return 'cancelled', None
    started = time.monotonic()
    with out_path.open('w') as out, err_path.open('w') as err:
        proc = subprocess.Popen(argv, cwd=job, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                start_new_session=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        try:
            while True:
                logs = out_path.read_text(errors='replace') + err_path.read_text(errors='replace')
                code = proc.poll()
                if code is not None:
                    return ('ok' if code == 0 else 'rate_limited' if rate_limited(logs) else 'cli_error'), code
                if rate_limited(logs):
                    terminate(proc)
                    return 'rate_limited', proc.returncode
                if stop.is_set():
                    terminate(proc)
                    return 'cancelled', proc.returncode
                if time.monotonic()-started >= seconds:
                    terminate(proc)
                    return 'timeout', proc.returncode
                stop.wait(.25)
        finally:
            terminate(proc)


def one(sid, args, folder, stop, backoff):
    source, job = confined(HYBRID/'jobs'/sid), confined(folder/sid)
    job.mkdir(parents=True, exist_ok=True)
    identity = dict(model=args.model, effort=args.effort, mode='look',
                    inputs={f: sha(source/f) for f in ('job.json', 'cv.json', 'original.png', 'annotated.png', 'legend.json')},
                    runner_sha256=sha(__file__), generator_sha256=sha(HERE/'hybrid.py'),
                    validator_sha256=sha(HERE/'common.py'), parser_sha256=sha(HERE/'run.py'))
    if args.resume and (job/'result.json').exists():
        prior = load(job/'result.json')
        if prior.get('identity') == identity and prior.get('status') == 'ok':
            if prior['reading_sha256'] != sha(job/'reading.json'):
                raise ValueError('successful reading was modified: '+sid)
            validate(load(job/'reading.json'), sid)
            return prior
        # Match Gemini: a completed visual response is not automatically reread.
        if prior.get('identity') == identity and prior.get('status') == 'invalid_json':
            return prior
    if (job/'reading.json').exists():
        raise ValueError('existing reading may not be overwritten: '+sid)
    meta, cv = load(source/'job.json'), load(source/'cv.json')
    unsealed(meta['video'])
    if meta.get('cv_sha256') is not None and meta['cv_sha256'] != sha(source/'cv.json'):
        raise ValueError('CV metadata digest mismatch')
    for image in meta['images']:
        if sha(source/image['file']) != image['sha256']:
            raise ValueError('prepared image digest mismatch')
    if cv.get('image_sha256') is not None and sha(source/'original.png') != cv['image_sha256']:
        raise ValueError('CV and original image differ')
    for f in ('original.png', 'annotated.png', 'legend.json', 'cv.json', 'annotation.json'):
        shutil.copy2(source/f, job/f)
    meta['job_dir'] = str(job)
    dump(job/'job.json', meta)
    prompt = attached_prompt(meta, cv)
    (job/'INSTRUCTIONS.md').write_text(prompt)
    result = dict(id=sid, identity=identity, status='cancelled', attempts=[], cv_seconds=cv['wall_seconds'],
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    started = time.monotonic()
    base_prompt, schema_left = prompt, getattr(args, 'schema_retries', 0)
    for retry in range(args.rate_retries+1+schema_left):
        if not backoff.wait(stop):
            break
        attempt = len(list(job.glob('stdout-*.log')))+1
        (job/f'prompt-{attempt}.txt').write_text(prompt)
        output = job/f'final-{attempt}.txt'
        tick = time.monotonic()
        state, code = invoke(argv_for(args, output, prompt), job, job/f'stdout-{attempt}.log',
                             job/f'stderr-{attempt}.log', args.timeout, stop)
        record = dict(attempt=attempt, seconds=time.monotonic()-tick, returncode=code, status=state)
        result['attempts'].append(record)
        result['status'] = state
        if state == 'rate_limited':
            logs = (job/f'stdout-{attempt}.log').read_text()+(job/f'stderr-{attempt}.log').read_text()
            delay = retry_seconds(logs, min(300, args.backoff_seconds*2**retry))
            record['backoff_seconds'] = delay
            backoff.defer(delay)
            print(json.dumps(dict(id=sid, status=state, retry=retry, backoff_seconds=delay)), flush=True)
            continue
        if state == 'ok':
            try:
                answer = parse_response(output.read_text())
                reading = merge(answer, cv, sid)
                # Preserve schema and frozen merge behavior, but label provenance honestly.
                for bar in reading['bars']:
                    for event in bar['events']:
                        for head in event['heads']:
                            head['rhythm_source'] = 'gpt-numbered'
                            if head['source'] == 'gemini-only-miss':
                                head['source'] = 'gpt-only-miss'
                reading['hybrid']['instrument_policy'] = 'CV only; all GPT-only additions unverified-instrument'
                validate(reading, sid)
                dump(job/'rhythm.json', answer)
                dump(job/'reading.json', reading)
            except (ValueError, KeyError, TypeError, ZeroDivisionError, OSError) as error:
                result['status'] = 'invalid_json'
                record['error'] = str(error)
                if schema_left > 0:
                    # Production reads may ask once more, quoting the validator; gold runs keep the default 0.
                    schema_left -= 1
                    prompt = base_prompt+'\n\nYour previous answer was rejected by the validator: '+str(error)+'. Answer again, following the JSON schema and allowed values exactly.'
                    continue
        break
    result['wall_seconds'] = time.monotonic()-started+cv['wall_seconds']+meta.get('annotation_seconds', 0)
    result['reading_sha256'] = sha(job/'reading.json') if result['status'] == 'ok' else None
    calls = []
    for log in job.glob('stdout-*.log'):
        for line in log.read_text(errors='replace').splitlines():
            try:
                item = json.loads(line).get('item', {})
                if item.get('type') in ('command_execution', 'mcp_tool_call', 'web_search', 'image_view', 'collab_tool_call'):
                    calls.append(item.get('type'))
            except (ValueError, AttributeError):
                pass
    result['tool_audit'] = dict(reported_tool_events=calls, trace_complete=False,
                                policy='two attached native images; no tools; JSONL audit is not an exhaustive certification')
    dump(job/'result.json', result)
    print(json.dumps(dict(id=sid, status=result['status'], seconds=round(result['wall_seconds'], 2))), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', default='hybrid-gpt-luna-xhigh')
    p.add_argument('--model', choices=MODELS, default=MODEL)
    p.add_argument('--systems', nargs='+')
    p.add_argument('--selection', choices=['gold', 'dense', 'selected'], default='gold')
    p.add_argument('--effort', choices=['medium', 'high', 'xhigh', 'max'], default='xhigh')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--timeout', type=float, default=900)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--codex', default='codex')
    p.add_argument('--rate-retries', type=int, default=6)
    p.add_argument('--backoff-seconds', type=float, default=30)
    p.add_argument('--schema-retries', type=int, default=0, help='rereads after a validator rejection (production only)')
    p.add_argument('--lock', default='gpt', help='lock name; use a distinct one to run batches in parallel')
    args = p.parse_args()
    if args.workers < 1 or args.timeout <= 0 or args.rate_retries < 0 or args.backoff_seconds <= 0:
        p.error('workers/timeout/backoff must be positive; retries nonnegative')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.run_name):
        p.error('unsafe run name')
    systems = args.systems or load(DATA/'selection.json')[args.selection]
    if len(systems) != len(set(systems)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in systems):
        p.error('duplicate or unsafe systems')
    folder = confined(DATA/'runs'/args.run_name)
    folder.mkdir(parents=True, exist_ok=True)
    config = dict(version=VERSION, backend='codex', model=args.model, effort=args.effort, mode='look',
                  workers=args.workers, timeout=args.timeout, rate_retries=args.rate_retries, schema_retries=args.schema_retries,
                  backoff_seconds=args.backoff_seconds, systems=systems)
    if (folder/'config.json').exists() and load(folder/'config.json') != config:
        p.error('config differs; use a new run name')
    stop, backoff, rows = threading.Event(), Backoff(), []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.lock):
        p.error('unsafe lock name')
    with run_lock(args.run_name, args.lock):
        dump(folder/'config.json', config)
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one, sid, args, folder, stop, backoff): sid for sid in systems}
            for future in concurrent.futures.as_completed(futures):
                try:
                    rows.append(future.result())
                except Exception as error:
                    rows.append(dict(id=futures[future], status='harness_error', error=str(error), wall_seconds=0))
                dump(folder/'run.json', {**config, 'batch_wall_seconds': time.monotonic()-started,
                     'completed': len(rows), 'cancelled': stop.is_set(), 'quota_stopped': False,
                     'results': sorted(rows, key=lambda r: r['id'])})
    print(json.dumps(dict(run=args.run_name, ok=sum(r['status']=='ok' for r in rows), total=len(rows))), flush=True)
    return int(any(r['status'] != 'ok' for r in rows))


if __name__ == '__main__':
    raise SystemExit(main())
