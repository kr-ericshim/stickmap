"""Run independent agy image jobs, retain raw responses and retry invalid JSON once."""
import argparse
import concurrent.futures
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import threading
sys.dont_write_bytecode = True
from common import DATA, ROOT, VERSION, confined, dump, load, sha, validate
from prepare import instructions
from coordination import batch_lock, invoke


def parse_response(text):
    # CLI can print tool/status chatter before the final response. Persist that
    # entire response; accept exactly one contract-bearing JSON object.
    decoder = json.JSONDecoder()
    matches = []
    for match in re.finditer(r'\{', text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict) and 'schema_version' in value and 'system_id' in value:
                matches.append(value)
        except ValueError:
            pass
    if len(matches) != 1:
        raise ValueError(f'expected one reading JSON object, found {len(matches)}')
    return matches[0]


def one(sid, a, run_dir):
    source = confined(DATA/'jobs'/sid)
    job = confined(run_dir/sid)
    job.mkdir(parents=True, exist_ok=True)
    identity = {'model': a.model, 'mode': a.mode, 'job_sha256': sha(source/'job.json'), 'legend_sha256': sha(source/'legend.json'), 'runner_sha256': sha(__file__), 'prompt_generator_sha256': sha(PathPrepare())}
    if a.resume and (job/'result.json').exists():
        prior = load(job/'result.json')
        if prior.get('identity') == identity and prior.get('status') == 'ok':
            validate(load(job/'reading.json'), sid)
            return prior
    if (job/'reading.json').exists():
        raise ValueError(f'existing reading would be overwritten: {job}; use a fresh --run-name or matching --resume')
    meta = load(source/'job.json')
    from hybrid import unsealed
    unsealed(meta['video'])
    meta['job_dir'] = str(job)
    for image in meta['images']:
        p = source/image['file']
        if sha(p) != image['sha256']:
            raise ValueError(f'input hash mismatch: {p}')
        shutil.copy2(p, job/image['file'])
    shutil.copy2(source/'legend.json', job/'legend.json')
    dump(job/'job.json', meta)
    prompt = instructions(meta, load(job/'legend.json'), a.mode)
    (job/'INSTRUCTIONS.md').write_text(prompt)
    attempts = []
    started = time.monotonic()
    result = {'id': sid, 'identity': identity, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'invalid_json', 'attempts': attempts}
    for attempt in range(2):
        if a.stop.is_set():
            result['status'] = 'quota_cancelled'
            break
        remaining = a.timeout-(time.monotonic()-started)
        if remaining <= 0:
            result['status'] = 'timeout'
            break
        extra = '' if attempt == 0 else '\nYour preceding response was invalid: '+attempts[-1]['error']+'. Re-read independently and return the complete corrected strict JSON contract.\n'
        full_prompt = prompt+extra
        (job/f'prompt-{attempt+1}.txt').write_text(full_prompt)
        # -p= after --model is intentional: agy flag order is significant.
        argv = [a.agy, '--model', a.model, '--dangerously-skip-permissions', '--output-format', 'text', '--log-file', str(job/f'cli-{attempt+1}.log'), '-p='+full_prompt]
        tick = time.monotonic()
        state, returncode = invoke(argv, job, job/f'stdout-{attempt+1}.log', job/f'stderr-{attempt+1}.log', remaining, a.stop)
        record = {'attempt': attempt+1, 'seconds': time.monotonic()-tick, 'returncode': returncode}
        attempts.append(record)
        if state != 'ok':
            record['error'] = state+'; see retained logs'
            result['status'] = state
            break
        try:
            reading = validate(parse_response((job/f'stdout-{attempt+1}.log').read_text()), sid)
            dump(job/'reading.json', reading)
            result['status'] = 'ok'
            break
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
            record['error'] = str(error)
    result['wall_seconds'] = time.monotonic()-started
    result['reading_sha256'] = sha(job/'reading.json') if result['status'] == 'ok' else None
    # Logs permit an observational audit; prompt restrictions are not a sandbox.
    logs = '\n'.join(p.read_text(errors='replace') for p in job.glob('cli-*.log'))
    result['tool_audit'] = {'terminal_regex_hits': len(re.findall(r'Invoking tool run_command', logs)), 'viewer_regex_hits': len(re.findall(r'Invoking tool (?:view_file|view_image)', logs)), 'trace_complete': False, 'look_policy_enforced': False}
    dump(job/'result.json', result)
    print(json.dumps({'id': sid, 'status': result['status'], 'seconds': round(result['wall_seconds'], 1), 'attempts': len(attempts)}), flush=True)
    return result


def PathPrepare():
    from pathlib import Path
    return Path(__file__).with_name('prepare.py')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selection', choices=['gold', 'dense', 'selected'], default='gold')
    ap.add_argument('--systems', nargs='+')
    ap.add_argument('--model', default='gemini-3.8-flash-high')
    ap.add_argument('--mode', choices=['look', 'zoom'], default='look')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=900, help='total seconds per system including its one JSON retry')
    ap.add_argument('--run-name')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--agy', default='agy')
    a = ap.parse_args()
    if a.model.startswith('gemini-3.1'):
        ap.error('gemini-3.1-* models are forbidden')
    if not 1 <= a.workers <= 4 or a.timeout <= 0:
        ap.error('workers must be 1..4 and timeout positive')
    name = a.run_name or f'{a.model}-{a.mode}'
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
        ap.error('run-name must be a single safe directory name')
    run_dir = confined(DATA/'runs'/name)
    run_dir.mkdir(parents=True, exist_ok=True)
    selected = a.systems or load(DATA/'selection.json')[a.selection]
    if len(selected) != len(set(selected)):
        ap.error('duplicate systems')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', sid) for sid in selected):
        ap.error('unsafe system id')
    config = {'version': VERSION, 'model': a.model, 'mode': a.mode, 'workers': a.workers, 'timeout': a.timeout, 'systems': selected}
    if (run_dir/'config.json').exists() and load(run_dir/'config.json') != config:
        ap.error('existing run config differs; choose a fresh run-name')
    dump(run_dir/'config.json', config)
    start = time.monotonic()
    rows = []
    a.stop = threading.Event()
    with batch_lock(name), concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(one, sid, a, run_dir): sid for sid in selected}
        for future in concurrent.futures.as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                row = {'id': futures[future], 'status': 'harness_error', 'error': str(error), 'wall_seconds': 0}
                rows.append(row)
                print(json.dumps(row), flush=True)
            dump(run_dir/'run.json', {**config, 'batch_wall_seconds': time.monotonic()-start, 'completed': len(rows), 'quota_stopped': a.stop.is_set(), 'results': sorted(rows, key=lambda r: r['id'])})
    print(json.dumps({'run': name, 'batch_seconds': round(time.monotonic()-start, 1), 'ok': sum(r['status'] == 'ok' for r in rows), 'total': len(rows)}), flush=True)


if __name__ == '__main__':
    main()
