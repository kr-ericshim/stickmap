"""Run numbered-head rhythm jobs with shared locking and fail-fast quota handling."""
import argparse
import concurrent.futures
import datetime
import json
import re
import shutil
import sys
import threading
import time
sys.dont_write_bytecode = True
from common import DATA, confined, dump, load, sha, validate
from coordination import batch_lock, invoke
from hybrid import HERE, HYBRID, MODEL, VERSION, instructions, merge, unsealed
from run import parse_response


def one(sid, args, folder, stop):
    source = confined(HYBRID/'jobs'/sid)
    job = confined(folder/sid)
    job.mkdir(parents=True, exist_ok=True)
    identity = {'model': MODEL, 'mode': 'look', 'inputs': {f: sha(source/f) for f in ('job.json', 'cv.json', 'original.png', 'annotated.png', 'legend.json')},
                'runner_sha256': sha(__file__), 'generator_sha256': sha(HERE/'hybrid.py'), 'coordination_sha256': sha(HERE/'coordination.py')}
    if args.resume and (job/'result.json').exists():
        prior = load(job/'result.json')
        if prior.get('identity') == identity and prior.get('status') == 'ok':
            if prior['reading_sha256'] != sha(job/'reading.json'):
                raise ValueError('successful reading was modified: '+sid)
            validate(load(job/'reading.json'), sid)
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
    prompt = instructions(meta, cv)
    (job/'INSTRUCTIONS.md').write_text(prompt)
    result = {'id': sid, 'identity': identity, 'status': 'quota_cancelled' if stop.is_set() else 'invalid_json', 'attempts': [],
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'cv_seconds': cv['wall_seconds']}
    started = time.monotonic()
    # One view pass per system. A malformed final response is retained for
    # representation-only recovery; never re-open the images in an auto retry.
    if not stop.is_set():
        attempt = len(list(job.glob('stdout-*.log')))+1
        (job/f'prompt-{attempt}.txt').write_text(prompt)
        argv = [args.agy, '--model', MODEL, '--dangerously-skip-permissions', '--disable-slash-commands',
                '--output-format', 'text', '--log-file', str(job/f'cli-{attempt}.log'), '-p='+prompt]
        state, code = invoke(argv, job, job/f'stdout-{attempt}.log', job/f'stderr-{attempt}.log', args.timeout, stop)
        record = {'attempt': attempt, 'seconds': time.monotonic()-started, 'returncode': code}
        result['attempts'].append(record)
        result['status'] = state
        if state == 'ok':
            try:
                answer = parse_response((job/f'stdout-{attempt}.log').read_text())
                reading = merge(answer, cv, sid)
                dump(job/'rhythm.json', answer)
                dump(job/'reading.json', reading)
            except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
                result['status'] = 'invalid_json'
                record['error'] = str(error)
    result['wall_seconds'] = time.monotonic()-started+cv['wall_seconds']+meta.get('annotation_seconds', 0)
    result['reading_sha256'] = sha(job/'reading.json') if result['status'] == 'ok' else None
    logs = '\n'.join(p.read_text(errors='replace') for p in job.glob('cli-*.log'))
    result['tool_audit'] = {'terminal_regex_hits': len(re.findall(r'Invoking tool run_command', logs)),
                            'viewer_regex_hits': len(re.findall(r'Invoking tool (?:view_file|view_image)', logs)),
                            'trace_complete': False, 'look_policy_enforced': False,
                            'policy': 'original once + annotation once; no other tools; text trace cannot certify compliance'}
    dump(job/'result.json', result)
    print(json.dumps({'id': sid, 'status': result['status'], 'seconds': round(result['wall_seconds'], 2)}), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-name', default='hybrid-medium-look')
    ap.add_argument('--systems', nargs='+')
    ap.add_argument('--selection', choices=['gold', 'dense', 'selected'], default='gold')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--timeout', type=float, default=900)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--agy', default='agy')
    args = ap.parse_args()
    if not 1 <= args.workers <= 4 or args.timeout <= 0:
        ap.error('workers must be 1..4 and timeout positive')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', args.run_name):
        ap.error('unsafe run name')
    systems = args.systems or load(DATA/'selection.json')[args.selection]
    if len(systems) != len(set(systems)):
        ap.error('duplicate systems')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', sid) for sid in systems):
        ap.error('unsafe system id')
    folder = confined(DATA/'runs'/args.run_name)
    folder.mkdir(parents=True, exist_ok=True)
    config = {'version': VERSION, 'model': MODEL, 'mode': 'look', 'workers': args.workers, 'timeout': args.timeout, 'systems': systems}
    if (folder/'config.json').exists() and load(folder/'config.json') != config:
        ap.error('config differs; use a new run name')
    rows, stop = [], threading.Event()
    with batch_lock(args.run_name):
        dump(folder/'config.json', config)
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(one, sid, args, folder, stop): sid for sid in systems}
            for future in concurrent.futures.as_completed(futures):
                try:
                    rows.append(future.result())
                except Exception as error:
                    rows.append({'id': futures[future], 'status': 'harness_error', 'error': str(error), 'wall_seconds': 0})
                dump(folder/'run.json', {**config, 'batch_wall_seconds': time.monotonic()-started, 'completed': len(rows),
                                        'quota_stopped': stop.is_set(), 'results': sorted(rows, key=lambda r: r['id'])})
    print(json.dumps({'run': args.run_name, 'ok': sum(r['status'] == 'ok' for r in rows), 'total': len(rows), 'quota_stopped': stop.is_set()}), flush=True)


if __name__ == '__main__':
    main()
