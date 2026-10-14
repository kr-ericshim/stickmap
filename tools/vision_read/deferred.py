"""Resume the authorized pilot once after the CLI-reported quota reset.

No account/subscription changes. No old models. Dense readings are prioritized.
Status and logs are local artifacts; this script cannot send a chat notification.
"""
import argparse
import datetime
import os
import re
import subprocess
import sys
import time
sys.dont_write_bytecode = True
from pathlib import Path
from common import DATA, ROOT, confined, dump, load, sha

HERE = Path(__file__).resolve().parent


def reset_at(run):
    deadlines = []
    for path in (DATA/'runs'/run).glob('*/stderr-*.log'):
        text = path.read_text(errors='replace')
        if 'RESOURCE_EXHAUSTED' not in text:
            continue
        match = re.search(r'Resets in\s+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?', text)
        if match and any(match.groups()):
            hours, minutes, seconds = [int(value or 0) for value in match.groups()]
            deadlines.append(path.stat().st_mtime+hours*3600+minutes*60+seconds+90)
    if not deadlines:
        raise ValueError('no CLI-reported quota reset duration found')
    return max(time.time()+1, max(deadlines))


def utc(stamp):
    return datetime.datetime.fromtimestamp(stamp, datetime.timezone.utc).isoformat()


def quota_in(run):
    return any('RESOURCE_EXHAUSTED' in p.read_text(errors='replace') for p in (DATA/'runs'/run).glob('*/stderr-*.log'))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quota-run', default='dense-medium-probe')
    ap.add_argument('--plan-only', action='store_true')
    ap.add_argument('--state', type=Path)
    ap.add_argument('--hybrid-run', help='prioritize prepared hybrid gold before the existing pilot')
    ap.add_argument('--hybrid-only', action='store_true', help='defer only the requested hybrid comparison')
    ap.add_argument('--max-quota-retries', type=int, default=1)
    a = ap.parse_args()
    target = reset_at(a.quota_run)
    statefile = confined(a.state or DATA/('deferred-preview.json' if a.plan_only else 'deferred.json'))
    if not a.plan_only and statefile.exists():
        prior = load(statefile)
        if prior.get('status') in ('waiting', 'running'):
            try:
                os.kill(prior['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                ap.error('a deferred pilot is already active')
    if a.hybrid_only and not a.hybrid_run:
        ap.error('--hybrid-only requires --hybrid-run')
    if a.max_quota_retries < 0:
        ap.error('max-quota-retries must be nonnegative')
    stages = [] if a.hybrid_only else [('dense-high-look-resumed-4w', 'dense', 'look'), ('high-zoom-budgeted-4w', 'gold', 'zoom')]
    selections = load(DATA/'selection.json')
    commands = [[sys.executable, '-B', str(HERE/'run.py'), '--selection', selection, '--model', 'gemini-3.8-flash-high',
                 '--mode', mode, '--workers', '4', '--timeout', '900', '--run-name', name, '--resume', '--systems', *selections[selection]] for name, selection, mode in stages]
    inputs = {str(DATA/'jobs'/sid/file): sha(DATA/'jobs'/sid/file)
              for sid in selections['gold']+selections['dense'] for file in ('job.json', 'legend.json')}
    if a.hybrid_run:
        from hybrid import HYBRID
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', a.hybrid_run):
            ap.error('unsafe hybrid-run')
        stages.insert(0, (a.hybrid_run, 'gold', 'hybrid'))
        gold = load(DATA/'bench/gold.json')['systems']
        prioritized = sorted(selections['gold'], key=lambda sid: not gold[sid].get('complete', False))
        commands.insert(0, [sys.executable, '-B', str(HERE/'hybrid_run.py'), '--selection', 'gold', '--workers', '4', '--timeout', '900', '--run-name', a.hybrid_run, '--resume', '--systems', *prioritized])
        inputs.update({str(HYBRID/'jobs'/sid/file): sha(HYBRID/'jobs'/sid/file)
                       for sid in selections['gold'] for file in ('job.json', 'cv.json', 'original.png', 'annotated.png', 'legend.json')})
    inputs.update({str(HERE/file): sha(HERE/file) for file in ('run.py', 'hybrid.py', 'hybrid_run.py', 'coordination.py', 'common.py', 'score.py', 'hybrid_compare.py', 'HYBRID_WSL_PLAN.md')})
    state = {'status': 'planned' if a.plan_only else 'waiting', 'pid': os.getpid(), 'scheduled_utc': utc(time.time()), 'not_before_utc': utc(target),
             'quota_evidence_run': a.quota_run, 'commands': commands, 'prepared_input_sha256': inputs, 'stages': [],
             'stage_definitions': stages, 'quota_retries': 0,
             'policy': 'shared agy.lock; at most four workers globally; first 429 cancels pending/in-flight jobs; retry after CLI reset once; no audio/vault scoring'}
    dump(statefile, state)
    print(state['status'], state['not_before_utc'], flush=True)
    if a.plan_only:
        return
    try:
        while time.time() < target:
            state['heartbeat_utc'] = utc(time.time())
            dump(statefile, state)
            time.sleep(min(30, max(.1, target-time.time())))
        if any(sha(path) != digest for path, digest in inputs.items()):
            raise ValueError('prepared jobs/legends changed while waiting; frozen pilot must be re-planned')
        for (name, selection, mode), command in zip(stages, commands):
            state.update(status='running', current_run=name, heartbeat_utc=utc(time.time()))
            dump(statefile, state)
            while True:
                with confined(DATA/f'deferred-{name}.log').open('a') as log:
                    subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
                batch = load(DATA/'runs'/name/'run.json')
                if mode == 'hybrid':
                    with confined(DATA/'deferred-hybrid-comparison.log').open('a') as log:
                        subprocess.run([sys.executable, '-B', str(HERE/'hybrid_compare.py'), '--run', name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                if not batch.get('quota_stopped') and not any(r['status'] == 'quota_exhausted' for r in batch['results']):
                    break
                if state['quota_retries'] >= a.max_quota_retries:
                    break
                target = reset_at(name)
                state.update(status='waiting', current_run=name, not_before_utc=utc(target), quota_evidence_run=name,
                             quota_retries=state['quota_retries']+1, heartbeat_utc=utc(time.time()))
                dump(statefile, state)
                while time.time() < target:
                    time.sleep(min(30, max(.1, target-time.time())))
                    state['heartbeat_utc'] = utc(time.time()); dump(statefile, state)
                if any(sha(path) != digest for path, digest in inputs.items()):
                    raise ValueError('prepared inputs changed during quota retry')
                state.update(status='running', heartbeat_utc=utc(time.time())); dump(statefile, state)
            batch = load(DATA/'runs'/name/'run.json')
            state['stages'].append({'run': name, 'statuses': [r['status'] for r in batch['results']], 'batch_wall_seconds': batch['batch_wall_seconds']})
            dump(statefile, state)
            if batch.get('quota_stopped') or any(r['status'] == 'quota_exhausted' for r in batch['results']):
                state.update(status='quota_blocked', reason='quota returned after the scheduled reset; no further automatic inference attempts', heartbeat_utc=utc(time.time()))
                dump(statefile, state)
                break
        else:
            state.update(status='inference_finished', heartbeat_utc=utc(time.time()))
            dump(statefile, state)
        # Always score attempted stages. Do not silently turn quota failures into
        # model accuracy. The report detects/censors RESOURCE_EXHAUSTED jobs.
        attempted = [r['run'] for r in state['stages']]
        zoom = 'high-zoom-budgeted-4w' if 'high-zoom-budgeted-4w' in attempted else 'high-zoom'
        dense_run = 'dense-high-look-resumed-4w' if 'dense-high-look-resumed-4w' in attempted else 'dense-high-look'
        for name in dict.fromkeys(['high-look', 'medium-look', zoom, dense_run]+attempted):
            with confined(DATA/f'deferred-score-{name}.log').open('w') as log:
                subprocess.run([sys.executable, '-B', str(HERE/'score.py'), '--run', name], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        if a.hybrid_run:
            with confined(DATA/'deferred-hybrid-comparison.log').open('w') as log:
                subprocess.run([sys.executable, '-B', str(HERE/'hybrid_compare.py'), '--run', a.hybrid_run], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        with confined(DATA/'deferred-report.log').open('w') as log:
            subprocess.run([sys.executable, '-B', str(HERE/'report.py'), '--runs', 'high-look', 'medium-look', zoom,
                            '--dense-run', dense_run], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        densepath = DATA/'runs/dense-high-look-resumed-4w/run.json'
        if not a.hybrid_only and densepath.exists():
            dense = load(densepath)
            targetfolder = DATA/'wsl_handoff/resumed-4w'
            if any(r['status'] == 'ok' for r in dense['results']) and not (targetfolder/'anchors.template.json').exists():
                subprocess.run([sys.executable, '-B', str(HERE/'export_timing.py'), '--run', 'dense-high-look-resumed-4w',
                                '--write-anchor-template', '--out', str(targetfolder)], cwd=ROOT, check=True)
        if state['status'] != 'quota_blocked':
            all_ok = all(all(status == 'ok' for status in row['statuses']) for row in state['stages'])
            state['status'] = 'complete' if all_ok else 'finished_with_failures'
        state.update(heartbeat_utc=utc(time.time()), report=str(DATA/'PILOT_REPORT.md'))
        dump(statefile, state)
        print(state['status'], flush=True)
    except BaseException as error:
        state.update(status='failed', reason=str(error), heartbeat_utc=utc(time.time()))
        dump(statefile, state)
        raise


if __name__ == '__main__':
    main()
