"""Take over dense scratch candidates only after the requested GPT gold pass."""
import argparse
import datetime
import os
import signal
import subprocess
import sys
import time
import shutil
sys.dont_write_bytecode = True
from common import DATA, ROOT, dump, load, sha, validate
from hybrid_gpt_compare import compare
from solos_extract import DENSE
from solos_finish import HOST, STATE, REMOTE, REMOTE_DATA, run, transfer_and_gate


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def stop_waiters(chosen):
    path = DATA/'hybrid-solos-dense-v1-state.json'
    state = load(path)
    if state.get('status') == 'superseded_by_gpt':
        if state['gpt_gold_run'] != chosen:
            raise ValueError('different GPT takeover already recorded')
        return
    if state.get('status') != 'waiting_for_quota_reset':
        raise ValueError('Gemini is not waiting; do not interrupt an active reading')
    outer = load(STATE)
    owners = [(outer.get('pid'), 'solos_finish.py'), (state.get('pid'), 'solos_run.py')]
    stopped = []
    # Stop the waiting parent first so its child exit cannot launch gates.
    for pid, expected in owners:
        command = subprocess.run(['ps', '-p', str(pid), '-o', 'command='], capture_output=True, text=True).stdout
        if not command.strip():
            continue
        if expected not in command or (expected == 'solos_run.py' and 'hybrid-solos-dense-v1' not in command):
            raise ValueError('coordinator PID identity differs: '+str(pid))
        children = subprocess.run(['pgrep', '-P', str(pid)], capture_output=True, text=True).stdout.split()
        if expected == 'solos_run.py' and children:
            raise ValueError('Gemini waiter has active children; do not interrupt')
        os.kill(pid, signal.SIGTERM)
        for _ in range(50):
            live = subprocess.run(['ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True).stdout.strip()
            if not live or live.startswith('Z'):
                break
            time.sleep(.1)
        else:
            raise RuntimeError('coordinator did not stop after SIGTERM')
        stopped.append(dict(pid=pid, command=command.strip(), signal='SIGTERM'))
    state.update(previous_status=state['status'], status='superseded_by_gpt', stopped_utc=utc(),
                 stopped_processes=stopped, gpt_gold_run=chosen, reason='complete gold joint >=95%; waiting coordinators stopped before GPT takeover')
    dump(path, state)


def successful(run_name, systems):
    results = {}
    for sid in systems:
        folder = DATA/'runs'/run_name/sid
        if not (folder/'result.json').exists():
            continue
        result = load(folder/'result.json')
        if result['status'] != 'ok':
            continue
        if result['reading_sha256'] != sha(folder/'reading.json'):
            raise ValueError('reading digest mismatch: '+sid)
        validate(load(folder/'reading.json'), sid)
        for name, digest in result['identity']['inputs'].items():
            if sha(DATA/'hybrid/jobs'/sid/name) != digest:
                raise ValueError('frozen input changed: '+sid+'/'+name)
        results[sid] = result
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold-runs', nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    comparison = compare(args.gold_runs)
    chosen = comparison['chosen_run']
    if not chosen:
        raise ValueError('no GPT configuration reaches the complete-gold criterion; keep Gemini')
    if args.workers < 1:
        parser.error('workers must be positive')
    config = comparison['configs'][chosen]
    original = load(DATA/'hybrid-solos-dense-v1-state.json')
    systems = original['systems']
    reused = successful('hybrid-solos-dense-v1', systems)
    pending = [sid for sid in systems if sid not in reused]
    suffix = 'luna' if config['model'] == 'gpt-6-luna' else 'sol'
    effort = config['effort']
    raw_run = 'hybrid-solos-dense-gpt-'+suffix+'-'+effort+'-v1'
    combined = raw_run+'-combined'
    stop_waiters(chosen)
    state = dict(status='dense_gpt', pid=os.getpid(), dense_run=combined, gpt_run=raw_run,
                 gold_run=chosen, model=config['model'], effort=effort, workers=args.workers,
                 reused_gemini_systems=list(reused), pending=len(pending), started_utc=utc(),
                 acceptance_limitation='Eric strict-empty control remains inconclusive; Larnell 97.04% passed')
    dump(STATE, state)
    if pending:
        command = [sys.executable, '-B', ROOT/'tools/vision_read/hybrid_gpt.py', '--run-name', raw_run,
                   '--model', config['model'], '--effort', effort, '--workers', str(args.workers), '--timeout', '1800', '--schema-retries', '1', '--resume', '--systems', *pending]
        with (DATA/(raw_run+'.log')).open('a') as log:
            # Nonzero means some reads failed; they are listed below and excluded, not filled in.
            subprocess.run(list(map(str, command)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    fresh = successful(raw_run, pending)
    if not fresh and pending:
        raise RuntimeError('no dense GPT readings succeeded; retain all raw results')
    failed = [sid for sid in pending if sid not in fresh]
    # Gate only systems that were actually read; failed reads are listed, never filled in.
    systems = [sid for sid in systems if sid in reused or sid in fresh]
    state.update(failed_reads=failed)
    folder = DATA/'runs'/combined
    folder.mkdir(parents=True, exist_ok=True)
    origins = {}
    results = []
    for sid in systems:
        origin = 'hybrid-solos-dense-v1' if sid in reused else raw_run
        source = DATA/'runs'/origin/sid
        target = folder/sid
        if target.exists():
            if sha(target/'reading.json') != sha(source/'reading.json'):
                raise ValueError('existing combined reading differs: '+sid)
        else:
            shutil.copytree(source, target)
        origins[sid] = origin
        results.append((reused if sid in reused else fresh)[sid])
    batch = load(DATA/'runs'/raw_run/'run.json') if pending else dict(batch_wall_seconds=0)
    combined_config = dict(version='hybrid-gpt-solos-v1', backend='codex + retained Gemini successes',
                           model=config['model'], effort=effort, workers=args.workers, systems=systems, failed_reads=failed,
                           origins=origins, gold_run=chosen)
    dump(folder/'config.json', combined_config)
    dump(folder/'run.json', {**combined_config, 'batch_wall_seconds': batch['batch_wall_seconds'],
                            'completed': len(results), 'quota_stopped': False, 'results': results})
    state.update(status='dense_audio_gates', systems_read=len(results), heartbeat_utc=utc()); dump(STATE, state)
    run(['rsync', '-a', str(ROOT/'tools/vision_read')+'/', HOST+':'+REMOTE+'/tools/vision_read/'])
    transfer_and_gate(combined, DENSE)
    report = DATA/'HYBRID_SOLOS_REPORT.md'
    text = report.read_text().replace('# Hybrid CV + Gemini dense solos', '# Hybrid CV + vision dense solos', 1)
    text = text.replace('Gemini 모델은 `gemini-3.8-flash-medium`, look, 최대 4 workers다.',
                        f"Dense는 `{config['model']}` {effort}, look, {args.workers} workers로 읽었다. 기존 Gemini 성공 {len(reused)}개를 그대로 재사용했고 GPT 신규 {len(pending)}개를 읽었다. Controls는 기존 Gemini 결과다.", 1)
    text += '\n## GPT takeover\n\n'+f'Complete gold 기준을 통과한 `{chosen}`를 사용했다. 같은 WSL `auto_timing.stage`, `solos_gate.py`, `solos_verify.py`와 scratch 출력 경로를 사용했다. Gemini 대기 coordinator 종료 기록은 `hybrid-solos-dense-v1-state.json`에 있다. 혼합 run의 config.origins가 각 raw reading 출처를 기록한다.\n\n'
    text += (DATA/'eric-rejection-diagnosis.md').read_text()
    report.write_text(text)
    run(['rsync', '-a', report, HOST+':'+REMOTE_DATA+'/REPORT.md'])
    for name in ('eric-rejection-diagnosis.json', 'eric-rejection-diagnosis.md'):
        run(['rsync', '-a', DATA/name, HOST+':'+REMOTE_DATA+'/'+name])
    state.update(status='complete', finished_utc=utc()); dump(STATE, state)
    print('COMPLETE: dense readings, unchanged WSL gates and report', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # A preflight rejection must not change the still-authorized Gemini state.
        if STATE.exists():
            state = load(STATE)
            if state.get('pid') == os.getpid():
                state.update(status='gpt_error', error=type(error).__name__+': '+str(error), heartbeat_utc=utc())
                dump(STATE, state)
        raise
