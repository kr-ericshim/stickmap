"""Resume a frozen hybrid batch, waiting for the shared lock and quota reset."""
import argparse
import datetime
import fcntl
import os
import subprocess
import sys
import time
from common import DATA, ROOT, dump, load
from deferred import reset_at


def utc(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).isoformat()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',required=True)
    p.add_argument('--videos',nargs='+',required=True)
    a=p.parse_args()
    systems=[]
    for video in a.videos:
        systems.extend(sorted(q.name for q in (DATA/'hybrid/jobs').glob(video+'_auto_*') if (q/'job.json').exists()))
    if not systems:
        raise ValueError('no prepared jobs')
    state_path=DATA/(a.run_name+'-state.json')
    # A process restart must not probe the account before a known reset.
    previous=load(state_path) if state_path.exists() else {}
    owner=(DATA/(a.run_name+'.runner.lock')).open('a+')
    try:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('this run already has a coordinator')
    state=dict(run=a.run_name,systems=systems,status='waiting_for_lock',model='gemini-3.8-flash-medium',workers=4,
               attempts=previous.get('attempts',0),pid=os.getpid(),ok=previous.get('ok',0))
    prior_run=DATA/'runs'/a.run_name/'run.json'
    if prior_run.exists() and load(prior_run).get('quota_stopped'):
        deadline=reset_at(a.run_name)
        if previous.get('not_before_utc'):
            deadline=max(deadline,datetime.datetime.fromisoformat(previous['not_before_utc']).timestamp())
        state.update(status='waiting_for_quota_reset',not_before_utc=utc(deadline));dump(state_path,state)
        while time.time()<deadline:
            state['heartbeat_utc']=utc(time.time());dump(state_path,state)
            time.sleep(min(15,max(.1,deadline-time.time())))
    while True:
        with (DATA/'agy.lock').open('a+') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                fcntl.flock(lock,fcntl.LOCK_UN)
                available=True
            except BlockingIOError:
                available=False
        if not available:
            state.update(status='waiting_for_lock',heartbeat_utc=utc(time.time()));dump(state_path,state)
            time.sleep(15);continue
        state.update(status='running',attempts=state['attempts']+1,heartbeat_utc=utc(time.time()));dump(state_path,state)
        cmd=[sys.executable,'-B',str(ROOT/'tools/vision_read/hybrid_run.py'),'--run-name',a.run_name,
             '--workers','4','--timeout','900','--resume','--systems',*systems]
        result=subprocess.run(cmd,cwd=ROOT,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
        run_path=DATA/'runs'/a.run_name/'run.json'
        if result.returncode or not run_path.exists():
            state.update(status='harness_error',returncode=result.returncode);dump(state_path,state)
            return 1
        batch=load(run_path)
        state['ok']=sum(r['status']=='ok' for r in batch['results'])
        if not batch.get('quota_stopped'):
            state.update(status='complete' if state['ok']==len(systems) else 'finished_with_failures',finished_utc=utc(time.time()))
            dump(state_path,state);return 0
        try:
            deadline=reset_at(a.run_name)
        except ValueError as error:
            state.update(status='quota_wait_unknown_reset',reason=str(error));dump(state_path,state)
            return 2
        state.update(status='waiting_for_quota_reset',not_before_utc=utc(deadline));dump(state_path,state)
        while time.time()<deadline:
            state['heartbeat_utc']=utc(time.time());dump(state_path,state)
            time.sleep(min(15,max(.1,deadline-time.time())))


if __name__=='__main__':
    raise SystemExit(main())
