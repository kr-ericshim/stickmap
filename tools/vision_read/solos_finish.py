"""Mac coordinator: finish controls, enforce acceptance, then run all five solos."""
import json
import argparse
import getpass
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from common import DATA,ROOT,dump,load
from auto_timing import stage
from solos_extract import DENSE,CONTROLS

HOST=os.environ.get('GPU_HOST','gpu-host')
REMOTE=os.environ.get('GPU_REMOTE_PATH') or '/home/'+getpass.getuser()+'/'+os.environ.get('GPU_REMOTE_DIR','groovemap')
REMOTE_DATA=REMOTE+'/scratch/hybrid_solos'
STATE=DATA/'hybrid-solos-state.json'


def run(args):
    subprocess.run(list(map(str,args)),cwd=ROOT,check=True)


def remote(args):
    command='cd '+shlex.quote(REMOTE)+' && STICKMAP_VISION_DATA='+shlex.quote(REMOTE_DATA)+' '+shlex.join(list(map(str,args)))
    run(['ssh',HOST,command])


def mirror():
    run(['rsync','-a',HOST+':'+REMOTE_DATA+'/REPORT.md',DATA/'HYBRID_SOLOS_REPORT.md'])
    run(['rsync','-a','--include=*/','--include=*.json','--exclude=*',
         HOST+':'+REMOTE_DATA+'/candidates/',DATA/'solos-results/'])
    for name in ('acceptance.json','integrity.json','cv-controls-verification.json'):
        run(['rsync','-a',HOST+':'+REMOTE_DATA+'/'+name,DATA/'solos-audit'/name])


def transfer_and_gate(name,videos):
    stage(name)
    remote(['mkdir','-p',REMOTE_DATA+'/runs/'+name,*[REMOTE_DATA+'/candidates/'+v for v in videos]])
    run(['rsync','-a',DATA/'runs'/name/'',HOST+':'+REMOTE_DATA+'/runs/'])
    for video in videos:
        run(['rsync','-a',str(DATA/'candidates'/video)+'/',HOST+':'+REMOTE_DATA+'/candidates/'+video+'/'])
    remote(['.venv/bin/python','-B','tools/vision_read/solos_gate.py','--workers',str(min(7,len(videos))),'--videos',*videos])
    remote(['.venv/bin/python','-B','tools/vision_read/solos_verify.py'])
    mirror()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume-after-controls',action='store_true')
    args=p.parse_args()
    state=dict(status='waiting_for_controls',control_run='hybrid-solos-controls-v1',dense_run='hybrid-solos-dense-v1',pid=os.getpid())
    dump(STATE,state)
    run(['rsync','-a',str(ROOT/'tools/vision_read')+'/',HOST+':'+REMOTE+'/tools/vision_read/'])
    while True:
        p=DATA/(state['control_run']+'-state.json')
        control=load(p) if p.exists() else {}
        if control.get('status') in ('complete','finished_with_failures'):
            break
        if control.get('status')=='harness_error':
            raise RuntimeError('control harness error; see immutable control logs')
        state.update(status='waiting_for_controls',control_state=control.get('status'),heartbeat_unix=time.time());dump(STATE,state)
        time.sleep(15)
    if args.resume_after_controls:
        if not (DATA/'solos-audit/acceptance.json').exists():
            raise ValueError('no completed control acceptance artifact to resume')
    else:
        state.update(status='control_audio_gates');dump(STATE,state)
        transfer_and_gate(state['control_run'],CONTROLS)
    acceptance=load(DATA/'solos-audit/acceptance.json')
    if acceptance.get('measured_failure'):
        state.update(status='stopped_on_acceptance',acceptance=acceptance,finished_unix=time.time());dump(STATE,state)
        print('STOP: measured acceptance below 95%',flush=True)
        return
    if not acceptance['passed']:
        state['acceptance_limitation']='inconclusive control; continue scratch candidates without claiming overall acceptance'
    state.update(status='dense_gemini');dump(STATE,state)
    run([sys.executable,'-B',ROOT/'tools/vision_read/solos_run.py','--run-name',state['dense_run'],'--videos',*DENSE])
    state.update(status='dense_audio_gates');dump(STATE,state)
    transfer_and_gate(state['dense_run'],DENSE)
    state.update(status='complete',finished_unix=time.time());dump(STATE,state)
    print('COMPLETE: all five dense solos gated and report mirrored',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        state=load(STATE) if STATE.exists() else {}
        state.update(status='error',error=type(error).__name__+': '+str(error),updated_unix=time.time());dump(STATE,state)
        raise
