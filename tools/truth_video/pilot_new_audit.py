"""Prepare native-pair crop sheets; compare only explicit visual manual labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

import pilot_audit
from pilot_new import PILOTS, folders


def prepare(channels):
    for channel in channels:
        pilot_audit.PILOTS=PILOTS/channel
        sources=[(p.name,'notation') for p in folders([channel])]
        pilot_audit.prepare(sources,PILOTS/f'audit-{channel}',
                            positions=[1,3,5,6,7],quota=6,candidate=True)


def report(channel):
    out=PILOTS/f'audit-{channel}'
    samples=json.loads((out/'samples.json').read_text())
    manual=json.loads((out/'manual.json').read_text())
    by_position={};by_video={}
    for s in samples:
        actual=manual.get(str(s['id']))
        if actual is None:raise ValueError(f'missing manual label {channel}/{s["id"]}')
        expected=f'{s["predicted"]["shape"]}:{s["predicted"]["pos"]}'
        for bucket,key in ((by_position,expected),(by_video,s['video'])):
            row=bucket.setdefault(key,dict(inspected=0,correct=0))
            row['inspected']+=1;row['correct']+=actual==expected
    result=dict(channel=channel,inspected=len(samples),
                correct=sum(r['correct'] for r in by_position.values()),
                by_position=by_position,by_video=by_video,
                scope='Assistant native-source/reader target audit of detected filled heads; no independent human/audio verification or full-score recall. Position conventions only; tom sizes withheld.')
    (out/'accuracy.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','report'))
    p.add_argument('--channels',nargs='+',required=True)
    a=p.parse_args()
    if a.phase=='prepare':prepare(a.channels)
    else:
        for channel in a.channels:report(channel)
