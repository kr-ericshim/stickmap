"""Verify frozen control CV inputs and protected canonical files."""
import argparse
import json
import sys
from common import DATA,ROOT,dump,load,sha
from solos_extract import CONTROLS


def cv_controls():
    from hybrid import cv_prediction
    fields=['notes','staff','barlines','raw_notes','raw_staff','error','raw_error','width','height','variant']
    rows=[]
    for video in CONTROLS:
        folder=DATA/'native'/video
        for item in load(folder/'manifest.json')['systems']:
            old=load(DATA/'hybrid/jobs'/item['id']/'cv.json')
            current=json.loads(json.dumps(cv_prediction(item,folder)))
            changed=[key for key in fields if old.get(key)!=current.get(key)]
            anchor=item['automatic_anchor'];proof=anchor.get('staff_continuation')
            stable=True
            if proof:
                import numpy as np
                stable=bool(np.max(np.ptp(np.asarray(proof['native_lines']),axis=0))<=old['staff']['spacing']*.5)
            rows.append(dict(id=item['id'],changed_fields=changed,registered_geometry_stable=stable))
    result=dict(passed=all(not r['changed_fields'] and r['registered_geometry_stable'] for r in rows),systems=rows,
                native_adapter_sha256=sha(ROOT/'tools/vision_read/native_cv.py'))
    dump(DATA/'cv-controls-verification.json',result)
    return result


def integrity():
    before={}
    for name in ('protected-before.json','protected-before-controls.json'):
        path=DATA/name
        if path.exists():before.update(load(path))
    changed=[]
    for name,digest in before.items():
        path=ROOT/name
        if not path.exists() or sha(path)!=digest:
            changed.append(name)
    result=dict(checked=len(before),changed=changed,passed=not changed,
                vision_code={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'tools/vision_read').glob('*.py'))})
    dump(DATA/'integrity.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cv-controls',action='store_true')
    args=p.parse_args()
    if args.cv_controls:
        r=cv_controls();print('control CV equivalence',r['passed'],'systems',len(r['systems']))
        if not r['passed']:raise SystemExit(1)
    r=integrity();print('protected integrity',r['passed'],'files',r['checked'],'changed',r['changed'])
    if not r['passed']:raise SystemExit(1)
