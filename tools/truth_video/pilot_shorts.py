"""Offline Ollie Shorts crop audit and artifact-derived pilot report.

The audit intentionally includes rejected <8 px heads as diagnostic evidence;
it never changes a reading/timing gate. Manual labels require viewing both
native and reader targets. No detector, listing request or training write runs.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import cv2
import numpy as np

from pilot_new import PILOTS, folders
from pilot_report import exported_fast

ROOT = PILOTS / 'ollieshorts'
AUDIT = ROOT / 'audit'
KEYS = ('filled:1', 'filled:2', 'filled:3', 'filled:5', 'filled:6',
        'filled:7', 'filled:8', 'cross:9', 'cross:8', 'cross:10', 'cross:12', 'cross:-1')


def target_tile(frame, x, y, lines, space):
    """Keep high ledger heads and make the intended member of a chord explicit."""
    left=max(0,round(x-space*2.5));right=min(frame.shape[1],round(x+space*2.5))
    top=max(0,round(min(lines[0]-space*1.5,y-space)))
    bottom=min(len(frame),round(max(lines[-1]+space*1.5,y+space)))
    patch=frame[top:bottom,left:right].copy()
    if patch.ndim==2: patch=cv2.cvtColor(patch,cv2.COLOR_GRAY2BGR)
    scale=min(145/patch.shape[1],210/patch.shape[0])
    patch=cv2.resize(patch,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
    image=np.full((220,175,3),255,np.uint8);image[:len(patch),30:30+patch.shape[1]]=patch
    for i,line in enumerate(lines):
        yy=round((line-top)*scale)
        cv2.putText(image,str(8-i*2),(1,yy+4),0,.4,(200,95,0),1)
        cv2.line(image,(20,yy),(30,yy),(200,95,0),1)
    cv2.circle(image,(30+round((x-left)*scale),round((y-top)*scale)),
               max(3,round(space*scale*.85)),(0,0,230),1)
    return image


def prepare_audit():
    cv2.setNumThreads(1)
    if (AUDIT / 'manual.json').exists():
        raise FileExistsError('preserve the existing visual audit')
    pool = defaultdict(list); seen = set()
    ignored = {'no verified channel labels', 'channel disabled: reading / legend validation incomplete',
               'native staff spacing below 8 px'}
    for folder in folders(['ollieshorts']):
        scan = json.loads((folder / 'notation/scan.json').read_text())
        records = {r['system']: r for r in scan['records']}
        for s in json.loads((folder / 'notation/strip.json').read_text()):
            # Unvisited rows can still supply spatial evidence, never timing.
            if any(q not in ignored | {'page row not visited by cursor'} for q in s['staff']['reasons']):
                continue
            for n in s['notes']:
                key = f"{n['shape']}:{n['pos']}"
                identity = (folder.name, key, round(n['x']), round(n['y']))
                if key not in KEYS or identity in seen: continue
                seen.add(identity)
                pool[key].append(dict(video=folder.name, system=s['system'],
                                      head=n, staff=s['staff'], record=records[s['system']], scan=scan))
    rng = np.random.default_rng(20261002); selected=[]
    for key in KEYS:
        entries=pool[key]; rng.shuffle(entries); balanced=defaultdict(list)
        for entry in entries: balanced[entry['video']].append(entry)
        quota = 8 if key in ('filled:1', 'filled:5', 'cross:9') else 6
        chosen=[]
        while any(balanced.values()) and len(chosen)<quota:
            for video in sorted(balanced):
                if balanced[video] and len(chosen)<quota: chosen.append(balanced[video].pop())
        selected.extend(chosen)
    rng.shuffle(selected); caps={}; samples=[]; tiles=[]
    AUDIT.mkdir(parents=True, exist_ok=True)
    try:
        for index, item in enumerate(selected, 1):
            n,s,r,d = (item[k] for k in ('head','staff','record','scan'))
            folder=ROOT/item['video']/'notation'
            gray=cv2.imread(str(folder/f"system_{item['system']:03d}.png"),0)
            cap=caps.setdefault(item['video'], None)
            if cap is None:
                cap=caps[item['video']]=cv2.VideoCapture(d['video'])
            frame_no=(r['start']+r['end'])//2
            cap.set(1,frame_no); ok,frame=cap.read()
            if not ok: raise ValueError('native frame unavailable')
            base=d['top']+r['crop'][0]; x=n['x']+d.get('left',0); y=n['y']+base
            native=dict(frame=frame_no,x=x,y=y,lines=[line+base for line in s['lines']])
            a=target_tile(frame,x,y,native['lines'],s['spacing'])
            b=target_tile(gray,n['x'],n['y'],s['lines'],s['spacing'])
            cell=np.full((270,360,3),255,np.uint8)
            cv2.putText(cell,f"{index} {item['video']} s{item['system']}",(5,18),0,.45,(0,0,0),1)
            cv2.putText(cell,'native                 reader',(5,40),0,.4,(0,0,0),1)
            cell[46:46+len(a),5:5+a.shape[1]]=a
            cell[46:46+len(b),185:185+b.shape[1]]=b
            tiles.append(cell)
            samples.append(dict(id=index,video=item['video'],system=item['system'],
                                predicted=f"{n['shape']}:{n['pos']}",
                                spacing=s['spacing'],native=native,head=n))
    finally:
        for cap in caps.values(): cap.release()
    for page in range((len(tiles)+11)//12):
        sheet=np.full((810,1440,3),255,np.uint8)
        for k,cell in enumerate(tiles[page*12:(page+1)*12]):
            row,col=divmod(k,4);sheet[row*270:(row+1)*270,col*360:(col+1)*360]=cell
        cv2.imwrite(str(AUDIT/f'blind_{page+1:02d}.png'),sheet)
    (AUDIT/'samples.json').write_text(json.dumps(samples,indent=2)+'\n')
    print(len(samples),'native pairs; enter explicit labels only after inspection')


def audit_report():
    samples=json.loads((AUDIT/'samples.json').read_text())
    manual=json.loads((AUDIT/'manual.json').read_text()); buckets=defaultdict(Counter)
    for s in samples:
        actual=manual[str(s['id'])]
        for key in (s['predicted'],s['video'],'below8' if s['spacing']<8 else 'at_least8'):
            buckets[key]['inspected']+=1
            buckets[key]['correct']+=actual==s['predicted']
    result=dict(inspected=len(samples),correct=sum(manual[str(s['id'])]==s['predicted'] for s in samples),
                groups={k:dict(v) for k,v in buckets.items()},
                errors=[dict(id=s['id'],predicted=s['predicted'],actual=manual[str(s['id'])])
                        for s in samples if manual[str(s['id'])]!=s['predicted']],
                scope='Assistant visual native-source/reader shape and position audit among detected heads; no full-score recall, independent human review, acoustic instrument key or small-staff gate approval')
    (AUDIT/'accuracy.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


def fast_counts(notes):
    """Existing pilot convention: deduplicated class/time, within each system.

    Unisons are not fast gaps; never bridge a rejected gap. Retain the separate
    same-class metric so fast alternation is not claimed as same-drum doubles.
    """
    return exported_fast(notes)


def blockers(systems):
    families=dict(fitted=('fitted attack agreement',),held_out=('held-out agreement',),
                  controls=('ambiguous alignment',),no_cursor=('no-cursor alignment',),
                  phase=('phase alias:', 'notated kicks/snares off', 'fewer than 6 notated kick/snare',
                         'instrument-band phase check unavailable'),
                  clef=('percussion clef',),below8=('native staff spacing below 8 px',),
                  geometry=('weak staff', 'uneven staff', 'expected one staff'),
                  unvisited=('page row not visited',),overlay=('large chromatic overlay',),
                  too_few=('fewer than six notated moments',))
    return {name:sum(any(any(r.startswith(p) for p in prefixes) for r in s['reasons'])
                     for s in systems) for name,prefixes in families.items()}


def report(output='truth'):
    rows=[]
    for folder in folders(['ollieshorts']):
        out=folder/output; result=json.loads((out/'result.json').read_text())
        strip=json.loads((out/'strip.json').read_text())
        alignment=json.loads((out/'alignment.json').read_text())
        spacings=[s['staff']['spacing'] for s in strip if 'spacing' in s['staff']]
        rows.append(dict(video=folder.name,native=json.loads((folder/'meta.json').read_text())['native_video'],
                         systems=result['systems'],clean_staff=sum(not s['staff']['reasons'] for s in strip),
                         observed_heads=sum(len(s['notes']) for s in strip),
                         spacing=[min(spacings),max(spacings)] if spacings else [],
                         strict_systems=result['strict_systems'],scoring_systems=result['scoring_systems'],
                         strict_notes=result['strict_notes'],scoring_notes=result['scoring_notes'],
                         strict_seconds=result['strict_seconds'],scoring_seconds=result['scoring_seconds'],
                         fast={tier:fast_counts(json.loads((out/f'{name}.json').read_text()))
                               for tier,name in (('strict','truth'),('scoring','truth.scoring'))},
                         blockers=blockers(alignment['systems']),
                         rejection_counts=result['rejection_counts'],artifact=str(out)))
    (ROOT/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('audit','audit-report','report'))
    p.add_argument('--output',default='truth')
    args=p.parse_args()
    if args.phase=='audit': prepare_audit()
    elif args.phase=='audit-report': audit_report()
    else: report(args.output)
