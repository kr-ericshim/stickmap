"""Disjoint native crop audit for the two woshudiue parenthesis glyphs."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from harvest_flags import ghost_score
from pilot_audit import tile
from pilot1080 import PILOTS


def prepare(out, exclude):
    out=Path(out);out.mkdir(exist_ok=True)
    if (out/'manual.json').exists():raise FileExistsError('preserve completed audit')
    old=json.loads(Path(exclude).read_text())
    keys={(s['video'],s['system'],round(s['predicted']['x'])) for s in old}
    profile=json.loads((PILOTS/'woshudiue-development.json').read_text())
    configs=[profile['ghost_template'],*profile.get('ghost_templates',[])]
    rng=np.random.default_rng(20261003);items=[]
    for video in ('SHvfn-bM_rg','8Eh6uT0j_D4'):
        folder=PILOTS/video/'clef-read';scan=json.loads((folder/'scan.json').read_text())
        records={r['system']:r for r in scan['records']};pool={True:[],False:[]}
        for s in json.loads((folder/'strip.json').read_text()):
            if s['staff']['reasons'] or video=='SHvfn-bM_rg' and s['system']==5:continue
            gray=cv2.imread(str(folder/f"system_{s['system']:03d}.png"),0)
            for i,n in enumerate(s['notes']):
                if (n['shape']!='filled' or n['pos']!=5 or
                    (video,s['system'],round(n['x'])) in keys or
                    video=='8Eh6uT0j_D4' and s['system']==5 and abs(n['x']-1022)<s['staff']['spacing']):continue
                scores=[ghost_score(gray,n,s['staff']['spacing'],cfg) for cfg in configs]
                positive=any(q>=cfg['threshold'] for q,cfg in zip(scores,configs))
                pool[positive].append(dict(video=video,system=s['system'],note=i,predicted=n,
                    staff=s['staff'],record=records[s['system']],scores=scores,positive=positive))
        for flag,rows in pool.items():rng.shuffle(rows);items+=rows[:6]
    rng.shuffle(items);cells=[];caps={}
    try:
        for index,item in enumerate(items,1):
            item['id']=index;video=item['video'];folder=PILOTS/video/'clef-read'
            scan=json.loads((folder/'scan.json').read_text());r=item['record'];s=item['staff'];n=item['predicted']
            if video not in caps:caps[video]=cv2.VideoCapture(scan['video'])
            caps[video].set(1,(r['start']+r['end'])//2);ok,frame=caps[video].read()
            if not ok:raise ValueError('source frame unavailable')
            base=scan['top']+r['crop'][0]
            a=tile(frame,n['x'],n['y']+base,[y+base for y in s['lines']],s['spacing'])
            gray=cv2.imread(str(folder/f"system_{item['system']:03d}.png"),0)
            b=tile(gray,n['x'],n['y'],s['lines'],s['spacing'])
            cell=np.full((270,360,3),255,np.uint8)
            cv2.putText(cell,f'{index} {video} s{item["system"]}',(5,18),0,.45,(0,0,0),1)
            cell[46:46+len(a),5:5+a.shape[1]]=a;cell[46:46+len(b),185:185+b.shape[1]]=b;cells.append(cell)
    finally:
        for cap in caps.values():cap.release()
    for page in range((len(cells)+11)//12):
        canvas=np.full((810,1440,3),255,np.uint8)
        for k,cell in enumerate(cells[page*12:(page+1)*12]):
            a,b=divmod(k,4);canvas[a*270:(a+1)*270,b*360:(b+1)*360]=cell
        cv2.imwrite(str(out/f'blind_{page+1:02d}.png'),canvas)
    (out/'samples.json').write_text(json.dumps(items,indent=2)+'\n')
    print(len(items),'fresh paired flag crops',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--exclude',required=True)
    a=p.parse_args();prepare(a.out,a.exclude)
