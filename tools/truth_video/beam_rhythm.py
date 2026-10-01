"""Read complete quarter-note beam groups in the verified Zildjian layout.

This is deliberately narrow: two/three beams imply 16ths/32nds. A group must
sum to exactly one quarter; every notated moment must be assigned exactly once.
Rest-containing or incomplete groups retain the original interval prior.
Neither attacks nor detector predictions are read by this module.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from align import moments


def read(folder):
    folder=Path(folder);ss=json.loads((folder/'strip.json').read_text());evidence=[]
    for s in ss:
        for key in ('rhythm_x','rhythm_t','rhythm_evidence','rhythm_calibration_times'):
            s.pop(key,None)
        x,groups=moments(s)
        im=cv2.imread(str(folder/f'system_{s["system"]:03d}.png'),0)
        if not s.get('staff',{}).get('lines'):continue
        ink=(im<100).astype('uint8');ink[:20]=0
        ink[round(s['staff']['lines'][0]-s['staff']['spacing']):]=0
        horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,25),np.uint8))
        # Short partial third beams between two 32nds can be under 25 px long.
        # Keep these for duration reading; only long beams establish groups.
        levels_ink=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,8),np.uint8))
        cols=np.where(horizontal.sum(0)>=4)[0]
        runs=np.split(cols,np.where(np.diff(cols)>3)[0]+1)
        beams=[(int(r[0]),int(r[-1])) for r in runs if len(r)>40]
        beat=np.full(len(x),np.nan);details=[];valid=True
        for k,(a,b) in enumerate(beams):
            ids=np.where((x>=a-17)&(x<=b-3))[0]
            if len(ids)<3:valid=False;break
            counts=[]
            for i,j in zip(ids,ids[1:]):
                col=round((x[i]+x[j])/2+7)
                group_rows=np.where(horizontal[:,a:b+1].sum(axis=1)>10)[0]
                top=int(group_rows[0]) if len(group_rows) else 20
                rows=np.where(levels_ink[top:top+32,col]>0)[0]
                chunks=np.split(rows,np.where(np.diff(rows)>1)[0]+1)
                counts.append(sum(len(r)>=2 for r in chunks))
            levels=[max(counts[max(0,i-1):min(len(counts),i+1)]) for i in range(len(ids))]
            if any(n not in (2,3) for n in levels):valid=False;break
            durations=[2**(2-n)/4 for n in levels]
            if abs(sum(durations)-1)>1e-8 or np.isfinite(beat[ids]).any():valid=False;break
            beat[ids]=k+np.r_[0,np.cumsum(durations[:-1])]
            details.append(dict(beam=[a,b],moment_indices=ids.tolist(),beam_levels=levels,durations_quarters=durations))
        valid=bool(valid and len(beams)>1 and np.isfinite(beat).all())
        row=dict(system=s['system'],valid=valid,groups=details,all_moments=len(x))
        # Independently read complete opening groups can establish the shared
        # video/audio lag for OTHER systems, even if a later rest is unsupported.
        # The printed opening tempo is quarter=88. Never use an opening rest page.
        if len(details)>=2 and details[0]['moment_indices'][0]==0:
            ids=[i for g in details for i in g['moment_indices']]
            s['rhythm_calibration_times']=(s['interval'][0]+beat[ids]*(60/88)).tolist()
        if valid:
            period=(s['interval'][1]-s['interval'][0])/len(beams)
            s['rhythm_x']=x.tolist();s['rhythm_t']=(s['interval'][0]+beat*period).tolist()
            s['rhythm_evidence']='beam-rhythm.json: complete 16th/32nd quarter groups; page interval establishes tempo prior'
            row['quarter_seconds']=period
        evidence.append(row)
    (folder/'strip.json').write_text(json.dumps(ss,indent=2));(folder/'beam-rhythm.json').write_text(json.dumps(evidence,indent=2))
    print(folder.name,'complete beam rhythm pages:',[r['system'] for r in evidence if r['valid']],flush=True)


if __name__=='__main__':
    import sys
    read(sys.argv[1])
