"""Head-local chromatic transitions, distinct from a pale background playhead.

These are video timestamps (one-frame uncertainty), not acoustic onset proof.
The existing audio held-out and shifted-control gates still apply.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from align import moments


def transition(values, frames, fps, stable=None):
    on=np.asarray(values,bool);frames=np.asarray(frames,int)
    if len(on)<4 or on[0]:return None
    rises=np.flatnonzero(on & ~np.r_[False,on[:-1]])
    valid=[i for i in rises if i>=2 and i+1<len(on) and on[i+1]
           and not on[i-1] and not on[i-2] and frames[i]-frames[i-1]==1
           and frames[i+1]-frames[i]==1]
    if stable is not None:valid=[i for i in valid if all(stable[i-2:i+2])]
    if len(valid)!=1:return None
    i=valid[0]
    return dict(time=float((frames[i]-.5)/fps),first_frame=int(frames[i]),
                uncertainty_seconds=1/fps,method='stable head ink; chroma rising edge with two-frame persistence')


def anchors(s, fps):
    """Require broad, ordered visual evidence, including simultaneous heads."""
    xs,groups=moments(s);cx=[];ct=[]
    for x,group in zip(xs,groups):
        ts=[s['notes'][i]['visual_timing']['time'] for i in group if 'visual_timing' in s['notes'][i]]
        if ts and np.ptp(ts)<=1.5/fps:
            cx.append(float(x));ct.append(float(np.median(ts)))
    if (len(cx)>=6 and len(cx)>=len(xs)*.7 and np.all(np.diff(ct)>0)
            and np.ptp(cx)>.7*np.ptp(xs)):
        return cx,ct
    return [],[]


def attach(folder, page_notes=False):
    folder=Path(folder);scan=json.loads((folder/'scan.json').read_text())
    systems=json.loads((folder/'strip.json').read_text());records={r['system']:r for r in scan['records']}
    samples={};patches={}
    for s in systems:
        # An opt-in full-page reader may establish a row visit from local note
        # colour even when the box was missed. All other staff failures remain.
        if any(not reason.startswith('channel disabled:') and
               not (page_notes and reason=='page row not visited by cursor')
               for reason in s['staff'].get('reasons',[])):continue
        gray=cv2.imread(str(folder/f"system_{s['system']:03d}.png"),0)
        sp=s['staff']['spacing'];arr=[]
        for n in s['notes']:
            x,y=round(n['x']),round(n['y']);d=max(2,round(sp*.48))
            x0=max(0,x-d);x1=min(gray.shape[1],x+d+1);y0=max(0,y-d);y1=min(len(gray),y+d+1)
            mask=gray[y0:y1,x0:x1]<105
            arr.append((x0,x1,y0,y1,mask))
        patches[s['system']]=arr;samples[s['system']]=([], [[] for _ in arr], [[] for _ in arr])
    cap=cv2.VideoCapture(scan['video']);frame_no=0
    ordered=sorted((records[k] for k in patches),key=lambda r:r['start']);active=[];next_row=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            while next_row<len(ordered) and ordered[next_row]['start']<=frame_no:
                active.append(ordered[next_row]);next_row+=1
            active=[r for r in active if r['end']>frame_no]
            for r in active:
                sid=r['system'];frames,values,stability=samples[sid];frames.append(frame_no)
                base=scan['top']+r['crop'][0]
                for vals,stable,(x0,x1,y0,y1,mask) in zip(values,stability,patches[sid]):
                    p=frame[base+y0:base+y1,scan.get('left',0)+x0:scan.get('left',0)+x1]
                    lo=p.min(2).astype(float);hi=p.max(2).astype(float)
                    colored=(hi-lo>55)&(hi>80)&(lo<115)
                    vals.append(bool(mask.sum()>=4 and colored[mask].mean()>.4))
                    live=np.where((hi>170)&(lo>105),hi,lo)<115
                    stable.append(bool((mask&live).sum()/max(1,(mask|live).sum())>=.5))
            frame_no+=1
    finally:cap.release()
    evidence=[]
    for s in systems:
        if s['system'] not in samples:continue
        frames,values,stability=samples[s['system']]
        for i,(n,vals,stable) in enumerate(zip(s['notes'],values,stability)):
            found=transition(vals,frames,scan['fps'],stable)
            if found:
                n['visual_timing']=found;evidence.append(dict(system=s['system'],note=i,**found))
        # Never extrapolate far from sparse color changes or reorder a reset.
        cx,ct=anchors(s,scan['fps'])
        if cx:
            s['color_x']=cx;s['color_t']=ct
            if page_notes and 'page row not visited by cursor' in s['staff']['reasons']:
                s['staff']['reasons'].remove('page row not visited by cursor')
                s['display_interval']=s['interval']
                s['interval']=[ct[0],ct[-1]+1/scan['fps']]
                s['staff']['row_visit_evidence']='at least six stable local colour edges, >=70% of moments and >70% of their x span'
    (folder/'strip.json').write_text(json.dumps(systems,indent=2))
    (folder/'color-timing.json').write_text(json.dumps(dict(fps=scan['fps'],events=evidence,
        limitation='Video frame timing only; no subframe precision or independent audio onset verification'),indent=2))
    return evidence
