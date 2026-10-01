"""Create reproducible, prediction-blind crops for manual review, not an automatic accuracy claim."""
import json
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np


def label(n):
    return 'hihat_'+n['articulation'] if n['inst']=='hihat' else n['inst']


def prepare(root):
    root=Path(root);out=root/'audit';out.mkdir(exist_ok=True)
    pool=defaultdict(list)
    excluded={('jrod',1),('jrod',4),('devon-taylor',0),('tension',1)}
    for p in sorted(root.glob('*/strip.json')):
        if p.parent.name=='cj-thompson':continue
        for s in json.loads(p.read_text()):
            if (p.parent.name,s['system']) in excluded or s['staff']['reasons']:continue
            for i,n in enumerate(s['notes']):
                item=dict(video=p.parent.name,system=s['system'],note=i,predicted=n,staff=s['staff'])
                pool[label(n)].append(item)
    quotas={'kick':25,'snare':25,'tom':25,'hihat_closed':25,'hihat_open':15,'hihat_pedal':12,
            'ride':18,'ride_bell':12,'crash':18,'side_stick':10,'cymbal_unknown':5}
    rng=np.random.default_rng(20261001);samples=[]
    for lab,num in quotas.items():
        entries=pool[lab]
        samples.extend(entries[int(i)] for i in rng.choice(len(entries),min(num,len(entries)),replace=False))
    rng.shuffle(samples)
    for i,item in enumerate(samples):item['id']=i+1
    (out/'samples.json').write_text(json.dumps(samples,indent=2))
    for page in range((len(samples)+29)//30):
        canvas=np.full((5*250,6*200,3),255,np.uint8)
        for k,item in enumerate(samples[page*30:(page+1)*30]):
            n=item['predicted'];staff=item['staff'];sp=staff['spacing']
            gray=cv2.imread(str(root/item['video']/f"system_{item['system']:03d}.png"),0)
            x0,x1=max(0,round(n['x']-sp*2.5)),min(gray.shape[1],round(n['x']+sp*2.5))
            y0,y1=max(0,round(staff['lines'][0]-sp*4)),min(gray.shape[0],round(staff['lines'][-1]+sp*2))
            crop=cv2.cvtColor(gray[y0:y1,x0:x1],cv2.COLOR_GRAY2BGR)
            scale=min(180/crop.shape[1],215/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
            y=round((n['y']-y0)*scale);x=round((n['x']-x0)*scale)
            cv2.line(crop,(0,y),(7,y),(0,0,220),1);cv2.line(crop,(crop.shape[1]-8,y),(crop.shape[1]-1,y),(0,0,220),1)
            cv2.line(crop,(x,0),(x,5),(0,0,220),1)
            a,b=(k//6)*250,(k%6)*200
            cv2.putText(canvas,f"{item['id']} {item['video'][:12]} s{item['system']}",(b+3,a+17),0,.38,(0,0,0),1)
            canvas[a+25:a+25+len(crop),b+10:b+10+crop.shape[1]]=crop
        cv2.imwrite(str(out/f'blind_{page+1:02d}.png'),canvas)
    print(len(samples),'heads sampled')


def report(root):
    root=Path(root);out=root/'audit'
    samples=json.loads((out/'samples.json').read_text())
    annotations=json.loads((out/'manual.json').read_text())
    scores=defaultdict(lambda:dict(verified=0,correct=0,errors=[]))
    for item in samples:
        actual=annotations[str(item['id'])]['label'];pred=label(item['predicted'])
        row=scores[pred];row['verified']+=1;row['correct']+=int(actual==pred)
        if actual!=pred:row['errors'].append(dict(id=item['id'],actual=actual))
    (out/'accuracy.json').write_text(json.dumps(dict(scores),indent=2))
    print(json.dumps(dict(scores),indent=2))


if __name__=='__main__':
    import sys
    (report if '--report' in sys.argv else prepare)(sys.argv[1])
