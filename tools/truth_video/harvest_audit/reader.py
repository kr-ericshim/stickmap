"""Stratified visual audit; never invent annotations from predictions.

prepare ROOT [--per-label 6]; report ROOT reads manually entered manual.json.
Existing audit manifests are never overwritten by prepare.
"""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import sys
import cv2
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from batch import manifest_read,song_path,dump


def label(n):
    if n['inst']=='hihat':return 'hihat_'+n.get('articulation','unverified')
    if n['inst']=='tom':return f'tom_pos_{n["pos"]}'
    return n['inst']


def prepare(root,quota,only=None,destination=None,include_rejected=False,exported_only=False):
    out=destination or root/'validation/audit';out.mkdir(parents=True,exist_ok=True)
    if (out/'samples.json').exists():raise FileExistsError('audit already exists; use a new root or preserve/rename it explicitly')
    pool=defaultdict(lambda:defaultdict(list))
    for r in manifest_read(root/'manifest.jsonl').values():
        if only and r.get('channel') not in only:continue
        try:folder=song_path(root,r)/'truth'
        except ValueError:continue
        p=folder/'strip.json'
        if not p.exists():continue
        accepted=None
        if exported_only:
            ap=folder/'alignment.scoring.json'
            if not ap.exists():continue
            accepted={s['system'] for s in json.loads(ap.read_text())['systems'] if s['scoring_usable']}
        for s in json.loads(p.read_text()):
            if accepted is not None and s['system'] not in accepted:continue
            if not s['staff'].get('lines'):continue
            # Development source system is not independent validation.
            if r['video']=='jfFi5GeGzUg' and s['system']==3:continue
            if not include_rejected and any(not e.startswith('channel disabled:') and e!='no verified channel labels' for e in s['staff']['reasons']):continue
            for i,n in enumerate(s['notes']):
                if exported_only and not n.get('exportable'):continue
                if n.get('confidence',0)<.84:continue
                item=dict(channel=r['channel'],video=r['video'],slug=r['slug'],system=s['system'],note=i,predicted=n,staff=s['staff'])
                pool[r['channel']][label(n)].append(item)
    samples=[];rng=np.random.default_rng(20261001)
    for channel,labs in sorted(pool.items()):
        for lab,entries in sorted(labs.items()):
            # Rotate across songs before taking additional examples from one.
            rng.shuffle(entries);by=defaultdict(list)
            for entry in entries:by[entry['video']].append(entry)
            ordered=[]
            while by:
                for video in list(by):
                    ordered.append(by[video].pop())
                    if not by[video]:del by[video]
            samples.extend(ordered[:quota])
    for i,s in enumerate(samples):s['id']=i+1
    dump(out/'samples.json',samples)
    for page in range((len(samples)+23)//24):
        canvas=np.full((4*250,6*200,3),255,np.uint8)
        for k,item in enumerate(samples[page*24:(page+1)*24]):
            n=item['predicted'];st=item['staff'];sp=st['spacing']
            g=cv2.imread(str(root/'songs'/item['slug']/'truth'/f'system_{item["system"]:03d}.png'),0)
            x0=max(0,round(n['x']-sp*2));x1=min(g.shape[1],round(n['x']+sp*2))
            crop=cv2.cvtColor(g[:,x0:x1],cv2.COLOR_GRAY2BGR)
            scale=min(180/crop.shape[1],215/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
            x=round((n['x']-x0)*scale);y=round(n['y']*scale)
            cv2.line(crop,(0,y),(7,y),(0,0,220),1);cv2.line(crop,(crop.shape[1]-8,y),(crop.shape[1]-1,y),(0,0,220),1)
            cv2.line(crop,(x,0),(x,6),(0,0,220),1)
            a,b=(k//6)*250,(k%6)*200
            cv2.putText(canvas,f'{item["id"]} {item["channel"]} s{item["system"]}',(b+3,a+18),0,.39,(0,0,0),1)
            canvas[a+27:a+27+len(crop),b+10:b+10+crop.shape[1]]=crop
        cv2.imwrite(str(out/f'blind_{page+1:02d}.png'),canvas,[cv2.IMWRITE_PNG_COMPRESSION,6])
    print(f'{len(samples)} prediction-blind head crops; annotate manual.json before report')


def report(root):
    out=root/'validation/audit';samples=json.loads((out/'samples.json').read_text());manual=json.loads((out/'manual.json').read_text())
    result={}
    for item in samples:
        a=manual.get(str(item['id']))
        if a is None:continue
        ch=result.setdefault(item['channel'],dict(inspected=0,correct=0,labels={},errors=[],videos=[]))
        ch['inspected']+=1;correct=a['label']==label(item['predicted']);ch['correct']+=int(correct)
        if item['video'] not in ch['videos']:ch['videos'].append(item['video'])
        r=ch['labels'].setdefault(label(item['predicted']),dict(inspected=0,correct=0));r['inspected']+=1;r['correct']+=int(correct)
        if not correct:ch['errors'].append(dict(id=item['id'],actual=a['label']))
    for c in result.values():c['scope']='assistant visual reading, stratified detected heads; not recall, acoustic fidelity, or independent human review'
    dump(out/'accuracy.json',result);print(json.dumps(result,indent=2))


def recheck(root):
    """Regression check of reviewed samples against final retained predictions."""
    out=root/'validation/audit';samples=json.loads((out/'samples.json').read_text());manual=json.loads((out/'manual.json').read_text())
    candidate=json.loads((out/'accuracy.json').read_text());result={};cache={}
    for item in samples:
        c=result.setdefault(item['channel'],dict(inspected=0,correct=0,labels={},errors=[],excluded=0,videos=[],
            candidate_audit=candidate[item['channel']],scope='Regression recheck on the same visual samples; retained verified labels only, not a fresh audit, recall, or acoustic accuracy.'))
        folder=root/'songs'/item['slug']/'truth'
        if str(folder) not in cache:cache[str(folder)]=json.loads((folder/'strip.json').read_text())
        old=item['predicted'];same=[s for s in cache[str(folder)] if s['system']==item['system']]
        near=[n for s in same for n in s['notes'] if n.get('exportable') and abs(n['x']-old['x'])<=3 and n['pos']==old['pos']]
        if len(near)!=1:c['excluded']+=1;continue
        pred=label(near[0]);actual=manual[str(item['id'])]['label'];correct=pred==actual
        c['inspected']+=1;c['correct']+=int(correct)
        lab=c['labels'].setdefault(pred,dict(inspected=0,correct=0));lab['inspected']+=1;lab['correct']+=int(correct)
        if item['video'] not in c['videos']:c['videos'].append(item['video'])
        if not correct:c['errors'].append(dict(id=item['id'],predicted=pred,actual=actual))
    dump(out/'accuracy.final.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','report','recheck']);p.add_argument('root',type=Path);p.add_argument('--per-label',type=int,default=8)
    p.add_argument('--only',nargs='+');p.add_argument('--out',type=Path);p.add_argument('--include-rejected',action='store_true');p.add_argument('--exported-only',action='store_true')
    a=p.parse_args();prepare(a.root,a.per_label,a.only,a.out,a.include_rejected,a.exported_only) if a.command=='prepare' else recheck(a.root) if a.command=='recheck' else report(a.root)
