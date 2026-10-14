"""Prediction-blind native/reconstructed head crops for the offline pilots.

Sampling uses staff positions, but labels/flags are hidden in the sheets. Manual
decisions must come from viewing the target and its native source. No detector,
timing fitter or scorer is called. Repeated static glyph locations count once.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import cv2
import numpy as np

from pilot1080 import PILOTS


def tile(frame, x, y, lines, space):
    x0,x1=max(0,round(x-space*2.5)),min(frame.shape[1],round(x+space*2.5))
    y0,y1=max(0,round(lines[0]-space*1.5)),min(len(frame),round(lines[-1]+space*1.5))
    im=frame[y0:y1,x0:x1].copy()
    if im.ndim==2:im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    for i,line in enumerate(lines):
        yy=round(line-y0)
        if 0<=yy<len(im):
            cv2.line(im,(0,yy),(4,yy),(230,150,30),1)
            cv2.putText(im,str(8-i*2),(0,yy-2),0,.22,(230,150,30),1)
    scale=min(170/im.shape[1],220/im.shape[0])
    im=cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
    yy=round((y-y0)*scale);xx=round((x-x0)*scale)
    cv2.line(im,(0,yy),(8,yy),(0,0,230),2)
    cv2.line(im,(im.shape[1]-9,yy),(im.shape[1]-1,yy),(0,0,230),2)
    cv2.line(im,(xx,0),(xx,7),(0,0,230),2)
    return im


def prepare(sources, out, positions=range(9), quota=6, candidate=False, include_unconfirmed_clef=False):
    out=Path(out)
    if (out/'manual.json').exists():raise FileExistsError('preserve completed manual audit')
    out.mkdir(parents=True,exist_ok=True);pool=defaultdict(list);seen=set()
    for video, name in sources:
        folder=PILOTS/video/name
        scan=json.loads((folder/'scan.json').read_text())
        records={r['system']:r for r in scan['records']}
        for s in json.loads((folder/'strip.json').read_text()):
            ignored={'no verified channel labels','channel disabled: reading / legend validation incomplete'} if candidate else set()
            if include_unconfirmed_clef:ignored.add('percussion clef / first bar not confirmed')
            if any(r not in ignored for r in s['staff']['reasons']):continue
            for i,n in enumerate(s['notes']):
                if n['shape']!='filled' or n['pos'] not in positions:continue
                key=(video,n['pos'],round(n['x']),round(n['y']))
                if not scan.get('reconstructed_scroll') and key in seen:continue
                seen.add(key)
                pool[n['pos']].append(dict(video=video,folder=name,system=s['system'],note=i,
                                          predicted=n,staff=s['staff'],record=records[s['system']],scan=scan))
    rng=np.random.default_rng(20261002);samples=[]
    for pos in positions:
        # Balance videos instead of filling a new layout's audit from one source.
        entries=pool[pos];rng.shuffle(entries);by_video=defaultdict(list)
        for entry in entries:by_video[entry['video']].append(entry)
        ordered=[]
        while any(by_video.values()):
            for video in sorted(by_video):
                if by_video[video]:ordered.append(by_video[video].pop())
        samples.extend(ordered[:quota])
    rng.shuffle(samples)
    caps={};tiles=[]
    try:
        for index,item in enumerate(samples,1):
            item['id']=index;n=item['predicted'];s=item['staff'];r=item['record'];d=item.pop('scan')
            folder=PILOTS/item['video']/item['folder']
            gray=cv2.imread(str(folder/f"system_{item['system']:03d}.png"),0)
            if item['video'] not in caps:caps[item['video']]=cv2.VideoCapture(d['video'])
            cap=caps[item['video']]
            if d.get('reconstructed_scroll'):
                choices=[(abs(n['x']+r['scroll_x_origin']+dx-d['right']/2),t,dx)
                         for t,dx in zip(r['sample_frames'],r['scroll_offsets'])
                         if d.get('motion_left',0)+s['spacing']*3 < n['x']+r['scroll_x_origin']+dx < d['right']-s['spacing']*3
                         and abs(n['x']+r['scroll_x_origin']+dx-d['right']/2)>s['spacing']*3]
                if not choices:raise ValueError('no native non-cursor crop for audit head')
                _,frame_no,dx=min(choices)
                x=n['x']+r['scroll_x_origin']+dx
            else:
                frame_no=(r['start']+r['end'])//2;x=n['x']+d.get('left',0)
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no);ok,frame=cap.read()
            if not ok:raise ValueError('native source frame unavailable')
            base=d['top']+r['crop'][0];y=n['y']+base
            item['native']=dict(frame=int(frame_no),x=float(x),y=float(y),
                                lines=[line+base for line in s['lines']])
            a=tile(frame,x,y,item['native']['lines'],s['spacing'])
            b=tile(gray,n['x'],n['y'],s['lines'],s['spacing'])
            cell=np.full((270,360,3),255,np.uint8)
            cv2.putText(cell,f'{index} {item["video"]} s{item["system"]}',(5,18),0,.45,(0,0,0),1)
            cv2.putText(cell,'native                 reader',(5,40),0,.4,(0,0,0),1)
            cell[46:46+len(a),5:5+a.shape[1]]=a
            cell[46:46+len(b),185:185+b.shape[1]]=b
            tiles.append(cell)
    finally:
        for cap in caps.values():cap.release()
    for page in range((len(tiles)+11)//12):
        canvas=np.full((810,1440,3),255,np.uint8)
        for k,cell in enumerate(tiles[page*12:(page+1)*12]):
            a,b=divmod(k,4);canvas[a*270:(a+1)*270,b*360:(b+1)*360]=cell
        cv2.imwrite(str(out/f'blind_{page+1:02d}.png'),canvas)
    (out/'samples.json').write_text(json.dumps(samples,indent=2)+'\n')
    print(out,len(samples),'native pairs',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',nargs='+',required=True,help='VIDEO:OUTPUT_FOLDER')
    p.add_argument('--out',required=True)
    p.add_argument('--positions',nargs='+',type=int,default=list(range(9)))
    p.add_argument('--quota',type=int,default=6)
    a=p.parse_args();prepare([s.split(':',1) for s in a.sources],a.out,a.positions,a.quota)
