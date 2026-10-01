"""Light-paper scrolling reconstruction with image-only translation checks.

Periodic/ambiguous matches, changing staff geometry and missing cursors reject
blocks. No detector classes or audio attacks enter the reconstruction.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from heads import staffs
from layouts import ink,cursor,detect


def motion_image(gray):
    im=(gray<110).astype(np.uint8)*255
    # Staff lines and lyrics carry no unique horizontal motion evidence.
    found=staffs(gray,min_space=6)
    for ls,sp in found:
        for y in ls:im[max(0,round(y)-1):round(y)+2]=0
    return im


def shift(a,b,max_shift=80):
    if a.shape!=b.shape or a.shape[1]<max_shift*3:return None
    template=a[:,max_shift:-max_shift]
    if np.count_nonzero(template)<40:return None
    response=cv2.matchTemplate(b,template,cv2.TM_CCORR_NORMED)[0]
    k=int(np.argmax(response));q=float(response[k]);other=response.copy()
    other[max(0,k-3):min(len(other),k+4)]=-1
    uniqueness=q-float(other.max())
    if not np.isfinite(q) or q<.88 or uniqueness<.025:return None
    return dict(dx=k-max_shift,correlation=q,uniqueness=uniqueness)


def probe(video,geometry):
    cap=cv2.VideoCapture(str(video));fps=cap.get(5);count=cap.get(7);evidence=[]
    for frac in (.25,.5,.75):
        pair=[]
        for frame_no in (round(count*frac),round(count*frac+fps*.2)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no);ok,f=cap.read()
            if ok:pair.append(motion_image(ink(f[geometry['top']:geometry['bottom']])))
        if len(pair)==2:evidence.append(shift(*pair))
    cap.release()
    return sum(bool(e and e['dx'] < -2) for e in evidence)>=2,evidence


def write(video,out,geometry):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video));fps=cap.get(5);total=int(cap.get(7))
    records=[];evidence=[];n=0;block=[];stride=max(1,round(fps*.2));block_frames=max(1,round(fps*4))

    def finish(samples,end):
        if len(samples)<5:return
        start=samples[0][0];offset=0;offsets=[0];qs=[];reasons=[]
        for previous,current in zip(samples,samples[1:]):
            delta=shift(motion_image(previous[1]),motion_image(current[1]))
            if delta is None or delta['dx']>2:
                reasons.append('scroll translation ambiguous or rightward discontinuity');break
            offset+=delta['dx'];offsets.append(offset);qs.append(delta)
        sid=len(records)
        evidence.append(dict(system=sid,start_frame=start,end_frame=end,matches=qs,reasons=reasons))
        if reasons:
            image=samples[0][1];sp=0;read_x=None;cx=[];ct=[]
        else:
            seen=[(t,b[0]-dx) for (t,g,b),dx in zip(samples,offsets) if b]
            if len(seen)<len(samples)*.8 or not seen or seen[-1][1]-seen[0][1]<30:
                reasons.append('scroll cursor missing or insufficient traversed span')
                image=samples[0][1];sp=0;read_x=None;cx=[];ct=[]
            else:
                x0=int(seen[0][1]-45);x1=int(seen[-1][1]+45)
                images=[];masks=[]
                for (t,g,b),dx in zip(samples,offsets):
                    transform=np.float32([[1,0,-dx-x0],[0,1,0]])
                    images.append(cv2.warpAffine(g,transform,(x1-x0,len(g)),borderValue=255))
                    masks.append(cv2.warpAffine(np.ones_like(g),transform,(x1-x0,len(g)),flags=cv2.INTER_NEAREST,borderValue=0))
                values=np.stack(images).astype(np.float32);values[np.stack(masks)==0]=np.nan
                # Require ink supported in several registered frames. Unlike a
                # union this does not accumulate single-frame compression noise.
                valid=np.sum(np.isfinite(values),axis=0)
                support=np.sum(values<110,axis=0)/np.maximum(1,valid)
                image=np.where((support>=.4)&(valid>=3),0,255).astype(np.uint8)
                found=staffs(image,min_space=6);sp=found[0][1] if len(found)==1 else 0
                if len(found)!=1:reasons.append('scroll reconstructed staff not unique')
                cx=[float(x-x0) for t,x in seen];ct=[t for t,x in seen]
                read_x=[cx[0],cx[-1]-.01]
        cv2.imwrite(str(out/f'system_{sid:03d}.png'),image,[cv2.IMWRITE_PNG_COMPRESSION,6])
        records.append(dict(system=sid,start=start,end=end,row=0,page_rows=1,crop=[0,len(image)],
            native_space=sp,cursor_frames=ct,cursor_x=cx,scroll_reasons=reasons,read_x=read_x))

    try:
        while True:
            ok,f=cap.read()
            if not ok:break
            if n and n%block_frames==0:finish(block,n);block=[]
            if n%stride==0:
                crop=f[geometry['top']:geometry['bottom']];block.append((n,ink(crop),cursor(crop)))
            n+=1
        finish(block,n)
    finally:cap.release()
    data=dict(version=3,video=str(Path(video).resolve()),fps=fps,frames=n,**geometry,
              record_cursors=True,reconstructed_scroll=True,records=records,
              systems=[r['start'] for r in records],cursor=[])
    (out/'scan.json').write_text(json.dumps(data));(out/'scroll-motion.json').write_text(json.dumps(evidence,indent=2))
    return data
