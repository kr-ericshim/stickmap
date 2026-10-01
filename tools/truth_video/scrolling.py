"""Reconstruct white scrolling notation from independently measured image motion.

No attacks or detector outputs enter this pass. Frames are translated using
white notation correlation; temporal medians suppress the moving video behind
the transparent staff. Red-cursor pixels are masked, not treated as white ink.
Only the span actually traversed by the cursor is retained for note reading.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from heads import staffs


def write(song,out,start_at=12,end_at=204):
    out.mkdir(parents=True,exist_ok=True)
    video=song/('video.hi.mp4' if (song/'video.hi.mp4').exists() else 'video.mp4')
    cap=cv2.VideoCapture(str(video));fps=cap.get(5);duration=cap.get(7)/fps
    records=[];systems=[];evidence=[]
    # The inspected VFJams layout has fixed staff lines and a stationary red box.
    for start in np.arange(start_at,min(end_at,duration-4),4):
        frames=[];offsets=[];scores=[];cursor=[];prev=None;offset=0
        for t in np.arange(start,start+4.001,.2):
            cap.set(cv2.CAP_PROP_POS_MSEC,float(t*1000));ok,f=cap.read()
            if not ok:continue
            crop=f[810:970]
            ink=(crop.min(axis=2)>190).astype('uint8')*255
            red=(crop[:,:,2].astype(float)-crop[:,:,1]>45)&(crop[:,:,2]>120)
            col=np.where(red.sum(axis=0)>60)[0]
            cx=float(np.median(col)) if len(col) else 974.
            ink[:,max(0,int(cx)-25):int(cx)+25]=0
            # Mask horizontal staff lines; they contain no translation evidence.
            moving=ink.copy()
            for y in [80,94,109,123,137]:moving[max(0,y-2):y+3]=0
            if prev is not None:
                template=prev[:,350:1450]
                response=cv2.matchTemplate(moving,template,cv2.TM_CCORR_NORMED)
                # Scrolling left, permitting pauses; discontinuities reject a block.
                lo,hi=150,365;idx=lo+int(np.argmax(response[0,lo:hi]));q=float(response[0,idx])
                offset+=idx-350;scores.append(q)
            prev=moving;frames.append(ink);offsets.append(offset);cursor.append((float(t),cx-offset))
        sid=len(records);reason=[]
        if not scores or min(scores)<.6:reason.append('scroll image correlation below .60')
        x0=int(min(x for t,x in cursor)-40);x1=int(max(x for t,x in cursor)+40)
        canvas=[]
        for im,dx in zip(frames,offsets):
            # A source pixel x belongs at global coordinate x-dx.
            translated=cv2.warpAffine(im,np.float32([[1,0,-dx-x0],[0,1,0]]),(x1-x0,len(im)),borderValue=0)
            canvas.append(translated)
        # Thin antialiased white strokes occupy alternating pixels while moving.
        # A median erases them. 25% temporal support retains these strokes while
        # rejecting most unrelated moving imagery; inspect every accepted block.
        image=np.where(np.mean(np.array(canvas)>0,axis=0)>=.25,0,255).astype('uint8')
        found=staffs(image,min_space=8)
        if len(found)!=1:reason.append(f'scroll reconstruction has {len(found)} staves')
        sp=found[0][1] if len(found)==1 else 0
        cv2.imwrite(str(out/f'system_{sid:03d}.png'),image)
        records.append(dict(system=sid,start=round(start*fps),end=round((start+4)*fps),row=0,page_rows=1,crop=[0,160],native_space=sp,
                            cursor_frames=[round(t*fps) for t,x in cursor],cursor_x=[x-x0 for t,x in cursor],
                            scroll_reasons=reason,read_x=[cursor[0][1]-x0,cursor[-1][1]-x0]))
        evidence.append(dict(system=sid,min_correlation=min(scores) if scores else 0,offsets=offsets,reasons=reason))
    cap.release()
    # No full-video cursor array: records explicitly own their observations.
    data=dict(version=2,video=str(video.resolve()),fps=fps,top=810,bottom=970,records=records,
              systems=[r['start'] for r in records],cursor=[],frames=round(duration*fps),record_cursors=True)
    (out/'scan.json').write_text(json.dumps(data));(out/'scroll-motion.json').write_text(json.dumps(evidence,indent=2))
    (out/'song.txt').write_text(str(song.resolve()))
    print(f'{out.name}: {len(records)} scrolling blocks',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('song',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--start',type=float,default=12);p.add_argument('--end',type=float,default=204)
    a=p.parse_args();write(a.song,a.out,a.start,a.end)
