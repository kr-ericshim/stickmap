"""Streaming scanner: bounded sample memory, native geometry, no file deletion."""
import json
from pathlib import Path
import cv2
import numpy as np
from heads import staffs
from layouts import ink, cursor, detect


def write(video, out, layout=None, y_min_fraction=0):
    cv2.setNumThreads(1)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    geometry=detect(video,layout,y_min_fraction)
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0:
        cap.release();raise ValueError('invalid video frame rate')
    records=[];boxes=[];ref=None;samples=[];start=0;n=0;previous_box=None;previous_frame=-999
    top,bottom=geometry['top'],geometry['bottom']

    def finish(end):
        if end-start < max(6,fps*.35) or not samples:
            return
        gray=np.median(np.stack(samples),axis=0).astype(np.uint8)
        found=staffs(gray,min_space=6)
        centers=[(ls[0]+ls[-1])/2 for ls,sp in found]
        for row,(lines,sp) in enumerate(found):
            # A narrow staff crop suppresses lyrics, sticking, NEXT panels and
            # video insets outside the actual staff; mark space above is kept.
            y0=max(0,round(lines[0]-sp*7.25))
            y1=min(len(gray),round(lines[-1]+sp*2.4))
            if row:y0=max(y0,round((centers[row-1]+centers[row])/2))
            if row+1<len(found):y1=min(y1,round((centers[row]+centers[row+1])/2))
            active=[(j,b) for j,b in enumerate(boxes,start) if b and b[1]<=centers[row]<=b[2]]
            sid=len(records)
            cv2.imwrite(str(out/f'system_{sid:03d}.png'),gray[y0:y1],[cv2.IMWRITE_PNG_COMPRESSION,6])
            records.append(dict(system=sid,start=start,end=end,row=row,page_rows=len(found),
                                crop=[y0,y1],native_space=sp,
                                cursor_frames=[j for j,b in active],cursor_x=[b[0] for j,b in active]))

    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            crop=frame[top:bottom];gray=ink(crop)
            box=cursor(crop,float(np.median(geometry['native_spacing'])))
            small=cv2.resize((gray<100).astype(np.float32),(320,max(40,len(gray)//3)),interpolation=cv2.INTER_AREA)
            diff=1 if ref is None else float(np.abs(small-ref).mean())
            reset=bool(box and previous_box and n-previous_frame<fps*.5
                       and previous_box[0]>gray.shape[1]*.55 and box[0]<gray.shape[1]*.35
                       and previous_box[0]-box[0]>gray.shape[1]*.4
                       and abs((box[1]+box[2])-(previous_box[1]+previous_box[2]))<4*np.median(geometry['native_spacing']))
            if diff>.028 or reset:
                finish(n);start=n;samples=[];boxes=[];ref=small
            if n%6==0:
                # Reservoir spread across the whole segment, never a video-sized stack.
                if len(samples)<24:samples.append(gray)
                else:samples[(n//6)%24]=gray
            boxes.append(box)
            if box:previous_box=box;previous_frame=n
            n+=1
        finish(n)
    finally:
        cap.release()
    data=dict(version=3,video=str(Path(video).resolve()),fps=fps,frames=n,
              **geometry,record_cursors=True,records=records,systems=[r['start'] for r in records],cursor=[])
    (out/'scan.json').write_text(json.dumps(data))
    print(f'{out.name}: {n/fps:.1f}s / {len(records)} systems ({geometry["layout"]})',flush=True)
    return data
