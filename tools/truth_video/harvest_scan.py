"""Streaming scanner: bounded sample memory, native geometry, no file deletion."""
import json
from pathlib import Path
import cv2
import numpy as np
from heads import heads, staffs
from layouts import ink, cursor, detect


def chromatic_obstructions(frame, space):
    """Large opaque colour ink (e.g. promotional lettering), never a playhead."""
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=((hsv[:,:,1]>100)&(hsv[:,:,2]>80)).astype(np.uint8)
    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
    return [[int(x),int(y),int(w),int(h)] for x,y,w,h,area in stats[1:]
            if w>space*2.5 and h>space*3.5 and area>space*space*4]


def staff_crops(gray, found):
    """Use the same native staff bounds for consensus and the saved reader crop."""
    centers=[(ls[0]+ls[-1])/2 for ls,sp in found]
    for row,(lines,sp) in enumerate(found):
        y0=max(0,round(lines[0]-sp*7.25))
        y1=min(len(gray),round(lines[-1]+sp*2.4))
        if row:y0=max(y0,round((centers[row-1]+centers[row])/2))
        if row+1<len(found):y1=min(y1,round((centers[row]+centers[row+1])/2))
        yield row,y0,y1


def stable_native_staff(samples, frames):
    """Recover a failed median from agreeing native frames, never upscaling.

    Raster plateaus can move a dip by one pixel in the median and defeat the
    integer staff search, despite consistent subpixel geometry in source frames.
    Require six source frames, >=75% geometry support and stable notation ink.
    Every selected head must also be redetected in a strict majority of the
    geometrically agreeing frames, at the same shape, staff position and native
    location. An unsupported head disqualifies that source frame; try another
    agreeing frame or reject the fallback. The ordinary gates still follow.
    """
    if len(samples)<6:return None
    geometries=[staffs(g,min_space=6) for g in samples];detections={}
    for index,reference in enumerate(geometries):
        if not reference:continue
        supporting=[i for i,g in enumerate(geometries) if len(g)==len(reference) and
                    max(abs(np.asarray([ls for ls,sp in g])-np.asarray([ls for ls,sp in reference])).flat)<=1.1]
        if len(supporting)<6 or len(supporting)<len(samples)*.75:continue
        midpoint=len(supporting)//2
        for selected in supporting[midpoint:]+supporting[:midpoint]:
            gray=samples[selected]
            if any(((gray<100)!=(g<100)).mean()>.028 for g in samples):continue
            required=len(supporting)//2+1;head_support=[];consistent=True
            for row,y0,y1 in staff_crops(gray,geometries[selected]):
                space=geometries[selected][row][1]
                for i in supporting:
                    key=(i,y0,y1)
                    if key not in detections:
                        try:detections[key]=heads(samples[i][y0:y1])[0]
                        except ValueError:detections[key]=[]
                for h in detections[selected,y0,y1]:
                    matches=[int(frames[i]) for i in supporting if any(
                        n['shape']==h['shape'] and n['pos']==h['pos'] and
                        abs(n['x']-h['x'])<=space*.35 and abs(n['y']-h['y'])<=space*.35
                        for n in detections[i,y0,y1])]
                    if len(matches)<required:
                        consistent=False;break
                    head_support.append(dict(row=row,x=h['x'],y=h['y']+y0,
                                             shape=h['shape'],pos=h['pos'],supporting_frames=matches))
                if not consistent:break
            if not consistent or not head_support:continue
            return gray,geometries[selected],dict(selected_frame=int(frames[selected]),
                supporting_frames=[int(frames[i]) for i in supporting],sample_count=len(samples),
                head_support_required=required,head_support=head_support,
                method='native-frame geometry consensus after failed median; stable notation ink; per-head majority')
    return None


def write(video, out, layout=None, y_min_fraction=0, reject_overlays=False, native_fallback=False):
    cv2.setNumThreads(1)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    geometry=detect(video,layout,y_min_fraction)
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps<=0:
        cap.release();raise ValueError('invalid video frame rate')
    records=[];boxes=[];ref=None;samples=[];sample_frames=[];obstructions=[];start=0;n=0;previous_box=None;previous_frame=-999
    top,bottom=geometry['top'],geometry['bottom']

    def finish(end):
        if end-start < max(6,fps*.35) or not samples:
            return
        gray=np.median(np.stack(samples),axis=0).astype(np.uint8)
        found=staffs(gray,min_space=6)
        fallback=None
        if not found and native_fallback:
            recovered=stable_native_staff(samples,sample_frames)
            if recovered:
                cv2.imwrite(str(out/f'median_failed_{start:06d}.png'),gray)
                gray,found,fallback=recovered
        centers=[(ls[0]+ls[-1])/2 for ls,sp in found]
        for row,y0,y1 in staff_crops(gray,found):
            lines,sp=found[row]
            # A narrow staff crop suppresses lyrics, sticking, NEXT panels and
            # video insets outside the actual staff; mark space above is kept.
            active=[(j,b) for j,b in enumerate(boxes,start) if b and b[1]<=centers[row]<=b[2]]
            sid=len(records)
            cv2.imwrite(str(out/f'system_{sid:03d}.png'),gray[y0:y1],[cv2.IMWRITE_PNG_COMPRESSION,6])
            records.append(dict(system=sid,start=start,end=end,row=row,page_rows=len(found),
                                crop=[y0,y1],native_space=sp,
                                cursor_frames=[j for j,b in active],cursor_x=[b[0] for j,b in active]))
            if fallback:records[-1]['native_frame_fallback']=fallback
            if reject_overlays:
                contaminated=sum(any(y+h>lines[0]-sp*2 and y<lines[-1]+sp*2
                                     for x,y,w,h in frame_boxes) for frame_boxes in obstructions)
                records[-1]['obstruction_samples']=contaminated
                if contaminated>=2:
                    records[-1]['reading_reasons']=['large chromatic overlay obscures staff']

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
                finish(n);start=n;samples=[];sample_frames=[];obstructions=[];boxes=[];ref=small
            if n%6==0:
                # Reservoir spread across the whole segment, never a video-sized stack.
                obstruction=chromatic_obstructions(crop,float(np.median(geometry['native_spacing']))) if reject_overlays else []
                if len(samples)<24:
                    samples.append(gray);sample_frames.append(n);obstructions.append(obstruction)
                else:
                    index=(n//6)%24;samples[index]=gray;sample_frames[index]=n;obstructions[index]=obstruction
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
