"""Light-paper scrolling reconstruction with image-only translation checks.

Periodic/ambiguous matches, changing staff geometry and missing cursors reject
blocks. No detector classes or audio attacks enter the reconstruction.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from heads import staffs
from layouts import ink,cursor,detect,white_staff


def motion_image(gray,left=0,right=None,native_lines=None):
    im=(gray<110).astype(np.uint8)*255
    # Staff lines and lyrics carry no unique horizontal motion evidence.
    found=staffs(gray,min_space=6) if native_lines is None else [(native_lines,0)]
    for ls,sp in found:
        for y in ls:im[max(0,round(y)-1):round(y)+2]=0
    return im[:,left:right]


def motion_pair(previous,current,geometry):
    left=geometry.get('motion_left',0);right=geometry.get('motion_right')
    if geometry.get('foreground')=='white_overlay':
        # Only the overlapping observed score field can establish a shift.
        # Blank margins and the performer's stationary white logos cannot vote.
        bounds=[sample[5] for sample in (previous,current)]
        if not all(bounds):return None
        left=max(left,*(b[0] for b in bounds))
        right=min(right if right is not None else previous[1].shape[1],*(b[1] for b in bounds))
        lines=[sample[4][0][0] for sample in (previous,current)]
    else:lines=[None,None]
    images=[motion_image(sample[3],left,right,ls) for sample,ls in zip((previous,current),lines)]
    return shift(*images)


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
    fractions=(.15,.25,.35,.5,.65,.75,.85) if geometry.get('layout')=='dailychops' else (.25,.5,.75)
    for frac in fractions:
        pair=[]
        for frame_no in (round(count*frac),round(count*frac+fps*.2)):
            cap.set(cv2.CAP_PROP_POS_FRAMES,frame_no);ok,f=cap.read()
            if not ok:continue
            crop=f[geometry['top']:geometry['bottom']];foreground=geometry.get('foreground')
            if foreground=='white_overlay':
                gray=ink(crop,foreground);native,bounds=white_staff(gray)
                pair.append((frame_no,gray,None,ink(crop,foreground,motion=True),native,bounds))
            else:
                pair.append(motion_image(ink(crop,foreground,motion=True),
                                         geometry.get('motion_left',0),geometry.get('motion_right')))
        if len(pair)==2:
            if geometry.get('foreground')=='white_overlay':
                evidence.append(motion_pair(*pair,geometry) if all(len(s[4])==1 for s in pair) else None)
            else:evidence.append(shift(*pair))
    cap.release()
    return sum(bool(e and e['dx'] < -2) for e in evidence)>=2,evidence


def write(video,out,geometry,reject_overlays=False,start_frame=0,end_frame=None):
    cv2.setNumThreads(1)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video));fps=cap.get(5);total=int(cap.get(7))
    if not np.isfinite(fps) or fps<=0:
        cap.release();raise ValueError('invalid video frame rate')
    if start_frame<0 or (end_frame is not None and end_frame<=start_frame):
        cap.release();raise ValueError('invalid native frame interval')
    if start_frame:cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)
    records=[];evidence=[];n=start_frame;block=[];obstructed=0;stride=max(1,round(fps*.2));block_frames=max(1,round(fps*4))

    def failed(sample,end,reason,start=None):
        sid=len(records);image=sample[1]
        cv2.imwrite(str(out/f'system_{sid:03d}.png'),image,[cv2.IMWRITE_PNG_COMPRESSION,6])
        records.append(dict(system=sid,start=sample[0] if start is None else start,end=end,
                            row=0,page_rows=1,crop=[0,len(image)],
                            native_space=float(np.median(geometry['native_spacing'])),
                            cursor_frames=[],cursor_x=[],scroll_reasons=[reason],read_x=None))
        evidence.append(dict(system=sid,start_frame=records[-1]['start'],end_frame=end,matches=[],reasons=[reason]))

    def finish(samples,end):
        if geometry.get('foreground')=='white_overlay' and samples:
            # Never register across absent or changed native geometry. Bound
            # the complete run, rather than allowing gradual vertical drift.
            lines=[];spaces=[]
            for index,sample in enumerate(samples):
                found=sample[4]
                reason=None
                if not found:
                    reason='white-overlay native staff absent or partial lines unconfirmed'
                elif len(found)!=1:
                    reason='white-overlay native staff not unique'
                else:
                    ls,sp=found[0]
                    left,right=sample[5]
                    support=[float((sample[1][round(y),left:right]<230).mean()) for y in ls]
                    if (sp<8 or min(support)<.45 or
                            max(abs(np.diff(ls)-sp))>1.1):
                        reason='white-overlay native staff geometry unconfirmed'
                    elif lines and (np.ptp(spaces+[sp])>1.1 or
                            np.max(np.ptp(np.asarray(lines+[ls]),axis=0))>np.median(spaces+[sp])*.5):
                        reason='white-overlay staff geometry changes across source frames'
                if reason:
                    if index:finish(samples[:index],sample[0])
                    stop=index+1
                    if not found:
                        # Intro/outro blanks are one rejected interval, rather
                        # than one apparent notation system per sampled frame.
                        while stop<len(samples) and not samples[stop][4]:stop+=1
                    failed(sample,samples[stop][0] if stop<len(samples) else end,reason)
                    finish(samples[stop:],end)
                    return
                lines.append(ls);spaces.append(sp)
        if len(samples)<5:
            if samples and geometry.get('split_scroll_discontinuities'):
                failed(samples[0],end,'scroll segment has fewer than five native samples')
            return
        start=samples[0][0];offset=0;offsets=[0];qs=[];reasons=[]
        for index,(previous,current) in enumerate(zip(samples,samples[1:]),1):
            # Brighter white cores supply motion evidence; the less restrictive
            # native grey images remain the source of reconstructed notation.
            delta=motion_pair(previous,current,geometry)
            if delta is None or delta['dx']>2:
                if geometry.get('split_scroll_discontinuities'):
                    # Do not carry a warp across an unverified edge. Keep the
                    # independently registered neighbours and an explicit
                    # rejected gap; every resulting segment still needs five
                    # samples, ink/cursor support and all ordinary gates.
                    finish(samples[:index],previous[0]+1)
                    failed(previous,current[0],
                           'scroll translation ambiguous or rightward discontinuity',start=previous[0]+1)
                    finish(samples[index:],end)
                    return
                reasons.append('scroll translation ambiguous or rightward discontinuity');break
            offset+=delta['dx'];offsets.append(offset);qs.append(delta)
        if obstructed>=2:reasons.append('large chromatic overlay obscures staff')
        sid=len(records)
        evidence.append(dict(system=sid,start_frame=start,end_frame=end,matches=qs,reasons=reasons))
        if reasons:
            image=samples[0][1];sp=0;read_x=None;cx=[];ct=[]
        else:
            seen=[(sample[0],sample[2][0]-dx) for sample,dx in zip(samples,offsets) if sample[2]]
            if len(seen)<len(samples)*.8 or not seen or seen[-1][1]-seen[0][1]<30:
                reasons.append('scroll cursor missing or insufficient traversed span')
                image=samples[0][1];sp=0;read_x=None;cx=[];ct=[]
            else:
                x0=int(seen[0][1]-45);x1=int(seen[-1][1]+45)
                images=[];masks=[]
                for sample,dx in zip(samples,offsets):
                    t,g,b=sample[:3]
                    transform=np.float32([[1,0,-dx-x0],[0,1,0]])
                    images.append(cv2.warpAffine(g,transform,(x1-x0,len(g)),borderValue=255))
                    masks.append(cv2.warpAffine(np.ones_like(g),transform,(x1-x0,len(g)),flags=cv2.INTER_NEAREST,borderValue=0))
                values=np.stack(images).astype(np.float32);values[np.stack(masks)==0]=np.nan
                # Require ink supported in several registered frames. Unlike a
                # union this does not accumulate single-frame compression noise.
                valid=np.sum(np.isfinite(values),axis=0)
                # Grey staff lines are native evidence too. Binarizing at 110
                # erased them, then enlarged diagnostic crops looked like
                # unsupported staffs. Keep supported native intensities, rather
                # than painting grey anti-aliased glyph edges solid black.
                support=np.sum(values<230,axis=0)/np.maximum(1,valid)
                image=np.where((support>=.4)&(valid>=3),
                               np.nanmedian(values,axis=0),255).astype(np.uint8)
                found=staffs(image,min_space=6,prefer_line_span=geometry.get('foreground')=='white_overlay');sp=found[0][1] if len(found)==1 else 0
                if len(found)!=1:reasons.append('scroll reconstructed staff not unique')
                cx=[float(x-x0) for t,x in seen];ct=[t for t,x in seen]
                read_x=[cx[0],cx[-1]-.01]
        cv2.imwrite(str(out/f'system_{sid:03d}.png'),image,[cv2.IMWRITE_PNG_COMPRESSION,6])
        records.append(dict(system=sid,start=start,end=end,row=0,page_rows=1,crop=[0,len(image)],
            native_space=sp,cursor_frames=ct,cursor_x=cx,scroll_reasons=reasons,read_x=read_x))
        if not reasons:
            records[-1]['scroll_x_origin']=x0
            records[-1]['scroll_offsets']=offsets
            records[-1]['sample_frames']=[sample[0] for sample in samples]
            if geometry.get('foreground')=='white_overlay':
                records[-1]['staff_geometry']=dict(
                    native_lines=np.median(lines,axis=0).tolist(),
                    native_spacing=float(np.median(spaces)),
                    supporting_frames=[sample[0] for sample in samples],
                    source_x_bounds=[sample[5] for sample in samples],
                    method='stable unique native five-line staff throughout registered segment')

    try:
        while end_frame is None or n<end_frame:
            ok,f=cap.read()
            if not ok:break
            if n and n%block_frames==0:finish(block,n);block=[];obstructed=0
            if n%stride==0:
                crop=f[geometry['top']:geometry['bottom']]
                foreground=geometry.get('foreground')
                gray=ink(crop,foreground)
                native,bounds=white_staff(gray) if foreground=='white_overlay' else ([],None)
                block.append((n,gray,cursor(crop,foreground=foreground),ink(crop,foreground,motion=True),native,bounds))
                if reject_overlays:
                    from harvest_scan import chromatic_obstructions
                    obstructed+=bool(chromatic_obstructions(crop,float(np.median(geometry['native_spacing']))))
            n+=1
        finish(block,n)
    finally:cap.release()
    data=dict(version=3,video=str(Path(video).resolve()),fps=fps,frames=n,scan_start_frame=start_frame,**geometry,
              record_cursors=True,reconstructed_scroll=True,records=records,
              systems=[r['start'] for r in records],cursor=[])
    (out/'scan.json').write_text(json.dumps(data));(out/'scroll-motion.json').write_text(json.dumps(evidence,indent=2))
    return data
