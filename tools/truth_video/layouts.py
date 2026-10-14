"""Image-derived geometry; identity/notation keys are never guessed from geometry."""
import json
from pathlib import Path
import cv2
import numpy as np
from heads import staffs, heads, staff_checks

HERE = Path(__file__).parent
LAYOUTS = {'kr_strip', 'drumscribe', 'page', 'dailychops', 'olliewoods', 'olliewoods_shorts', 'xiangsong'}


def shorts_paper_top(frame, edge_margins=False):
    """Bottom white page in a native vertical frame, excluding performance ink.

    Sustained near-white paper must extend to the bottom; short staff/beam rows
    are bridged by row support only. This locates a field, not accepted staffs.
    """
    h, w = frame.shape[:2]
    if h <= w:
        raise ValueError('Ollie Woods Shorts requires a native vertical frame')
    white = frame.min(2) > 225
    paper = white.mean(1) >= .7
    if edge_margins:
        # Xiangsong's dense top row can join the performance boundary through
        # 31+ dark rows. Its inspected page has white native outer margins on
        # both sides. These locate paper only; staff/glyph gates are unchanged.
        margin = max(4, round(w/200))
        paper |= (white[:, :margin].mean(1) >= .8) & (white[:, -margin:].mean(1) >= .8)
    bad = np.flatnonzero(~paper)
    runs = np.split(bad, np.where(np.diff(bad)>1)[0]+1)
    # Staff/beam rows interrupt paper for a few pixels. A continuous 31-row
    # non-paper run belongs to performance/background, not native glyph ink.
    long = [run for run in runs if len(run) >= 31]
    top = int(long[-1][-1]+1) if long else 0
    if top < h*.3 or h-top < h*.2:
        raise ValueError('no sustained bottom white Shorts page')
    return top


def detect_shorts(video, layout='olliewoods_shorts'):
    if layout not in ('olliewoods_shorts', 'xiangsong'):
        raise ValueError('unsupported native vertical page profile')
    cap = cv2.VideoCapture(str(video)); observations=[]; tops=[]; sizes=[]; absent=[]
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fraction in (.03, .1, .25, .5, .75, .9, .95):
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(count*fraction))
            ok, frame = cap.read()
            if not ok: continue
            try:
                top = shorts_paper_top(frame, edge_margins=layout == 'xiangsong')
            except ValueError as e:
                absent.append(dict(fraction=fraction, reason=str(e)))
                continue  # Score-free/faded end cards do not certify geometry.
            found = staffs(ink(frame[top:]), min_space=6, prefer_line_span=True)
            if len(found) < 2:
                raise ValueError('Shorts page needs at least two observed native staffs')
            tops.append(top); sizes.append(frame.shape[:2]); observations.extend(found)
    finally:
        cap.release()
    if len(tops) < 3 or len(set(sizes)) != 1 or np.ptp(tops) > 4:
        raise ValueError('unstable native Shorts page boundary')
    h, w = sizes[0]
    return dict(layout=layout, image_layout='native vertical bottom white page',
                left=0, top=min(tops), right=w, bottom=h,
                paper_tops=tops, absent_samples=absent,
                native_spacing=[round(sp,3) for _,sp in observations],
                prefer_staff_line_span=True,
                cursor_style='blue_box' if layout == 'xiangsong' else 'orange_line',
                identity_source='channel-specific native vertical page survey')


def detect_region(video, profile):
    """Native score region explicitly inspected for an opt-in channel profile.

    Fractions crop source pixels without resizing. Detection never certifies
    labels, clefs or timing; all ordinary per-system checks still follow.
    Late-only lesson overlays are sampled through 95 percent of the source.
    """
    region = profile['score_region']
    if (len(region) != 4 or not all(0 <= v <= 1 for v in region)
            or region[0] >= region[2] or region[1] >= region[3]):
        raise ValueError('invalid inspected score region')
    foreground = profile.get('foreground')
    variants = profile.get('score_variants')
    if variants:
        validate_score_variants(variants)
    cap = cv2.VideoCapture(str(video)); observations = []; survey=[]
    h=w=0
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for fraction in (.03, .1, .25, .5, .75, .9, .95):
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(count*fraction))
            ok, frame = cap.read()
            if not ok: continue
            h, w = frame.shape[:2]
            left, top, right, bottom = [round(v*d) for v,d in zip(region,(w,h,w,h))]
            crop=frame[top:bottom,left:right]
            if variants:
                variant, found, reason = select_score_variant(crop, variants,
                    profile.get('prefer_staff_line_span',False))
                survey.append(dict(frame=round(count*fraction),variant=variant,reason=reason))
            else:
                found=staffs(ink(crop,foreground),min_space=6,
                    prefer_line_span=profile.get('prefer_staff_line_span',False))
            observations.extend(found)
    finally:
        cap.release()
    if not h or not w:
        raise ValueError('no decodable native video frame')
    result = dict(layout=profile['layout'], image_layout='inspected native strip',
                left=left, top=top, right=right, bottom=bottom,
                foreground=foreground, cursor_style=profile.get('cursor_style','chromatic'),
                # An advisory cursor scale, never evidence of a detected staff.
                # Local segments still search native lines and reject absence.
                native_spacing=[round(sp,3) for ls,sp in observations] or [11.],
                native_spacing_observed=bool(observations),
                prefer_staff_line_span=profile.get('prefer_staff_line_span',False),
                identity_source='channel-specific offline frame survey')
    if not observations:
        result['survey_reasons']=['no native five-line staff in inspected score region']
    if variants:
        result['score_variants']=variants
        result['variant_survey']=survey
    return result


def validate_score_variants(variants):
    if not isinstance(variants,list) or not variants:
        raise ValueError('invalid inspected score variants')
    for v in variants:
        if not isinstance(v,dict) or v.get('foreground') not in ('white_strip','dark_strip'):
            raise ValueError('unsupported inspected score variant')
        if v.get('cursor_style') not in ('chromatic','red_block','dim_line','green_line','gray_box','none'):
            raise ValueError('unsupported inspected cursor style')
    if len({v['foreground'] for v in variants})!=len(variants):
        raise ValueError('duplicate inspected score foreground')


def select_score_variant(frame, variants, prefer_line_span=False):
    """Select only an explicitly inspected polarity with supported native ink.

    Size/clef/labels/timing are checked downstream. Neither the survey nor this
    selection grants continuation proof. Competing polarities reject locally.
    """
    candidates=[]
    native_paper=frame.min(2)
    for index,v in enumerate(variants):
        gray=ink(frame,v['foreground'])
        found=staffs(gray,min_space=6,prefer_line_span=prefer_line_span)
        supported=[]
        for ls,sp in found:
            support=[float((gray[round(y)]<230).mean()) for y in ls]
            y0=max(0,round(ls[0]-sp*4));y1=min(len(gray),round(ls[-1]+sp*2.4))
            # Normalizing dark performance pixels can produce contour dips.
            # The actual native field must have the inspected paper polarity;
            # transformed brightness alone cannot establish light paper.
            light_paper=float(np.median(native_paper[y0:y1]))>105
            if light_paper != (v['foreground']=='dark_strip'):continue
            if (min(support)>=.45 and max(abs(np.diff(ls)-sp))<=1.1 and
                    (gray[y0:y1]>180).mean()>.55):
                supported.append((ls,sp))
        if supported:candidates.append((index,supported))
    if len(candidates)==1:
        index,found=candidates[0]
        return index,found,None
    return None,[],('ambiguous native score polarity' if candidates else
                    'no native five-line staff in inspected score region')


def region_cursor(frame, space, style):
    if style == 'none': return None
    if style == 'chromatic': return cursor(frame, space)
    if style == 'green_line':
        # Baquetazo's yellow paper shifts its pale green cursor into the
        # generic yellow hue exclusion. Brown note/stem edges cannot supply
        # green excess, and only a unique tall narrow column can qualify.
        b,g,r=[c.astype(np.int16) for c in cv2.split(frame)]
        background=(cv2.medianBlur(frame[:,:,1],31).astype(np.int16)-
                    cv2.medianBlur(frame[:,:,2],31).astype(np.int16))
        mask=((g-r-background>8)&(g-r>8)&(g-b>8)).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,
                             np.ones((max(5,round(space)),1),np.uint8))
        # The inspected ROI also includes a black border/performance margin.
        # Apply the same tall-column requirement to sustained native paper,
        # rather than counting those margins as missing cursor support.
        paper=(np.median(frame.min(2),axis=1)>105).astype(np.uint8)[:,None]
        paper=cv2.morphologyEx(paper,cv2.MORPH_CLOSE,
                              np.ones((max(3,round(space)),1),np.uint8))[:,0]
        yy=np.flatnonzero(paper)
        fields=[q for q in np.split(yy,np.where(np.diff(yy)>1)[0]+1) if len(q)>=space*8]
        if len(fields)!=1:return None
        field=fields[0];mask[:field[0]]=0;mask[field[-1]+1:]=0
        height=max(space*8,len(field)*.7)
        support=mask.sum(0)
        xs=np.flatnonzero(support>=height*.6)
        runs=np.split(xs,np.where(np.diff(xs)>1)[0]+1);candidates=[]
        for run in runs:
            if not 1<=len(run)<=max(4,space*.55):continue
            ys=np.flatnonzero(mask[:,run].mean(1)>=.6)
            if not len(ys):continue
            span=ys[-1]-ys[0]+1
            if span>=height and len(ys)>=span*.6:
                core=run[support[run]>=support[run].max()*.7]
                candidates.append([float(core.mean()),int(ys[0]),int(ys[-1]+1),len(run)])
        return candidates[0] if len(candidates)==1 else None
    if style == 'blue_box':
        # An opt-in translucent blue rectangle, not blue note/stem ink alone.
        # Require a unique tall, filled plate in the observed page field.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 0] >= 85) & (hsv[:, :, 0] <= 120) &
                (hsv[:, :, 1] > 27) & (hsv[:, :, 2] > 120)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                               np.ones((max(5, round(space)), 1), np.uint8))
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        candidates = [s for s in stats[1:] if s[3] >= space*5 and
                      space*.5 <= s[2] <= space*6 and s[4] >= s[2]*s[3]*.55]
        if len(candidates) != 1: return None
        x, y, w, h, _ = candidates[0]
        return [float(x+(w-1)/2), int(y), int(y+h), int(w)]
    if style == 'orange_line':
        # In Shorts the note and sticking letter turn orange with the line.
        # Their connected ink must not widen or move its centre. Sustained
        # narrow columns locate only the printed red/orange vertical playhead.
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # Kathe's translucent warm-grey line is much less saturated than
        # Weckl's orange bar; sustained geometry supplies the discrimination.
        warm=(hsv[:,:,0]<22)|(hsv[:,:,0]>170)
        mask=(warm&(hsv[:,:,1]>20)&(hsv[:,:,2]>80)).astype(np.uint8)
        support=mask.sum(0)
        if not support.max(): return None
        columns=(support>=max(space*3,support.max()*.3)).astype(np.uint8)[None,:]
        columns=cv2.morphologyEx(columns,cv2.MORPH_CLOSE,np.ones((1,3),np.uint8))[0]
        xs=np.flatnonzero(columns);runs=np.split(xs,np.where(np.diff(xs)>1)[0]+1)
        candidates=[]
        for run in runs:
            if not 1<=len(run)<=max(3,space*.8): continue
            ys=np.flatnonzero(mask[:,run].mean(1)>=.45)
            if not len(ys): continue
            height=ys[-1]-ys[0]+1
            if height>=space*6 and len(ys)>=height*.5:
                core=run[support[run]>=support[run].max()*.7]
                candidates.append([float(core.mean()),int(ys[0]),int(ys[-1]+1),len(run)])
        return candidates[0] if len(candidates)==1 else None
    if style in ('red_block','dim_line'):
        if style == 'red_block':
            hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            mask=(((hsv[:,:,0]<10)|(hsv[:,:,0]>170)) &
                  (hsv[:,:,1]>100)&(hsv[:,:,2]>45)).astype(np.uint8)
            max_width=space*4
        else:
            # A thin translucent green/white line on dark purple paper may
            # never reach the ordinary chromatic cursor's 120 value floor.
            # Require long narrow local contrast in the inspected native ROI.
            lo=frame.min(2)
            background=cv2.medianBlur(lo,31)
            mask=(lo.astype(np.int16)-background>18).astype(np.uint8)
            horizontal=cv2.morphologyEx(mask,cv2.MORPH_OPEN,
                                        np.ones((1,max(30,round(space*6))),np.uint8))
            mask &= 1-horizontal
            max_width=max(4,space*.55)
            # Connected components merge the thin line with note stems at
            # crossings. Require sustained column support instead, with the
            # same unique, narrow, tall source-line evidence.
            columns=np.flatnonzero(mask.mean(0)>=.7)
            runs=np.split(columns,np.where(np.diff(columns)>1)[0]+1)
            candidates=[]
            for run in runs:
                if not 1<=len(run)<=max_width:continue
                ys=np.flatnonzero(mask[:,run].mean(1)>=.6)
                if len(ys) and ys[-1]-ys[0]+1>=max(space*8,len(frame)*.7):
                    candidates.append([float(run.mean()),int(ys[0]),int(ys[-1]+1),len(run)])
            return candidates[0] if len(candidates)==1 else None
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((max(5,round(space)),1),np.uint8))
        _,_,stats,_=cv2.connectedComponentsWithStats(mask)
        candidates=[s for s in stats[1:] if s[3]>=max(space*8,len(frame)*.7) and
                    1<=s[2]<=max_width and s[4]>=s[2]*s[3]*.6]
        if len(candidates)!=1:return None
        x,y,w,h,_=candidates[0]
        return [float(x+(w-1)/2),int(y),int(y+h),int(w)]
    if style != 'gray_box': raise ValueError('unsupported inspected cursor style')
    # Opt-in Gwon light strips have an achromatic gray rectangular playhead.
    # Ordinary line/head pixels are dark, paper is white. The tall solid box
    # must span at least five staff spaces; sticking/lyrics cannot qualify.
    # OpenCV's native channel operations are equivalent to the NumPy reductions
    # and keep full-video scans bounded on CPU.
    b,g,r = cv2.split(frame)
    lo, hi = cv2.min(cv2.min(b,g),r), cv2.max(cv2.max(b,g),r)
    mask = ((hi.astype(np.int16)-lo < 20) & (lo > 90) & (hi < 225)).astype(np.uint8)
    mask = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((max(5,round(space)),1),np.uint8))
    # Anti-aliased gray staff pixels connect the real box to all five lines.
    # Find sustained gray columns, then inspect their vertical rectangle,
    # rather than allowing the whole connected staff to determine its width.
    # Express support in native staff spaces, matching the original box's
    # minimum height; empty margins must not change cursor qualification.
    columns=np.flatnonzero(mask.sum(0)>=space*5*.65)
    runs=np.split(columns,np.where(np.diff(columns)>1)[0]+1)
    candidates=[]
    for run in runs:
        if not space*.5<=len(run)<=space*6:continue
        support=mask[:,run].sum(0)
        strong=(support>=max(space*5*.65,support.max()*.7)).astype(np.uint8)[None,:]
        # Black note stems cut narrow holes inside the gray plate. Bridge only
        # enclosed holes below one native staff space, preserving its edges.
        strong=cv2.morphologyEx(strong,cv2.MORPH_CLOSE,
                               np.ones((1,max(3,round(space*.8))),np.uint8))[0]
        core=run[strong.astype(bool)]
        cores=np.split(core,np.where(np.diff(core)>1)[0]+1)
        cores=[q for q in cores if space*.5<=len(q)<=space*6]
        if len(cores)!=1:continue
        run=cores[0]
        ys=np.flatnonzero(mask[:,run].mean(1)>=.65)
        chunks=np.split(ys,np.where(np.diff(ys)>space*2)[0]+1)
        chunks=[q for q in chunks if len(q) and q[-1]-q[0]+1>=space*5
                and len(q)>=(q[-1]-q[0]+1)*.65]
        if len(chunks)==1:
            ys=chunks[0]
            candidates.append([float(run.mean()),int(ys[0]),int(ys[-1]+1),len(run)])
    return candidates[0] if len(candidates)==1 else None


def channel_profile(channel=None):
    # Manifest strings must never become arbitrary file paths.
    paths = {p.stem: p for p in (HERE/'legends').glob('*.json')}
    p = paths.get(channel)
    if p:
        data = json.loads(p.read_text())
        if data.get('harvest_profile'):
            return data, p
    return dict(name='unverified', enabled=False, rules={}, min_x_fraction=.04,
                harvest_profile=True, reason='unknown channel; no verified legend'), None


def ink(frame, foreground=None, motion=False):
    """Preserve dark colored played heads, whiten pale colored cursor backgrounds."""
    b,g,r=cv2.split(frame)
    lo,hi=cv2.min(cv2.min(b,g),r),cv2.max(cv2.max(b,g),r)
    if foreground == 'dark_strip':
        # Inspected black notation on white, yellow or textured light paper.
        # Normalize slow native paper shading; black glyphs remain black and
        # grey lines retain their contrast. This never resizes or paints ink.
        gray=np.where((hi>170)&(lo>105),hi,lo).astype(np.uint8)
        paper=cv2.medianBlur(gray,51)
        return np.minimum(255,gray.astype(np.float32)*255/np.maximum(1,paper)).astype(np.uint8)
    if foreground in ('white_overlay', 'white_strip'):
        # An explicitly selected white-on-video score. Preserve native white
        # glyph intensity; saturated performance pixels and the red playhead
        # are background. This is not inversion of the performance frame.
        # Dedicated dark-paper strips have gray staff lines darker than the
        # DailyChops overlay. Preserve their actual intensity, bounded by the
        # inspected score ROI; never apply this to arbitrary performance video.
        threshold = 55 if foreground == 'white_strip' else 200 if motion else 145
        white = (lo > threshold) & (hi.astype(np.int16)-lo < 45)
        return np.where(white, 255-lo, 255).astype(np.uint8)
    if foreground is not None:
        raise ValueError(f'unsupported notation foreground: {foreground}')
    return np.where((hi > 170) & (lo > 105), hi, lo).astype(np.uint8)


def white_staff(gray):
    """Find native white staff ink even when only part of the viewport remains.

    The ordinary median-row search needs lines across half the image. A score
    entering/leaving a white overlay can occupy less than that. Search native
    half-width windows, deduplicate agreeing geometry, then require the same
    unique staff in its sustained five-line span. No pixels are resized or
    painted, and the caller still applies every support/spacing/run gate.
    """
    found=staffs(gray,prefer_line_span=True)
    if not found:
        width=gray.shape[1];window=max(1,width//2)
        candidates=[]
        for left in (0,width//4,width-window):
            for ls,sp in staffs(gray[:,left:left+window],prefer_line_span=True):
                if not any(max(abs(np.asarray(ls)-other))<=1.1 for other,_ in candidates):
                    candidates.append((ls,sp))
        found=sorted(candidates)
    if len(found)!=1:return found,None
    ls,sp=found[0]
    common=(gray[np.rint(ls).astype(int)]<230).mean(0)>=.8
    columns=np.flatnonzero(common)
    runs=[r for r in np.split(columns,np.where(np.diff(columns)>2)[0]+1)
          if len(r)>=sp*8]
    if not runs:return [],None
    left=max(0,int(runs[0][0]-sp));right=min(gray.shape[1],int(runs[-1][-1]+sp)+1)
    local=staffs(gray[:,left:right],prefer_line_span=True)
    if len(local)!=1 or max(abs(np.asarray(local[0][0])-ls))>1.1:
        return [],None
    return local,[left,right]


def detect_white_overlay(video):
    """Select the lowest sustained white staff, excluding the upper chord row.

    This opt-in DailyChops variant has a native drum staff below 75% of the
    frame. Spread-out frames bound the search region; geometry changes are
    checked locally by the scrolling reader and split into separate segments.
    Registration, reading and ordinary staff/timing gates still follow.
    """
    cap = cv2.VideoCapture(str(video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); observations = []
    try:
        for fraction in (.1, .25, .5, .75):
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(count*fraction))
            ok, frame = cap.read()
            if not ok: continue
            h, w = frame.shape[:2]
            gray = ink(frame, 'white_overlay')
            # The native bottom ROI excludes upper notation/video rows before
            # looking for partial spans; full-frame windowing is not evidence.
            bottom=round(h*.75)
            found = [(np.asarray(ls)+bottom, sp) for ls, sp in white_staff(gray[bottom:])[0]]
            if found: observations.append(max(found, key=lambda pair: pair[0][-1]))
    finally:
        cap.release()
    if len(observations) < 3:
        raise ValueError('no stable bottom white-overlay drum staff')
    positions = np.asarray([ls for ls, sp in observations])
    # This bounds the search crop, not accepted staff geometry. Eric's rendered
    # score moves vertically by three native pixels between scenes; each saved
    # block still has to pass the ordinary 1.1 px spacing-residual check.
    lines = np.median(positions, axis=0); sp = np.median([sp for ls, sp in observations])
    result=dict(layout='dailychops', image_layout='white bottom scrolling drum staff',
                foreground='white_overlay', left=0, right=w,
                top=max(0, int(positions[:,0].min()-sp*4)),
                bottom=min(h, int(positions[:,-1].max()+sp*.8)+1),
                native_spacing=[round(s, 3) for ls, s in observations],
                motion_left=round(sp*10), motion_right=w-round(sp*4),
                identity_source='opt-in white overlay; lowest sustained native staff')
    result['sampled_native_lines']=positions.tolist()
    # A continuation viewport loses its clef through scrolling. Establish the
    # actual printed percussion clef in this video's opening, rather than
    # granting a channel-wide absence waiver from geometry alone.
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    try:
        for seconds in range(4,21):
            cap.set(cv2.CAP_PROP_POS_FRAMES,round(fps*seconds));ok,frame=cap.read()
            if not ok:continue
            gray=ink(frame[result['top']:result['bottom']],'white_overlay')
            candidates=[(ls,s) for ls,s in staffs(gray,prefer_line_span=True)
                        if ls[0]+result['top']>=frame.shape[0]*.75]
            if len(candidates)!=1:continue
            ls,s=candidates[0]
            common=(gray[np.rint(ls).astype(int)]<230).mean(0)>=.8
            cols=np.flatnonzero(common)
            runs=[r for r in np.split(cols,np.where(np.diff(cols)>2)[0]+1) if len(r)>=s*8]
            if not runs:continue
            # The fixed red cursor splits the common-line run. Keep the
            # actual left header rather than selecting the longer right run.
            run=min(runs,key=lambda r:r[0]);x0=max(0,round(run[0]-s*3))
            x1=min(w,round(max(r[-1] for r in runs)+s)+1)
            crop=gray[:,x0:x1]
            try:notes,actual,spacing=heads(crop,prefer_line_span=True)
            except ValueError:continue
            if max(abs(np.asarray(actual)-ls))>1.1:continue
            checks=staff_checks(crop,actual,spacing,[{**n,'inst':'unknown'} for n in notes],
                                strict_clef=True,clef_origin=0)
            if checks['clef_confirmed'] and not checks['reasons']:
                result['opening_clef']=dict(frame=round(fps*seconds),seconds=seconds,
                    crop=[x0,result['top'],x1,result['bottom']],
                    native_lines=[y+result['top'] for y in actual],native_spacing=spacing,
                    method='native printed opening clef; ordinary strict paired-stroke staff checks')
                break
    finally:cap.release()
    # Opening proof is useful evidence, not a video-wide prerequisite. A
    # successfully registered stable native run supplies local continuation
    # evidence; missing/changed geometry still rejects its own segment.
    return result


def cursor(frame, space=11, foreground=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if foreground == 'white_overlay':
        # These two inspected scrolling scores have one fixed red column at
        # the centre. Coloured drums/lights cannot become moving playheads.
        red=((hsv[:,:,0]<10)|(hsv[:,:,0]>170))&(hsv[:,:,1]>120)&(hsv[:,:,2]>120)
        _,_,stats,_=cv2.connectedComponentsWithStats(red.astype(np.uint8))
        candidates=[s for s in stats[1:] if s[3]>=len(frame)*.8 and
                    1<=s[2]<=min(90,space*7) and s[4]>=s[3]*s[2]*.7 and
                    abs(s[0]+(s[2]-1)/2-frame.shape[1]/2)<=space*2]
        if len(candidates)!=1:return None
        x,y,w,h,_=candidates[0]
        return [float(x+(w-1)/2),int(y),int(y+h),int(w)]
    if foreground is not None:raise ValueError(f'unsupported cursor foreground: {foreground}')
    # Yellow bar shading is deliberately not a playhead. Blue, green, pink,
    # lavender and orange vertical cursors are accepted on geometric evidence.
    mask = ((hsv[:, :, 1] > 27) & (hsv[:, :, 2] > 120) &
            ~((hsv[:, :, 0] >= 22) & (hsv[:, :, 0] <= 38))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((max(5, round(space)), 1), np.uint8))
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = [s for s in stats[1:] if s[3] >= max(18, space*2.5)
                  and 1 <= s[2] <= min(90, space*7) and s[4] >= s[3]*.6]
    if not candidates:
        return None
    x,y,w,h,_ = max(candidates, key=lambda s: s[3]/max(1, s[2])**.2)
    return [float(x+(w-1)/2), int(y), int(y+h), int(w)]


def detect(video, requested=None, y_min_fraction=0):
    if requested in ('olliewoods_shorts', 'xiangsong'):
        return detect_shorts(video, requested)
    cap = cv2.VideoCapture(str(video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    observations = []; paper_tops=[]; motion_starts=[]; staff_spans=[]
    h=w=0
    for fraction in (.1, .25, .5, .75):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(count*fraction))
        ok, frame = cap.read()
        if not ok:
            continue
        h,w=frame.shape[:2]
        gray = ink(frame)
        for lines, sp in staffs(gray, min_space=6):
            if lines[0] < h*y_min_fraction:
                continue
            # The validated light DailyChops variant places one staff above
            # the performance. Horizontal edges in cymbals/clothing are not
            # additional page rows. Dark bottom overlays need their own reader.
            if requested == 'dailychops' and lines[-1] > h*.3:
                continue
            y0=max(0,round(lines[0]-sp*4));y1=min(h,round(lines[-1]+sp*3))
            # A light field is required; inverted score/video overlays need a
            # separately validated foreground extractor, not global inversion.
            if (gray[y0:y1] > 180).mean() > .55:
                observations.append((lines,sp))
                if requested == 'dailychops':
                    common=(gray[np.round(lines).astype(int)]<230).mean(0)>=.8
                    cols=np.flatnonzero(common)
                    runs=np.split(cols,np.where(np.diff(cols)>2)[0]+1)
                    runs=[r for r in runs if len(r)>=sp*8]
                    if runs:
                        # Skip the fixed left-edge fade, and the performer's
                        # moving hair outside the actual notation field.
                        run=max(runs,key=len)
                        motion_starts.append(int(run[0]+sp*2))
                        staff_spans.append([max(0,int(run[0]-sp)),min(w,int(run[-1]+sp)+1)])
                desired=max(0,int(lines[0]-sp*7.25))
                end=max(desired,int(lines[0]-sp*3.7))
                window=max(7,round(sp))
                light=np.convolve((gray>180).mean(1),np.ones(window)/window,mode='same')
                bad=np.flatnonzero(light[desired:end]<.6)+desired
                paper_tops.append(max(desired,int(bad[-1])+1) if len(bad) else desired)
    cap.release()
    if not observations:
        raise ValueError('no stable light five-line staff; count-in/inverted/occluded notation or missing score')
    low=min(paper_tops)
    high=max(ls[-1]+sp*3 for ls,sp in observations)
    mids=np.array([(ls[0]+ls[-1])/2 for ls,sp in observations])
    inferred='page' if np.ptp(mids)>h*.3 else 'dailychops' if np.median(mids)<h*.5 else 'kr_strip'
    layout=requested if requested in LAYOUTS else inferred
    # Manifest is a hint; broad page/top searches remain driven by observed ink.
    result=dict(layout=layout, image_layout=inferred, left=0, right=w,
                top=max(0,int(low)), bottom=min(h,int(high)+1),
                native_spacing=[round(sp,3) for _,sp in observations],
                identity_source='manifest' if requested in LAYOUTS else 'image geometry only')
    if requested == 'dailychops' and motion_starts:
        result['motion_left']=max(motion_starts)
        result['staff_x_bounds']=[min(a for a,b in staff_spans),max(b for a,b in staff_spans)]
    return result
