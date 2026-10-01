"""Read shapes separately from instrument names so a layout's legend stays inspectable."""
from pathlib import Path
import cv2
import numpy as np
from scipy.signal import find_peaks


def staffs(gray, min_space=6):
    profile = np.median(gray, axis=1).astype(float)
    background = np.percentile(profile, 90)
    dips, _ = find_peaks(background-profile, height=6, distance=2)
    candidates = []
    for a in dips:
        for b in dips[dips > a]:
            space = (b-a)/4
            if not min_space <= space <= 30:
                continue
            expected = a+np.arange(5)*space
            rows = [int(dips[np.argmin(abs(dips-y))]) for y in expected]
            if max(abs(np.array(rows)-expected)) > max(.8, space*.1):
                continue
            strength = np.mean(background-profile[rows])
            candidates.append((strength, rows, space))
    kept = []
    for strength, rows, space in sorted(candidates, reverse=True):
        if any(not (rows[-1] < ls[0]-sp or rows[0] > ls[-1]+sp) for ls,sp in kept):
            continue
        kept.append((list(map(float, rows)), float((rows[-1]-rows[0])/4)))
    refined=[]
    for lines,space in kept:
        # find_peaks chooses an integer edge on even-width black plateaus.
        # Measure their ink centroid before testing spacing, rather than calling
        # a 2-pixel/3-pixel rasterization alternation an uneven musical staff.
        centers=[]
        for y in lines:
            yy=np.arange(max(0,int(y)-2),min(len(profile),int(y)+3))
            weight=np.maximum(background-profile[yy],0)
            centers.append(float(np.average(yy,weights=weight)) if weight.sum() else y)
        refined.append((centers,float((centers[-1]-centers[0])/4)))
    return sorted(refined)


def staff_lines(gray):
    found = staffs(gray)
    if len(found) != 1:
        raise ValueError(f'expected one staff, found {len(found)}')
    return found[0]


def staff_checks(gray, lines, space, notes, strict_clef=False):
    dark = gray < 150
    ys = np.array(lines).astype(int)
    support = [float((gray[y] < 230).mean()) for y in ys]
    bars = np.where(dark[ys[0]:ys[-1]+1].mean(axis=0) > .88)[0]
    runs = np.split(bars, np.where(np.diff(bars)>2)[0]+1)
    bars = [float(np.mean(r)) for r in runs if len(r)]
    middle = dark[ys[1]+1:ys[3]].mean(axis=0)
    clef_cols = np.where((middle > .85) & (np.arange(gray.shape[1]) < gray.shape[1]*.12))[0]
    clef_runs = np.split(clef_cols, np.where(np.diff(clef_cols)>1)[0]+1)
    clef = [float(np.mean(r)) for r in clef_runs if len(r)>=max(2,round(space*.2))]
    # A score may be inset farther than 12% of the frame (Ollie Woods).
    # Audited opt-in profiles use sustained five-line ink before the first
    # note. Legacy profiles retain their original check exactly.
    if strict_clef:
        clef=[]
        common=(gray[ys] < 230).mean(axis=0)>=.8
        columns=np.flatnonzero(common)
        spans=np.split(columns,np.where(np.diff(columns)>2)[0]+1)
        starts=[int(r[0]) for r in spans if len(r)>=space*8]
        if starts:
            start=starts[0]
            # Include heads immediately after the staff start. Skipping the
            # first two spaces lets their stems masquerade as a clef pair.
            first=min((n['x']-max(space*.7,n.get('w',0)/2) for n in notes
                       if n['x']>=start-space),default=gray.shape[1])
            end=min(start+space*4,start+gray.shape[1]*.12,first)
            # Up/down stems attach near either head edge. Exclude that full
            # column neighbourhood, including stems left of a head's body.
            note_columns=np.zeros(gray.shape[1],dtype=bool)
            for n in notes:
                radius=max(space,n.get('w',0)/2)
                note_columns[max(0,int(np.floor(n['x']-radius))):
                             min(gray.shape[1],int(np.ceil(n['x']+radius))+1)]=True
            cols=np.flatnonzero((middle>.85)&(np.arange(gray.shape[1])>=start)&
                                (np.arange(gray.shape[1])<end)&~note_columns)
            runs=np.split(cols,np.where(np.diff(cols)>1)[0]+1)
            strokes=[r for r in runs if max(2,round(space*.2))<=len(r)<=round(space*.75)
                     and dark[ys[0]:ys[-1]+1,r].mean()<.85]
            pairs=[(a,b) for a,b in zip(strokes,strokes[1:])
                   if space*.35<=b.mean()-a.mean()<=space*1.5]
            if len(pairs)==1:
                clef=[float(r.mean()) for r in pairs[0]]
    reasons, warnings = [], []
    if np.max(np.abs(np.diff(lines)-space)) > 1.1:
        reasons.append('uneven staff lines')
    if min(support) < .45:
        reasons.append('weak staff line support')
    if len(clef) < 2:
        reasons.append('percussion clef / first bar not confirmed')
    if space < 8:
        reasons.append('native staff spacing below 8 px')
    edges = [bars[0] if bars else 0]+[b for b in bars if b>gray.shape[1]*.15]+[gray.shape[1]]
    occupancy = {}
    for label in ('kick','snare'):
        occupied = sum(any(a <= n['x'] < b and n['inst']==label for n in notes) for a,b in zip(edges,edges[1:]))
        occupancy[label] = occupied/max(1,len(edges)-1)
        if occupancy[label] < .5:
            warnings.append(f'{label} absent in most bars: inspect solo or legend')
    return dict(lines=lines, spacing=space, line_support=support, bar_x=bars,
                clef_confirmed=len(clef)>=2, bar_occupancy=occupancy, reasons=reasons, warnings=warnings)


def ornament_score(gray, space, name, x, y, left, top):
    template=cv2.imread(str(Path(__file__).parent/'ornaments'/f'{name}.png'),0)
    template=cv2.resize(template,None,fx=space/11,fy=space/11)
    h,w=template.shape
    x0,y0=round(x-left*space),round(y-top*space)
    patch=gray[max(0,y0-2):y0+h+3,max(0,x0-2):x0+w+3]
    if patch.shape[0]<h or patch.shape[1]<w:return 0.0
    return float(cv2.matchTemplate(patch,template,cv2.TM_CCOEFF_NORMED).max())


def parenthesis_scores(ink, space, x, y):
    scores=[]
    for side,name in ((-1,'left'),(1,'right')):
        template=cv2.imread(str(Path(__file__).parent/'ornaments'/f'paren_{name}.png'),0)
        x0,x1=(x-1.65*space,x-.65*space) if side<0 else (x+.65*space,x+1.65*space)
        patch=ink[max(0,round(y-space)):round(y+space)+1,max(0,round(x0)):round(x1)+1]
        best=0.0
        for stretch in (.85,1,1.15):
            t=cv2.resize(template,None,fx=space/11*stretch,fy=space/11*stretch)
            if min(np.array(patch.shape)-t.shape)>=0:
                best=max(best,float(cv2.matchTemplate(patch,t,cv2.TM_CCOEFF_NORMED).max()))
        scores.append(best)
    return scores


def heads(gray, template_dir=None):
    lines, space = staff_lines(gray)
    dark = (gray < 110).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(round(space*.85),round(space*.62)))
    core = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    _, _, stats, cents = cv2.connectedComponentsWithStats(core)
    found = []
    for (x,y,w,h,area),(cx,cy) in zip(stats[1:], cents[1:]):
        if w > space*1.65 or w < space*.65 or area < space*space*.43:
            continue
        count = max(1,round(h/space)) if h > space*1.5 else 1
        for i in range(count):
            yy = cy if count==1 else y+(i+.5)*h/count
            pos = (lines[-1]-yy)/(space/2)
            if not -3 <= pos <= 12 or abs(pos-round(pos)) > .32:
                continue
            # Beam intersections above the staff are not filled heads.
            if pos > 8 or h/count < space*.7:
                continue
            found.append(dict(x=float(cx),y=float(yy),pos=round(pos),shape='filled',
                              shape_confidence=.94, w=int(w),h=float(h/count), grace=bool(w<space*1.05)))
    # Real glyph crops include anti-aliasing; match at this staff's native scale.
    paths=list((Path(__file__).parent/'templates').glob('*.png'))
    if template_dir:
        paths+=list(Path(template_dir).glob('*.png'))
    for path in sorted(paths):
        shape = path.stem.split('_')[0]
        template = cv2.imread(str(path),0)
        scale = space/11
        template = cv2.resize(template,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
        th,tw = template.shape
        if th >= gray.shape[0] or tw >= gray.shape[1]:
            continue
        response = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
        for pos in range(-3,13):
            cy = lines[-1]-pos*space/2
            y0,y1 = max(0,round(cy-(th-1)/2-2)),min(len(response),round(cy-(th-1)/2+3))
            if y1<=y0:
                continue
            scores = response[y0:y1].max(axis=0)
            peaks,_ = find_peaks(scores,height=.72 if shape=='cross' else .9 if shape=='diamond' else .82 if shape=='filled' else .78,distance=max(3,round(space)))
            for x in peaks:
                yy = y0+int(np.argmax(response[y0:y1,x]))
                cx,cy = x+(tw-1)/2,yy+(th-1)/2
                actual=(lines[-1]-cy)/(space/2)
                if abs(actual-pos)>.35:
                    continue
                found.append(dict(x=float(cx),y=float(cy),pos=pos,shape=shape,
                                  shape_confidence=round(float(scores[x]),3),w=tw,h=th,grace=False))
    # Circled crosses win over the cross inside them; each location is one head.
    kept=[]
    for h in sorted(found,key=lambda h:(h['shape']=='circled',h['shape']=='diamond',h['shape_confidence']),reverse=True):
        if any(abs(h['x']-k['x'])<space*.7 and abs(h['y']-k['y'])<space*.55 for k in kept):
            continue
        kept.append(h)
    nolines=dark.copy()
    for y in lines:
        r=round(y)
        if r>=3 and r+3<len(dark):
            nolines[r-1:r+2,(dark[r-3]==0)&(dark[r+3]==0)]=0
    for h in kept:
        x,y=h['x'],h['y']
        def arc(a,b):
            patch=nolines[max(0,round(y-space*.6)):round(y+space*.6)+1,max(0,round(x+a*space)):round(x+b*space)]
            return int(patch.sum())
        parens=parenthesis_scores(nolines,space,x,y) if h['shape']=='filled' else [0,0]
        h['ghost']=bool(arc(-1.15,-.75)>=4 and arc(.75,1.15)>=4 and min(parens)>=.28 and sum(parens)>=.78)
        h['ghost_confidence']=round(min(parens),3) if h['ghost'] else None
        h['small']=h['grace']
        if h['grace']:
            h['grace_score']=ornament_score(gray,space,'grace_body',x,y,.8,21/11)
            h['grace']=h['grace_score']>=.8
    return sorted(kept,key=lambda h:h['x']),lines,space
