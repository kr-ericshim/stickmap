"""Conservative raster ties between adjacent heads at the same staff position.

Heads remain inspectable; a positively traced continuation is excluded from
both the acoustic onset sequence and every timed export. Partial/offscreen
ties without two visible endpoints are deliberately not inferred.
"""
import numpy as np


def mark_continuations(gray,notes,lines,space):
    ink=gray<150
    previous={}
    for note in sorted(notes,key=lambda n:n['x']):
        key=(note['pos'],note['shape'])
        source=previous.get(key);previous[key]=note
        if source is None or note.get('grace') or source.get('grace'):continue
        gap=note['x']-source['x']
        if not space*1.8<=gap<=space*12:continue
        x0=round(source['x']+max(source['w']/2,space*.55))
        x1=round(note['x']-max(note['w']/2,space*.55))
        if x1-x0<space:continue
        xs=np.arange(max(0,x0),min(gray.shape[1],x1+1))
        t=(xs-x0)/max(1,x1-x0)
        baseline=source['y']+(note['y']-source['y'])*t
        best=None
        for side in (-1,1):
            for shoulder in (.05,.2,.4,.6):
                for height in (.5,.75,1.,1.25,1.5,1.75):
                    yy=baseline+side*space*(shoulder+height*4*t*(1-t))
                    ys=np.rint(yy).astype(int)
                    # Staff lines cannot supply tie pixels. Enough of the
                    # curve must remain independently visible between them.
                    valid=(ys>=2)&(ys<len(gray)-2)
                    for line in lines:valid &= abs(yy-line)>2
                    if valid.sum()<len(xs)*.75:continue
                    hits=np.zeros(len(xs),bool)
                    for delta in (-1,0,1):
                        hits[valid] |= ink[ys[valid]+delta,xs[valid]]
                    # Require evidence in each third, including both ends;
                    # one curved fragment, beam or head body cannot qualify.
                    thirds=np.array_split(np.flatnonzero(valid),3)
                    if all(len(q) and hits[q].mean()>=.88 for q in thirds):
                        score=float(hits[valid].mean())
                        if best is None or score>best['support']:
                            best=dict(source_x=source['x'],source_y=source['y'],
                                      side=side,support=score,
                                      method='same-position endpoints with supported curved native ink')
        if best:
            note['tied_continuation']=True
            note['exportable']=False
            note['tie_evidence']=best
