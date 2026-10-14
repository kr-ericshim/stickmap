"""Keep every staff and its reading checks, even when it has no usable cursor."""
import argparse
import json
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from scipy.signal import find_peaks
from heads import heads, staff_checks, staffs

HERE = Path(__file__).parent


def marks(gray, space, extra=None):
    found=[]
    paths=list((HERE/'marks').glob('*.png'))
    if extra:paths+=list((HERE/extra).glob('*.png'))
    for path in paths:
        # 'open_round.png' and 'open_oval.png' are two glyph variants of one mark.
        mark=path.stem.split('_')[0]
        template=cv2.imread(str(path),0)
        template=cv2.resize(template,None,fx=space/11,fy=space/11)
        h,w=template.shape
        if h>=len(gray):
            continue
        response=cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
        maximum=cv2.dilate(response,np.ones((max(3,round(space)),max(3,round(space))),np.uint8))
        ys,xs=np.where((response==maximum)&(response>(.83 if mark!='closed' else .86)))
        for x,y in zip(xs,ys):
            found.append(dict(mark=mark,x=float(x+(w-1)/2),y=float(y+(h-1)/2),confidence=float(response[y,x])))
    return found


def label_heads(gray, legend):
    found,lines,space=heads(gray, HERE/legend['template_dir'] if legend.get('template_dir') else None,
                          prefer_line_span=legend.get('prefer_staff_line_span',False))
    notes=[]
    for h in found:
        if h['x'] < gray.shape[1]*legend['min_x_fraction']:
            continue
        rule=legend['rules'].get(f"{h['shape']}:{h['pos']}",{'inst':'unknown','confidence':0})
        notes.append({**h,**{k:v for k,v in rule.items() if k!='confidence'},
                      'confidence':round(min(h['shape_confidence'],rule['confidence']),3),
                      'accent':False,'flam':False,'roll':False,'marks':[]})
    # Small heads preceding a full-size head are grace notes, not ride bells.
    grace=[]
    for h in notes:
        if not h['grace']:
            continue
        main=[n for n in notes if not n['grace'] and 0<n['x']-h['x']<space*3 and abs(n['pos']-h['pos'])<=1]
        if main:
            n=min(main,key=lambda n:n['x']);n['flam']=True
            n['marks'].append(dict(mark='flam',confidence=.8,grace_x=h['x'],grace_y=h['y']))
            grace.append(h)
    notes=[h for h in notes if h not in grace and
           (not h.get('small',False) or (legend.get('keep_small_filled') and
            h['shape']=='filled' and not h.get('grace') and h['w']>=space*.85))]
    from ties import mark_continuations
    mark_continuations(gray,notes,lines,space)
    for m in marks(gray,space,legend.get('mark_template_dir')):
        candidates=[n for n in notes if abs(n['x']-m['x'])<space*.8]
        if m['mark'] in ('open','closed'):
            candidates=[n for n in candidates if n['inst']=='hihat' and n.get('articulation')!='pedal' and space<n['y']-m['y']<space*7]
        elif m['mark']=='accent':
            candidates=[n for n in candidates if space*1.2<abs(n['y']-m['y'])<space*5.5 and (m['y']<lines[0]-space or m['y']>lines[-1]+space)]
        else:
            candidates=[n for n in candidates if space<abs(n['y']-m['y'])<space*5]
        if candidates:
            n=min(candidates,key=lambda n:abs(n['y']-m['y']))
            if m['mark'] in ('open','closed'):
                n['articulation']=m['mark']
            else:
                n[m['mark']]=True
            n['marks'].append(m)
    ghost_templates=legend.get('ghost_templates',[])
    if legend.get('ghost_template'):ghost_templates=[legend['ghost_template'],*ghost_templates]
    if ghost_templates:
        from harvest_flags import ghost_score
        for n in notes:
            scores=[ghost_score(gray,n,space,config) for config in ghost_templates]
            score=max(scores)
            n['ghost']=any(q>=cfg['threshold'] for q,cfg in zip(scores,ghost_templates))
            n['ghost_confidence']=round(score,3) if n['ghost'] else None
            if n['ghost']:
                n['marks'].append(dict(mark='ghost_parentheses',confidence=round(score,3),method='calibrated complete parenthesized head glyph'))
    checks=staff_checks(gray,lines,space,notes,strict_clef=legend.get('strict_clef_pair',False))
    if legend.get('clef_template') and not checks['clef_confirmed']:
        from harvest_flags import clef_score
        score=clef_score(gray,lines,space,notes,legend['clef_template'])
        checks['clef_template_score']=score
        if score>=legend['clef_template']['threshold']:
            checks['clef_confirmed']=True
            checks['reasons']=[r for r in checks['reasons'] if r!='percussion clef / first bar not confirmed']
            checks['clef_evidence']=legend['clef_template']['evidence']
    if legend.get('harvest_profile'):
        _,_,components,_=cv2.connectedComponentsWithStats((gray<140).astype(np.uint8))
        for n in notes:
            rule=legend['rules'].get(f"{n['shape']}:{n['pos']}",{})
            n['exportable']=bool(rule.get('verified') and n['confidence']>=legend.get('min_confidence',.82))
            if n.get('tied_continuation'):n['exportable']=False
            n['legend_evidence']=rule.get('evidence','unverified mapping')
            for flag in ('ghost','accent','flam','roll'):
                if flag not in legend.get('verified_flags',[]):
                    n[flag]=False
            n['flag_verification']=legend.get('verified_flags',[])
            if n.get('articulation')=='open' and not legend.get('open_verified'):
                n['exportable']=False
            if n['inst']=='hihat' and n.get('articulation')=='closed':
                # An unmatched o/+ must not silently become a closed-hat
                # negative. Isolated upper symbols (including ambiguous text)
                # conservatively withhold that articulation.
                uncertain=[(int(x),int(y),int(w),int(h)) for x,y,w,h,area in components[1:]
                           if space*.2<=w<=space*1.4 and space*.2<=h<=space*1.6
                           and area>=3 and abs(x+w/2-n['x'])<space*.8
                           and space*1.3<n['y']-(y+h/2)<space*7]
                if uncertain:
                    n['unresolved_upper_marks']=uncertain;n['exportable']=False
        if not any(n['exportable'] for n in notes):
            checks['reasons'].append('no verified channel labels')
        if not legend.get('enabled'):
            checks['reasons'].append('channel disabled: reading / legend validation incomplete')
    if legend.get('clef_omitted_verified'):
        checks['reasons']=[r for r in checks['reasons'] if r!='percussion clef / first bar not confirmed']
        checks['clef_evidence']=legend['clef_omitted_verified']
    return notes,checks


def stable_strip_runs(folder,records,legend):
    """Local continuation evidence for explicitly inspected native strips.

    At least three consecutive, unique, supported staffs must agree. Missing
    geometry, page rows, native size/support failures and scene changes end a
    run. This removes only the clef-absence reason, never a label/timing gate.
    """
    if not (legend.get('harvest_profile') and legend.get('score_region') and
            legend.get('enabled') and legend.get('strict_clef_pair')):
        return {}
    evidence={};run=[];positions=[];spaces=[]
    def finish():
        if len(run)>=3:
            proof=dict(systems=[r['system'] for r in run],
                       method='three consecutive supported native staffs in inspected score region')
            for r in run:evidence[r['system']]=proof
    for r in records:
        gray=cv2.imread(str(folder/f"system_{r['system']:03d}.png"),0)
        found=staffs(gray,prefer_line_span=legend.get('prefer_staff_line_span',False)) if gray is not None else []
        valid=len(found)==1 and r.get('page_rows',1)==1 and not r.get('reading_reasons')
        if valid:
            ls,sp=found[0]
            checks=staff_checks(gray,ls,sp,[])
            valid=not [q for q in checks['reasons'] if q!='percussion clef / first bar not confirmed']
        if not valid:
            finish();run=[];positions=[];spaces=[];continue
        native=np.asarray(ls)+r.get('crop',[0])[0]
        if run and (r['start']<run[-1]['start'] or r['start']>run[-1]['end'] or
                    r.get('foreground')!=run[-1].get('foreground') or
                    np.ptp(spaces+[sp])>1.1 or
                    np.max(np.ptp(np.asarray(positions+[native]),axis=0))>np.median(spaces+[sp])*.5):
            finish();run=[];positions=[];spaces=[]
        run.append(r);positions.append(native);spaces.append(sp)
    finish()
    return evidence


def overlay(gray, s, path):
    scale=2
    vis=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    for y in s.get('staff',{}).get('lines',[]):
        cv2.line(vis,(0,round(y)),(gray.shape[1]-1,round(y)),(210,170,60),1)
    vis=cv2.resize(vis,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
    table_rows=(len(s['notes'])+5)//6
    vis=cv2.copyMakeBorder(vis,45,table_rows*22+25,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
    for i,n in enumerate(s['notes']):
        x,y=round(n['x']*scale),round(n['y']*scale)+45
        color=(20,130,20) if n['confidence']>=.8 else (0,100,230)
        cv2.circle(vis,(x,y),round(s['staff']['spacing']*1.1),color,1)
        cv2.putText(vis,str(i),(x-6,y-13),0,.36,color,1)
        text=f"{i}: {n['inst']} p{n['pos']} {n['confidence']:.2f}"
        if n.get('articulation'):text+=' '+n['articulation']
        text+=' '+''.join(k[0].upper() for k in ['ghost','accent','flam','roll'] if n.get(k))
        yy=gray.shape[0]*scale+65+(i//6)*22
        cv2.putText(vis,text,(8+(i%6)*(vis.shape[1]//6),yy),0,.44,color,1)
    title=f"system {s['system']}  {s['interval'][0]:.2f}..{s['interval'][1]:.2f}s  "
    title+='; '.join(s['staff']['reasons']+s['staff']['warnings']) or 'staff checks passed'
    cv2.putText(vis,title,(8,25),0,.48,(0,0,0),1)
    cv2.imwrite(str(path),vis)


def read(folder, legend=None):
    folder=Path(folder)
    legend_path=Path(legend) if legend else HERE/'legends/kbdrums.json'
    legend=json.loads(legend_path.read_text())
    d=json.loads((folder/'scan.json').read_text());fps=d['fps']
    variant=('white_overlay' if d.get('foreground')=='white_overlay' else
             'static_light' if d.get('layout')=='dailychops' and not d.get('reconstructed_scroll') else None)
    overrides=legend.get('variants',{}).get(variant,{})
    if overrides:
        legend={**legend,**overrides,'rules':{**legend['rules'],**overrides.get('rules',{})}}
    elif (variant=='static_light' and legend.get('static_light_min_x_fraction') is not None):
        legend={**legend,'min_x_fraction':legend['static_light_min_x_fraction']}
    records=d.get('records')
    if records is None:
        cur=np.array(d['cursor']);starts=d['systems']+[len(cur)];records=[]
        for k,(a,b) in enumerate(zip(starts,starts[1:])):
            on=np.where(cur[a:b]>=0)[0]+a
            records.append(dict(system=k,start=a,end=b,cursor_frames=on.tolist(),cursor_x=cur[on].tolist()))
    check=folder/'check';check.mkdir(exist_ok=True)
    manual_path=folder/'reading-corrections.json'
    manual=json.loads(manual_path.read_text()) if manual_path.exists() else {}
    rejection_path=folder/'reading-rejections.json'
    reading_rejections=json.loads(rejection_path.read_text()) if rejection_path.exists() else {}
    systems=[]
    from media import source_video_id
    video_id=source_video_id(d['video']) if d.get('video') else None
    source_rejections=legend.get('source_rejections',{}).get(video_id,[])
    continuation=stable_strip_runs(folder,records,legend) if not d.get('reconstructed_scroll') else {}
    for r in records:
        page_rows=r.get('page_rows',sum(q['start']==r['start'] for q in records))
        if not d.get('record_cursors') and sum(q['start']==r['start'] for q in records)==1:
            on=[j for j in range(r['start'],r['end']) if d['cursor'][j]>=0]
            r={**r,'cursor_frames':on,'cursor_x':[d['cursor'][j] for j in on]}
        gray=cv2.imread(str(folder/f"system_{r['system']:03d}.png"),0)
        if gray is None:
            continue
        s=dict(system=r['system'],interval=[r['start']/fps,r['end']/fps],row=r.get('row',0),
               legend=legend['name'],notes=[],cursor_x=[],cursor_t=[])
        if legend.get('moment_tolerance'):
            s['moment_tolerance']=legend['moment_tolerance']
        if page_rows>1 and r['cursor_frames']:
            s['display_interval']=s['interval']
            s['interval']=[r['cursor_frames'][0]/fps,(r['cursor_frames'][-1]+1)/fps]
        if r.get('native_space',99)<6:
            factor=3
            gray=cv2.resize(gray,None,fx=factor,fy=factor,interpolation=cv2.INTER_CUBIC)
        else:
            factor=1
        try:
            s['notes'],s['staff']=label_heads(gray,legend)
            for correction in manual.get(str(s['system']),[]):
                if correction['action']=='add':
                    n=correction['note'].copy()
                    n['y']=s['staff']['lines'][-1]-n['pos']*s['staff']['spacing']/2
                    n.update(manual_reading=True,reading_evidence=correction['evidence'])
                    s['notes'].append(n)
            s['notes'].sort(key=lambda n:n['x'])
            s['staff']['reasons'].extend(r.get('scroll_reasons',[]))
            clef_evidence = (legend.get('white_overlay_clef_omitted_verified')
                             if d.get('foreground') == 'white_overlay'
                             else legend.get('scroll_clef_omitted_verified'))
            if d.get('foreground')=='white_overlay':
                proof=r.get('staff_geometry') or d.get('opening_clef')
                s['staff']['clef_guard']=dict(
                    reconstructed_scroll=bool(d.get('reconstructed_scroll')),
                    registration_reasons=r.get('scroll_reasons',[]),
                    local_staff_geometry=bool(r.get('staff_geometry')),
                    opening_clef=bool(d.get('opening_clef')),
                    profile_continuation_approved=bool(clef_evidence),
                    continuation_applied=bool(d.get('reconstructed_scroll') and
                        not r.get('scroll_reasons') and proof and clef_evidence))
                if not proof:clef_evidence=None
                elif clef_evidence:s['staff']['continuation_evidence']=proof
            if d.get('reconstructed_scroll') and not r.get('scroll_reasons') and clef_evidence:
                s['staff']['reasons']=[q for q in s['staff']['reasons']
                                      if q!='percussion clef / first bar not confirmed']
                s['staff']['clef_evidence']=clef_evidence
            if r['system'] in continuation:
                s['staff']['reasons']=[q for q in s['staff']['reasons']
                                      if q!='percussion clef / first bar not confirmed']
                s['staff']['continuation_evidence']=continuation[r['system']]
            s['staff']['reasons'].extend(r.get('reading_reasons',[]))
            s['staff']['reasons'].extend(reading_rejections.get(str(s['system']),[]))
            for exclusion in source_rejections:
                a,b=exclusion['interval']
                if s['interval'][0]<b and s['interval'][1]>a:
                    s['staff']['reasons'].append(exclusion['reason'])
                    s['staff'].setdefault('source_rejections',[]).append(exclusion)
                    for note in s['notes']:
                        note['exportable']=False
                        note['reading_exclusion']=exclusion['reason']
            if r.get('read_x'):
                lo,hi=sorted(r['read_x'])
                s['notes']=[n for n in s['notes'] if lo<=n['x']<=hi]
            if page_rows>1 and not r['cursor_frames']:
                s['staff']['reasons'].append('page row not visited by cursor')
            if factor>1:
                s['staff']['reasons'].append('upscaled low-resolution source; inspection only')
            xs=np.array(r['cursor_x'])*factor;ts=np.array(r['cursor_frames'])/fps
            if len(xs)>2:
                s['cursor_mode']='box' if np.mean(abs(np.diff(xs))<=1)>.55 else 'sweep'
                if s['cursor_mode']=='box':
                    groups=[];start=0
                    for j in range(1,len(xs)+1):
                        if j==len(xs) or abs(xs[j]-xs[start])>s['staff']['spacing']*.6:
                            if j-start>=2:
                                groups.append((float(np.median(xs[start:j])),float(ts[start])))
                            start=j
                    xs=np.array([x for x,t in groups]);ts=np.array([t for x,t in groups])
                keep=np.r_[True,np.diff(xs)>=-2] if len(xs) else np.array([],bool)
                xs,ts=xs[keep],ts[keep]
                first=np.r_[True,np.diff(np.maximum.accumulate(xs))>.5] if len(xs) else np.array([],bool)
                s['cursor_x']=xs[first].tolist();s['cursor_t']=ts[first].tolist()
                s['cursor_coverage']=float(np.ptp(xs)/gray.shape[1]) if len(xs) else 0
                partial_visit=legend.get('trim_partial_cursor') and len(xs)>=6
                if legend.get('harvest_profile') and len(xs)>3 and (s['cursor_coverage']>.35 or partial_visit):
                    lo,hi=float(xs.min()-s['staff']['spacing']),float(xs.max()+s['staff']['spacing'])
                    s['unvisited_heads']=[n for n in s['notes'] if not lo<=n['x']<=hi]
                    s['notes']=[n for n in s['notes'] if lo<=n['x']<=hi]
                    s['read_x']=[lo,hi]
                    if partial_visit:
                        s['reading_span_evidence']='at least six cursor anchors; retain only visited head positions; colour/timing gates unchanged'
            else:
                s['cursor_coverage']=0
        except ValueError as e:
            s['staff']=dict(reasons=[str(e),*r.get('scroll_reasons',[]),
                                    *r.get('reading_reasons',[]),
                                    *reading_rejections.get(str(s['system']),[])],warnings=[],lines=[])
        overlay(gray,s,check/f"system_{s['system']:03d}.png")
        systems.append(s)
    keep={f"system_{s['system']:03d}.png" for s in systems}
    for p in check.glob('system_*.png'):
        if p.name not in keep:p.unlink()
    (folder/'strip.json').write_text(json.dumps(systems,indent=2))
    print(f'{folder.name}: {len(systems)} systems, {sum(len(s["notes"]) for s in systems)} heads',flush=True)
    return systems


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder');p.add_argument('--legend')
    a=p.parse_args();read(a.folder,a.legend)
