"""Convert hybrid readings using CV staff and video-only bar-time anchors.

There are no reviewed flags, acoustic peak inputs, detector labels or reference
truth times in this adapter. All acoustic decisions remain in align/tiers.
"""
import argparse
import copy
from fractions import Fraction
from pathlib import Path
import numpy as np
from common import DATA, ROOT, dump, flatten, load, sha, validate


def video_times(xs, anchor, spacing):
    cx=np.asarray(anchor['cursor_x'],float);ct=np.asarray(anchor['cursor_t'],float)
    xs=np.asarray(xs,float)
    values=np.interp(xs,cx,ct)
    if anchor.get('cursor_mode')=='box':
        near=np.argmin(abs(xs[:,None]-cx[None,:]),axis=1)
        close=abs(xs-cx[near])<spacing*1.25
        values[close]=ct[near[close]]
    slope=(ct[-1]-ct[0])/max(1,cx[-1]-cx[0])
    values[xs<cx[0]]=ct[0]+(xs[xs<cx[0]]-cx[0])*slope
    values[xs>cx[-1]]=ct[-1]+(xs[xs>cx[-1]]-cx[-1])*slope
    return values


def convert_auto(reading,meta,cv,legend):
    anchor=meta['automatic_anchor']
    staff=copy.deepcopy(cv['staff'])
    staff.setdefault('reasons',[])
    if len(staff.get('lines',[]))!=5:
        raise ValueError('CV did not establish a native five-line staff')
    spacing=staff['spacing']
    if spacing<8 and 'native staff spacing below 8 px' not in staff['reasons']:
        staff['reasons'].append('native staff spacing below 8 px')
    if anchor.get('staff_continuation'):
        staff['reasons']=[r for r in staff['reasons'] if r!='percussion clef / first bar not confirmed']
        staff['continuation_evidence']=anchor['staff_continuation']
    heads=flatten(reading)
    native={f"cv-{n['label']}":n for n in cv['notes']}
    owned=anchor.get('owned_x')
    def belongs(h):
        return owned is None or h['x'] is None or owned[0]<=h['x']<=owned[1]
    included=[h for h in heads if belongs(h)]
    omitted=[]
    timed=[]
    for h in included:
        if h.get('grace'):
            omitted.append(dict(id=h['id'],reason='resolved grace, no separate attack',blocking=False))
        elif h['onset'] is None or h['tie_continuation'] is None or h.get('rhythm_uncertain'):
            omitted.append(dict(id=h['id'],reason='unresolved owned onset/tie/rhythm',blocking=True))
        else:
            timed.append(h)
    if any(x['blocking'] for x in omitted):
        staff['reasons'].append('hybrid contains unresolved heads inside the visited score span')
    if owned is not None and any(h['x'] is None for h in heads):
        staff['reasons'].append('unlocated hybrid head prevents automatic visited-span coverage')
    # The axis preserves exact within-bar rhythm. Unknown-meter bar extents
    # are lower bounds from the reading, not invented time signatures.
    offset=Fraction(0);offsets=[];extents=[]
    meter=reading['time_signature']
    for bar in reading['bars']:
        offsets.append(offset)
        known=[Fraction(e['onset'])+Fraction(e['duration'] or '0') for e in bar['events'] if e['onset'] is not None]
        extent=Fraction(meter['numerator'],meter['denominator']) if meter and not bar['partial_left'] else max(known,default=Fraction(0))
        if known:
            extent=max(extent,max(known))
        extents.append(str(extent));offset+=extent
    evidence=[];bar_points={};rates=[]
    has_cursor=len(anchor.get('cursor_x',[]))>=4 and np.ptp(anchor['cursor_x'])>0
    if has_cursor:
        for bi in range(reading['bar_count']):
            hs=[h for h in timed if h['bar']==bi and h['x'] is not None and not h['tie_continuation']]
            by_onset={}
            for h in hs:
                by_onset.setdefault(Fraction(h['onset']),[]).append(h['x'])
            q=np.asarray([float(x) for x in sorted(by_onset)])
            px=np.asarray([float(np.mean(by_onset[x])) for x in sorted(by_onset)])
            if len(q):
                t=video_times(px,anchor,spacing)
                bar_points[bi]=(q,t,px)
                # Median pair slopes are resistant to engraving padding,
                # grace clusters and a cursor sample just before a head.
                slopes=[(t[j]-t[i])/(q[j]-q[i]) for i in range(len(q)) for j in range(i+1,len(q))
                        if q[j]-q[i]>=1/16 and t[j]>t[i]]
                if slopes:
                    rates.append(float(np.median(slopes)))
    global_rate=float(np.median(rates)) if rates else None
    bar_fit={}
    if global_rate is not None and global_rate>0:
        for bi,(q,t,px) in bar_points.items():
            slopes=[(t[j]-t[i])/(q[j]-q[i]) for i in range(len(q)) for j in range(i+1,len(q))
                    if q[j]-q[i]>=1/16 and t[j]>t[i]]
            rate=float(np.median(slopes)) if len(q)>=3 and slopes else global_rate
            origin=float(np.median(t-q*rate))
            bar_fit[bi]=(origin,rate)
            evidence.append(dict(bar=bi,origin_seconds=origin,seconds_per_whole=rate,
                                 onsets_whole=q.tolist(),native_x=px.tolist(),video_times=t.tolist(),
                                 residual_p95_ms=float(np.percentile(abs(t-origin-q*rate),95)*1000),
                                 method='median slopes from printed rational onsets and video cursor only'))
    notes=[];points={}
    absolute=[float(offsets[h['bar']]+Fraction(h['onset'])) for h in timed if not h['tie_continuation']]
    loq,hiq=(min(absolute),max(absolute)) if absolute else (0,0)
    for h in timed:
        q=offsets[h['bar']]+Fraction(h['onset']);x=float(q*4096)
        orig=native.get(h['id'],{})
        rule=legend['rules'].get(f"{h['notehead']}:{h['pos']}",{})
        inst=(orig.get('inst') if h.get('source')=='cv' and orig.get('verified') and rule.get('verified')
              and h['instrument']==orig.get('inst')==rule.get('inst') else 'unknown')
        exportable=bool(inst!='unknown' and orig.get('exportable') and not h['tie_continuation'])
        n={**orig,'id':h['id'],'x':x,'y':h['y'],'native_x':h['x'],'native_y':h['y'],
           'pos':h['pos'],'shape':h['notehead'],'inst':inst,'exportable':exportable,
           'tied_continuation':h['tie_continuation'],'bar':h['bar'],'onset_q':str(Fraction(h['onset'])*4),
           'duration_q':str(Fraction(h['duration'])*4) if h['duration'] is not None else None,
           'tuplet':h['tuplet'],'observed_ghost':h['ghost'],'observed_accent':h['accent'],
           'ghost':bool(orig.get('ghost',False)),'accent':bool(orig.get('accent',False)),
           'flam':bool(orig.get('flam',False)),'roll':bool(orig.get('roll',False)),
           'confidence':orig.get('confidence',0),'shape_confidence':orig.get('shape_confidence',0),
           'head_independently_reviewed':False,'identity_provenance':'native CV and existing verified channel legend' if inst!='unknown' else 'unknown; no CV instrument approval',
           'source_system':meta['record'].get('source_system')}
        notes.append(n)
        if h['tie_continuation']:
            continue
        if h['bar'] in bar_fit:
            origin,rate=bar_fit[h['bar']];t=origin+float(Fraction(h['onset']))*rate
        elif has_cursor and h['x'] is not None:
            # A bar with too little rhythmic evidence retains an observed video
            # knot; it does not fabricate a per-bar tempo.
            t=float(video_times([h['x']],anchor,spacing)[0])
        else:
            a,b=anchor['interval'];t=a+(float(q)-loq)/max(1e-9,hiq-loq)*(b-a-.1)
        if x in points and abs(points[x]-t)>.005:
            staff['reasons'].append('inconsistent chord time priors')
        points[x]=t
    xs=sorted(points)
    if any(b-a<=spacing*.55 for a,b in zip(xs,xs[1:])):
        staff['reasons'].append('distinct rational moments collide on rhythm axis')
    if len(xs)<6:
        staff['reasons'].append('fewer than six resolved notated moments')
    return dict(system=meta['system'],vision_system_id=meta['system_id'],notes=notes,staff=staff,
                interval=anchor['interval'],cursor_x=[],cursor_t=[],rhythm_x=xs,rhythm_t=[points[x] for x in xs],
                moment_tolerance=.55,native_spacing=spacing,vision_omitted_heads=omitted,
                unvisited_head_ids=[h['id'] for h in heads if not belongs(h)],
                hybrid_rejected_cv_heads=reading['hybrid'].get('rejected_cv_heads',[]),
                axis='exact whole-note rhythm * 4096; unknown-meter bar extents are reported lower bounds',
                time_anchor_evidence={**anchor,'bar_fits':evidence,'bar_offsets_whole':list(map(str,offsets)),
                                      'bar_extents_whole':extents,'audio_used':False,'manual_review_used':False})


def stage(run):
    import sys
    sys.path.insert(0,str(ROOT))
    from tools.harvest.vault import Vault
    folder=DATA/'runs'/run
    config=load(folder/'config.json');videos={};records=[]
    outcomes={r['id']:r for r in load(folder/'run.json')['results']} if (folder/'run.json').exists() else {}
    for sid in config['systems']:
        job=folder/sid;meta=load(job/'job.json') if (job/'job.json').exists() else load(DATA/'hybrid/jobs'/sid/'job.json')
        if Vault().is_vault(meta['video']):
            raise ValueError('vault input forbidden: '+meta['video'])
        videos.setdefault(meta['video'],[])
        result=load(job/'result.json') if (job/'result.json').exists() else outcomes.get(sid,{'status':'not_run'})
        if result['status']!='ok':
            records.append(dict(id=sid,video=meta['video'],status=result['status']));continue
        if sha(job/'reading.json')!=result['reading_sha256']:
            raise ValueError('reading changed after inference: '+sid)
        reading=validate(load(job/'reading.json'),sid)
        try:
            system=convert_auto(reading,meta,load(job/'cv.json'),load(job/'legend.json'))
        except (ValueError,TypeError,KeyError,OverflowError,ZeroDivisionError) as error:
            system=dict(system=meta['system'],vision_system_id=sid,notes=[],
                        staff={'reasons':['automatic anchor conversion failed: '+str(error)]},
                        interval=meta['automatic_anchor']['interval'],cursor_x=[],cursor_t=[])
        videos.setdefault(meta['video'],[]).append(system)
        records.append(dict(id=sid,video=meta['video'],status='staged',notes=len(system['notes']),reasons=system['staff']['reasons']))
    for video,systems in videos.items():
        dump(DATA/'candidates'/video/'strip.json',systems)
    dump(DATA/('handoff-'+run+'.json'),dict(run=run,videos=sorted(videos),systems=records,audio_used=False))
    return videos


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True)
    print({k:len(v) for k,v in stage(p.parse_args().run).items()})
