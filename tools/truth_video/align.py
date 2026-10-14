"""Align to unlabeled attacks and test predictions on moments withheld from the warp.

Regular grooves let a whole-beat or subdivision shift agree with class-agnostic
attacks almost as well as the true alignment. Band-limited onset strength from
the drum stem (bands.py) breaks that tie without detector labels: notated kicks
must sit on low-band onsets and notated snares on mid-band onsets, and a
system is rejected unless its alignment beats every reachable shift.
"""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
import bands as band_evidence

PHASE = dict(min_notes=6, min_agreement=.6, min_gain=.15)


def match(ref, est, tol=.05):
    pairs=[];i=j=0
    while i<len(ref) and j<len(est):
        delta=est[j]-ref[i]
        if abs(delta)<=tol:
            pairs.append((i,j));i+=1;j+=1
        elif delta<0:j+=1
        else:i+=1
    return pairs


def quality(times, attacks):
    return {f'within_{ms}ms':len(match(sorted(times),attacks,ms/1000))/max(1,len(times)) for ms in (20,40)}


def moments(s):
    groups=[]
    tolerance=s.get('staff',{}).get('spacing',11)*s.get('moment_tolerance',.55)
    for i,n in sorted(enumerate(s['notes']),key=lambda pair:pair[1]['x']):
        if n.get('tied_continuation'):continue
        if groups and n['x']-np.mean([s['notes'][j]['x'] for j in groups[-1]])<tolerance:
            groups[-1].append(i)
        else:groups.append([i])
    return np.array([np.mean([s['notes'][j]['x'] for j in group]) for group in groups]),groups


def moment_instruments(s, groups):
    """Notated instrument names per moment; the legend decides names, never the audio."""
    return [{s['notes'][j].get('inst','unknown') for j in group} for group in groups]


def prior(s,x):
    if s.get('color_x'):
        # Color edges are only used inside their observed span. The cursor
        # supplies the remaining prior, still subject to all timing gates.
        fallback={k:v for k,v in s.items() if k not in ('color_x','color_t')}
        t,_=prior(fallback,x)
        inside=(x>=s['color_x'][0])&(x<=s['color_x'][-1])
        t[inside]=np.interp(x[inside],s['color_x'],s['color_t'])
        return t,'color'
    if s.get('rhythm_x'):
        return np.interp(x,s['rhythm_x'],s['rhythm_t']),'rhythm'
    cx,ct=np.array(s['cursor_x']),np.array(s['cursor_t'])
    if len(cx)>3 and s.get('cursor_coverage',1)>.35:
        t=np.interp(x,cx,ct)
        if s.get('cursor_mode')=='box':
            nearest=np.argmin(abs(x[:,None]-cx[None,:]),axis=1)
            close=abs(x-cx[nearest])<s['staff']['spacing']*1.25
            t[close]=ct[nearest[close]]
        # Extrapolation is small and explicit; clamping makes separate notes simultaneous.
        slope=(ct[-1]-ct[0])/max(1,cx[-1]-cx[0])
        t[x<cx[0]]=ct[0]+(x[x<cx[0]]-cx[0])*slope
        t[x>cx[-1]]=ct[-1]+(x[x>cx[-1]]-cx[-1])*slope
        return t,'cursor'
    lo,hi=s['interval']
    return lo+(x-x[0])/max(1,x[-1]-x[0])*(hi-lo-.1),'interval'


def assignment(pred, attacks, radius):
    """Skipped notes and attacks cost less than forcing a distant correspondence."""
    n,m=len(pred),len(attacks)
    cost=np.full((n+1,m+1),np.inf);back=np.zeros((n+1,m+1),np.uint8)
    cost[0]=np.arange(m+1)*.08;cost[:,0]=np.arange(n+1)*2.5
    for i in range(1,n+1):
        for j in range(1,m+1):
            delta=abs(pred[i-1]-attacks[j-1])
            c=(delta/(radius/2))**2 if delta<radius else 1e6
            options=(cost[i-1,j-1]+c,cost[i-1,j]+2.5,cost[i,j-1]+.08)
            k=int(np.argmin(options));cost[i,j]=options[k];back[i,j]=k
    i,j=n,m;pairs=[]
    while i and j:
        k=back[i,j]
        if k==0:pairs.append((i-1,j-1));i-=1;j-=1
        elif k==1:i-=1
        else:j-=1
    return list(reversed(pairs))


def fit_prior(t, attacks, mode, center=0, width=.4, evidence=None):
    """Offset (and scale for interval priors) with the best class-agnostic attack agreement.

    `evidence=(bands, instruments)` adds the pooled kick/snare band agreement to
    the objective. Rhythmic aliases tie on attacks alone; they do not tie on
    which kind of onset each notated drum lands on.
    """
    best=None
    # A system-level offset accounts for highlight lead/lag; interval mode also needs scale.
    for scale in ([.85,.925,1,1.075,1.15] if mode=='interval' else [1]):
        for offset in np.arange(center-width,center+width+.001,.01):
            candidate=t[0]+(t-t[0])*scale+offset
            q=quality(candidate,attacks)
            value=q['within_20ms']+q['within_40ms']*.5-abs(offset)*.03-abs(scale-1)*.1
            if evidence is not None:
                value+=band_evidence.agreement(evidence[0],candidate,evidence[1])['pooled'] or 0
            if best is None or value>best[0]:best=(value,candidate,float(offset),scale)
    return best[1:]


def warp(x, t, attacks, radius=.16, snap=True):
    pairs=assignment(t,attacks,radius)
    if len(pairs)<2:return t, pairs
    i,j=np.array(pairs).T
    # Offset interpolation retains engraving/cursor pace between observed attacks.
    offsets=median_filter(attacks[j]-t[i],size=min(5,len(i)//2*2+1),mode='nearest')
    mapped=t+np.interp(x,x[i],offsets)
    pairs=assignment(mapped,attacks,.09)
    if snap:
        for a,b in pairs:
            if abs(mapped[a]-attacks[b])<=.06:mapped[a]=attacks[b]
    return mapped,pairs


def phase_established(check):
    """Enough kick/snare moments, most on their onset bands, no alias contradicts it, and some instrument clearly beats every alias."""
    return bool(check and check['count']>=PHASE['min_notes'] and check['pooled']>=PHASE['min_agreement']
                and check.get('contradicted_by') is None and (check['margin'] is None or check['margin']>=PHASE['min_gain']))


def phase_reasons(check, song_phase=None):
    """Why an alignment's instrument-band agreement does not establish its phase.

    `song_phase` is the same check on all other systems' priors at the shared
    lag. It can back a short or rhythmically ambiguous system (four-on-the-
    floor kicks agree with themselves a beat later) that does not contradict
    it; it never rescues a system whose own kicks/snares prefer a shift.
    """
    if check is None:
        return ['instrument-band phase check unavailable (no drum stem)']
    backed=phase_established(song_phase)
    if check['count']<PHASE['min_notes']:
        if not backed:
            return [f'fewer than {PHASE["min_notes"]} notated kick/snare moments for the instrument-band phase check and no song-level phase evidence']
        if check['count'] and check['pooled']<PHASE['min_agreement']:
            return [f'notated kicks/snares off their onset bands (instrument-band agreement below {PHASE["min_agreement"]:.0%})']
        return []
    reasons=[]
    if check['pooled']<PHASE['min_agreement']:
        reasons.append(f'notated kicks/snares off their onset bands (instrument-band agreement below {PHASE["min_agreement"]:.0%})')
    if check.get('contradicted_by') is not None:
        worst=min(check['contradiction_margins'],key=check['contradiction_margins'].get)
        reasons.append(f'phase alias: a {check["contradicted_by"]:+.2f} s shift puts more notated {worst}s on their onset band')
    elif check['best_shift'] is not None and check['margin']<PHASE['min_gain'] and not backed:
        reasons.append(f'phase alias: a {check["best_shift"]:+.2f} s shift agrees as well with kick/snare onset bands (gain < {round(100*PHASE["min_gain"])} points) and no song-level phase evidence')
    return reasons


def align_system(s, attacks, seed=None, bands=None, song_phase=None):
    x,groups=moments(s)
    result=dict(system=s['system'],interval=s['interval'],moments=len(x),usable=False,
                reasons=list(s.get('staff',{}).get('reasons',[])))
    if len(x)<6:
        result['reasons'].append('fewer than six notated moments')
        return [],result
    t,mode=prior(s,x)
    lo,hi=s['interval'];local=attacks[(attacks>=lo-.65)&(attacks<=hi+.65)]
    if len(local)<3:
        result['reasons'].append('too few audio attacks');return [],result
    instruments=moment_instruments(s,groups)
    evidence=(bands,instruments) if bands is not None else None
    result['mode']=mode;result['before']=quality(t,local)
    calibrated,offset,scale=fit_prior(t,local,mode, seed if seed is not None else 0, .04 if seed is not None else .4, evidence)
    result['offset_seconds']=offset;result['scale']=scale
    result['calibrated']=quality(calibrated,local)
    mapped,pairs=warp(x,calibrated,local)
    result['after']=quality(mapped,local)
    result['assigned_fraction']=len(pairs)/len(x)
    # Every held-out moment is excluded from both offset fitting and attack assignment.
    predicted=np.zeros(len(x))
    for fold in range(3):
        train=np.arange(len(x))%3!=fold;test=~train
        fold_evidence=(bands,[q for q,keep in zip(instruments,train) if keep]) if bands is not None else None
        train_t,off,sc=fit_prior(t[train],local,mode, seed if seed is not None else 0, .04 if seed is not None else .4, fold_evidence)
        train_map,_=warp(x[train],train_t,local,snap=False)
        base=t[0]+(t-t[0])*sc
        # fit_prior anchors on the first training time, which can differ by one note.
        base=t[train][0]+(t-t[train][0])*sc+off
        predicted[test]=base[test]+np.interp(x[test],x[train],train_map-train_t)
    result['held_out']=quality(predicted,local)
    result['warp_p95_ms']=float(np.percentile(abs(mapped-calibrated),95)*1000)
    result['strictly_monotone']=bool(np.all(np.diff(mapped)>.005))
    # Dense attacks can make almost any timestamp look good: compare shifted predictions.
    null=[quality(predicted+delta,local)['within_40ms'] for delta in (-.37,-.23,.23,.37)]
    null20=[quality(predicted+delta,local)['within_20ms'] for delta in (-.37,-.23,.23,.37)]
    result['shifted_control_20ms']=float(np.mean(null20))
    result['held_out_gain_20ms']=result['held_out']['within_20ms']-result['shifted_control_20ms']
    result['shifted_control_40ms']=float(np.mean(null))
    result['held_out_gain_40ms']=result['held_out']['within_40ms']-result['shifted_control_40ms']
    if result['after']['within_20ms']<.85 or result['after']['within_40ms']<.95:
        result['reasons'].append('fitted attack agreement below 85% / 95%')
    if result['held_out']['within_20ms']<.55 or result['held_out']['within_40ms']<.8:
        result['reasons'].append('held-out agreement below 55% / 80%')
    if max(result['held_out_gain_40ms'],result['held_out_gain_20ms'])<.15:
        result['reasons'].append('ambiguous alignment in dense attacks (held-out gain at both tolerances < 15 points)')
    if not result['strictly_monotone']:
        result['reasons'].append('warp is not strictly monotone')
    if mode in ('interval','rhythm') and (result['held_out']['within_20ms']<.7 or result['held_out']['within_40ms']<.9):
        result['reasons'].append('no-cursor alignment needs 70% / 90% held-out agreement')
    # Which drum is notated is trusted; the fitted times must put kicks on
    # low-band onsets and snares on mid-band onsets better than any reachable
    # shift. bands=None (no audio evidence supplied) records that no check ran.
    if bands is not None:
        result['phase']=band_evidence.phase_check(bands,mapped,instruments,local)
        result['phase']['song_level']=song_phase
        result['reasons'].extend(phase_reasons(result['phase'],song_phase))
    else:
        result['phase']=None
    result['usable']=not result['reasons']
    result['moment_x']=x.tolist();result['initial_times']=t.tolist();result['times']=mapped.tolist();result['held_out_times']=predicted.tolist()
    result['moment_groups']=[list(map(int,g)) for g in groups]
    result['duration_seconds']=float(mapped[-1]-mapped[0]) if result['usable'] else 0
    return emit(s,result),result


def emit(s, result):
    """Timed, exportable notes of a usable system."""
    events=[]
    if result.get('usable'):
        for time,group in zip(result['times'],result['moment_groups']):
            for index in group:
                n=s['notes'][index]
                if n.get('exportable') is False or n.get('tied_continuation'):
                    continue
                events.append({**n,'time':float(time),'system':s['system'],
                               'timing_confidence':round(result['held_out']['within_40ms'],3)})
    return events


PHASE_PREFIXES=('phase alias','notated kicks/snares off','fewer than 6 notated kick/snare','instrument-band phase check unavailable')


def rejudge_phase(s, result, song_phase):
    """Re-apply the phase reasons with better song-level evidence; nothing else changes."""
    result['reasons']=[r for r in result['reasons'] if not r.startswith(PHASE_PREFIXES)]
    result['phase']['song_level']=song_phase
    result['reasons'].extend(phase_reasons(result['phase'],song_phase))
    result['usable']=not result['reasons']
    result['duration_seconds']=float(result['times'][-1]-result['times'][0]) if result['usable'] else 0
    return emit(s,result)


def unique_events(events):
    """One positive per identical audio time, instrument and staff position.

    Adjacent page visits may observe the same boundary head twice. Keep the
    strict observation when tiers differ and retain the contributing systems.
    Do not merge nearby times or transfer technique flags between observations.
    """
    seen={}
    for event in events:
        key=(event['time'],event['inst'],event.get('pos'),event.get('articulation'))
        if key not in seen:
            seen[key]=event.copy();continue
        old=seen[key]
        systems=sorted(set(old.get('observed_systems',[old['system']])+[event['system']]))
        if event.get('tier')=='strict' and old.get('tier')!='strict':
            old=event.copy();seen[key]=old
        old['observed_systems']=systems
    return sorted(seen.values(),key=lambda event:event['time'])


def build_truth(song, folder, bands='auto'):
    """Align every system of `folder/strip.json` to the song's attacks.

    `bands='auto'` loads band-limited onsets from the song's drum stem; without
    a stem every system is rejected as unverifiable rather than silently
    accepted on class-agnostic attacks alone.
    """
    song,folder=Path(song),Path(folder)
    systems=json.loads((folder/'strip.json').read_text())
    attacks=np.unique(np.load(song/'attacks.htdemucs.npy').astype(float))
    if isinstance(bands,str) and bands=='auto':
        bands=band_evidence.load(song)
    truth=[];reports=[]
    priors={}
    for s in systems:
        if s.get('rhythm_calibration_times'):
            priors[s['system']]=(np.array(s['rhythm_calibration_times']),None)
            continue
        x,groups=moments(s)
        if len(x)>5:
            t,mode=prior(s,x)
            if mode in ('cursor','color'):priors[s['system']]=(t,moment_instruments(s,groups))
    for s in systems:
        # Other systems establish the video's lag without training on this system.
        others=[(t,q) for k,(t,q) in priors.items() if k!=s['system']]
        seed=None;song_phase=None
        if others:
            pooled_t=np.concatenate([t for t,_ in others])
            pooled_q=sum(([set()]*len(t) if q is None else q for t,q in others),[])
            order=np.argsort(pooled_t,kind='stable');pooled_t=pooled_t[order];pooled_q=[pooled_q[i] for i in order]
            evidence=(bands,pooled_q) if bands is not None else None
            seed=fit_prior(pooled_t,attacks,'cursor',evidence=evidence)[1]
            if bands is not None:
                # Song-level phase evidence: the other systems' priors at the shared lag.
                song_phase=band_evidence.phase_check(bands,pooled_t+seed,pooled_q,attacks)
        events,report=align_system(s,attacks,seed,bands,song_phase)
        if bands is None and report.get('mode'):
            report['phase']=None;report['reasons'].extend(phase_reasons(None));report['usable']=False;report['duration_seconds']=0;events=[]
        truth.extend(events);reports.append(report)
    if bands is not None:
        # Second pass: the fitted times of every system whose own kicks/snares
        # establish the phase form cleaner song-level evidence than raw cursor
        # priors. Systems that only lacked backing are judged again; fits,
        # timing gates and label rules are untouched.
        established=[(r,s) for r,s in zip(reports,systems) if r.get('phase') and phase_established(r['phase'])
                     and not [q for q in r['reasons'] if not q.startswith(PHASE_PREFIXES)]]
        if established:
            pooled_t=np.concatenate([r['times'] for r,_ in established])
            pooled_q=sum((moment_instruments(s,r['moment_groups']) for r,s in established),[])
            order=np.argsort(pooled_t,kind='stable')
            song_phase=band_evidence.phase_check(bands,pooled_t[order],[pooled_q[i] for i in order],attacks)
            song_phase['source']='fitted times of %d individually established systems'%len(established)
            truth=[]
            for r,s in zip(reports,systems):
                if r.get('phase'):truth.extend(rejudge_phase(s,r,song_phase))
    truth=unique_events(truth)
    # Each scored interval belongs to one accepted system; gaps never become negatives.
    intervals=[[r['times'][0],r['times'][-1]] for r in reports if r['usable']]
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
        else:merged.append([a,b])
    report=dict(version=2,song=str(song.resolve()),systems=reports,intervals=merged,
                trustworthy_seconds=sum(b-a for a,b in merged),
                thresholds=dict(after_20ms=.85,after_40ms=.95,held_out_20ms=.55,held_out_40ms=.8,held_out_gain_either_tolerance=.15,no_cursor_held_out_20ms=.7,no_cursor_held_out_40ms=.9,
                                phase_min_kick_snare_moments=PHASE['min_notes'],phase_min_agreement=PHASE['min_agreement'],phase_min_gain_over_shifts=PHASE['min_gain']),
                band_evidence=None if bands is None else dict(source=str(band_evidence.stem_path(song) or ''),bands=band_evidence.BANDS,rules=band_evidence.RULES,version=band_evidence.VERSION))
    for key in ('before','calibrated','after','held_out'):
        available=[r for r in reports if key in r];total=sum(r['moments'] for r in available)
        report[key]={f'within_{ms}ms':sum(r[key][f'within_{ms}ms']*r['moments'] for r in available)/max(1,total) for ms in (20,40)}
        report[key]['moments']=total
    (folder/'truth.json').write_text(json.dumps(truth,indent=2))
    (folder/'alignment.json').write_text(json.dumps(report,indent=2))
    print(f"{folder.name}: {sum(r['usable'] for r in reports)}/{len(reports)} usable systems, {len(truth)} events, {report['trustworthy_seconds']:.2f}s",flush=True)
    return truth,report
