"""Count JD notation correspondences for actual pre/post tom-processing hits.

Run with PYTHONPATH pointing at the immutable local detector snapshot. No model
inference or network access: evaluate.detect uses the supplied activation caches.
"""
import argparse
import contextlib
import hashlib
import io
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import numpy as np
from evaluate import detect, score, confusion, family
from align import match


def inside(t, intervals):
    return any(a-.05 <= t <= b+.05 for a,b in intervals)


def labels_at(t, truth):
    labels=set()
    for n in truth:
        if abs(n['time']-t) <= .05:
            inst=n['inst']
            labels.add('bell' if inst in ('bell','ride_bell') else
                       'snare' if inst=='side_stick' else inst)
    return sorted(labels) or ['nothing']


def notation_at(t,truth):
    return [{k:n.get(k) for k in ('time','inst','shape','pos','ghost','system')}
            for n in truth if abs(n['time']-t)<=.05]


def bin_pitch(p):
    if p is None: return 'unresolved'
    if p < 105: return '<105 Hz'
    if p < 150: return '105-150 Hz'
    if p < 300: return '150-300 Hz'
    return '>=300 Hz'


def summarize_candidates(rows):
    by_share=defaultdict(Counter);by_pitch=defaultdict(Counter);cross=defaultdict(Counter)
    for r in rows:
        outcome='+'.join(r['truth_labels'])
        by_share[r['share_bin']][outcome]+=1
        by_pitch[r['pitch_bin']][outcome]+=1
        cross[r['share_bin']+' / '+r['pitch_bin']][outcome]+=1
    return dict(count=len(rows),by_share=dict(by_share),by_pitch=dict(by_pitch),share_pitch=dict(cross))


def run(song,folder):
    from stickmap import toms
    pre=detect(song,stage='pre-toms')
    # Instrument the real final stage, recording precisely the calls it makes.
    actual_pitch={};original_pitch=toms._pitch
    def capture(stem,time,next_tom,others):
        p=original_pitch(stem,time,next_tom,others);actual_pitch[time]=p;return p
    toms._pitch=capture
    try: post=toms.tom_sizes(song/'drums.htdemucs.wav',pre)
    finally: toms._pitch=original_pitch
    stem=toms._Stem(song/'drums.htdemucs.wav')
    filtered=toms._without_bells(stem,pre)
    others=toms._other_drums(stem,filtered)
    all_toms=[h for h in pre if h.inst=='tom']
    candidates=[]
    for i,h in enumerate(all_toms):
        struck=stem.struck(h.time,toms._RING[0])
        head=float(struck[(stem.freqs>=toms._HEAD[0])&(stem.freqs<toms._HEAD[1])].sum())
        heard=float(struck[(stem.freqs>=toms._HEARD[0])&(stem.freqs<toms._HEARD[1])].sum())
        share=head/heard if heard else None
        actual=h.time in actual_pitch
        following=next((q.time for q in all_toms[i+1:] if q.time>h.time+.025),None)
        p=actual_pitch[h.time] if actual else original_pitch(stem,h.time,following,others)
        final=[q.inst for q in post if abs(q.time-h.time)<1e-7 and q.inst.startswith(('tom','ride_bell'))]
        candidates.append(dict(time=h.time,head_share=share,
                               share_bin='<0.1' if share is not None and share<.1 else '0.1-0.3' if share is not None and share<.3 else '>=0.3' if share is not None else 'no spectrum',
                               pitch_hz=p,pitch_bin=bin_pitch(p),pitch_source='actual' if actual else 'counterfactual_removed_before_pitch',
                               final=final or ['removed']))
    (folder/'detector-hits.json').write_text(json.dumps([asdict(h) for h in post],indent=2))
    (folder/'tom-candidates.all.json').write_text(json.dumps(candidates,indent=2))
    result={}
    for tier in ['strict','scoring']:
        suffix='' if tier=='strict' else '.scoring'
        truth=json.loads((folder/f'truth{suffix}.json').read_text())
        alignment=json.loads((folder/f'alignment{suffix}.json').read_text())
        intervals=alignment['intervals']
        if not truth:
            result[tier]={'notes':0,'seconds':0,'scores':{},'tom_candidates':summarize_candidates([])}
            continue
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream): out=score(truth,post,'current detector',intervals)
        scores={}
        for inst,(pairs,ref,est) in out.items():
            scores[inst]=dict(tp=len(pairs),predicted=len(est),reference=len(ref),
                              precision=len(pairs)/len(est) if est else None,
                              recall=len(pairs)/len(ref) if ref else None)
        sn_pairs,sn_ref,sn_est=out['snare'];got={i for i,j in sn_pairs}
        ghosts=[i for i,n in enumerate(sn_ref) if n.get('ghost')]
        ghost_table=Counter(f'truth={sn_ref[i]["ghost"]},detector={sn_est[j].ghost}' for i,j in sn_pairs)
        rows=[{**r,'truth_labels':labels_at(r['time'],truth),'notation':notation_at(r['time'],truth)} for r in candidates if inside(r['time'],intervals)]
        bells=[{**asdict(h),'truth_labels':labels_at(h.time,truth),'notation':notation_at(h.time,truth)} for h in post if h.inst=='ride_bell' and inside(h.time,intervals)]
        lows=[{**asdict(h),'truth_labels':labels_at(h.time,truth),'notation':notation_at(h.time,truth)} for h in post if h.inst=='tom_low' and inside(h.time,intervals)]
        misses=[]
        for i,n in enumerate(sn_ref):
            if i in got:continue
            labels=sorted({h.inst for h in post if abs(h.time-n['time'])<=.05}) or ['nothing']
            misses.append(dict(time=n['time'],ghost=n['ghost'],detected=labels))
        result[tier]=dict(notes=len(truth),seconds=alignment['trustworthy_seconds'],scores=scores,
                          confusion={k:dict(v) for k,v in confusion(truth,post).items()},
                          ghost=dict(reference=len(ghosts),matched=sum(i in got for i in ghosts),agreement=dict(ghost_table)),
                          tom_candidates=summarize_candidates(rows),
                          bells=dict(count=len(bells),truth=dict(Counter('+'.join(r['truth_labels']) for r in bells))),
                          tom_low=dict(count=len(lows),truth=dict(Counter('+'.join(r['truth_labels']) for r in lows))),
                          snare_misses=dict(Counter('+'.join(r['detected']) for r in misses)))
        (folder/f'correspondence.{tier}.json').write_text(json.dumps(dict(toms=rows,bells=bells,tom_low=lows,snare_misses=misses),indent=2))
        (folder/f'detector-score.{tier}.txt').write_text(stream.getvalue())
    (folder/'diagnostics.json').write_text(json.dumps(result,indent=2))
    print(folder.name,{t:r['scores'] for t,r in result.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path)
    a=p.parse_args();run(Path((a.folder/'song.txt').read_text().strip()),a.folder)
