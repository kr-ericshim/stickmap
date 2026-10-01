"""Offline resumable harvest reader. Disabled channels require --validate.

No source is deleted. Each attempt is isolated and atomically published; old
attempts are retained. The fetcher's append-only journal is never rewritten.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
# Matches fill_seq.TIMING, used by tools/finetune/dense.py. Phase, monotonicity
# and every notation/staff failure are deliberately outside this set.
TIMING_ONLY_REASONS=('held-out agreement','fitted attack agreement','ambiguous alignment','no-cursor alignment')
sys.path.insert(0,str(HERE.parent/'harvest'))
from manifest_io import read as manifest_read, append as manifest_append
from layouts import channel_profile


def dump(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.'+uuid.uuid4().hex[:10]+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def song_path(root,row):
    slug=row.get('slug')
    if not isinstance(slug,str) or not slug or Path(slug).name!=slug or slug in ('.','..'):
        raise ValueError('unsafe/missing song slug')
    p=(root/'songs'/slug).resolve()
    if p.parent!=(root/'songs').resolve():raise ValueError('song escapes harvest root')
    return p


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            yield False;return
        try:yield True
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def signature(song,profile):
    h=hashlib.sha256(json.dumps(profile,sort_keys=True).encode())
    for p in sorted(HERE.glob('*.py')):h.update(p.name.encode());h.update(p.read_bytes())
    for directory in ('templates','harvest_templates','marks','harvest_marks','ornaments'):
        for p in sorted((HERE/directory).rglob('*.png')):h.update(str(p.relative_to(HERE)).encode());h.update(p.read_bytes())
    for name in ('video.mp4','attacks.htdemucs.npy'):
        p=song/name
        if p.exists():h.update(f'{name}:{p.stat().st_size}:{p.stat().st_mtime_ns}'.encode())
    return h.hexdigest()


def scan_signature(song,layout,scroll_probe=False,y_min_fraction=0,reject_overlays=False,native_fallback=False):
    h=hashlib.sha256(str(layout).encode())
    for name in ('layouts.py','harvest_scan.py','heads.py'):
        h.update((HERE/name).read_bytes())
    if scroll_probe:h.update((HERE/'harvest_scroll.py').read_bytes())
    if y_min_fraction:h.update(str(y_min_fraction).encode())
    if reject_overlays:h.update(b'reject chromatic overlays')
    if native_fallback:h.update(b'native frame consensus after failed median')
    p=song/'video.mp4'
    h.update(f'{p.stat().st_size}:{p.stat().st_mtime_ns}'.encode())
    return h.hexdigest()


def summarize(folder):
    a=json.loads((folder/'alignment.json').read_text())
    b=json.loads((folder/'alignment.scoring.json').read_text())
    strict=json.loads((folder/'truth.json').read_text())
    scoring=json.loads((folder/'truth.scoring.json').read_text())
    timing_only=[s['system'] for s in a['systems'] if not s['usable'] and s['reasons']
                 and all(r.startswith(TIMING_ONLY_REASONS) for r in s['reasons'])]
    def counts(notes):
        c=Counter((f"hihat_{n.get('articulation','unverified')}" if n['inst']=='hihat' else
                   f"tom_pos_{n['pos']}" if n['inst']=='tom' else n['inst']) for n in notes)
        for flag in ('ghost','accent'):c[flag]=sum(bool(n.get(flag)) for n in notes)
        return dict(c)
    return dict(strict_systems=sum(s['usable'] for s in a['systems']),
                scoring_systems=sum(s['scoring_usable'] for s in b['systems']),
                systems=len(a['systems']),strict_seconds=a['trustworthy_seconds'],
                scoring_seconds=b['trustworthy_seconds'],strict_notes=len(strict),scoring_notes=len(scoring),
                strict_labels=counts(strict),scoring_labels=counts(scoring),
                timing_only_systems=len(timing_only),timing_only_system_ids=timing_only,
                rejection_counts=dict(Counter(r for s in a['systems'] for r in s['reasons'])))


def quality_report(folder,profile,stats):
    a=json.loads((folder/'alignment.scoring.json').read_text())
    lines=[f'# {profile["name"]}: harvest reading quality','',
           f'Channel enabled: **{profile.get("enabled",False)}**. Legend: {profile.get("legend_confidence","unverified")}.',
           'Scoring includes strict. Timing is held-out prediction against supplied class-agnostic attacks, not independent hand-timed acoustic truth.',
           'Unverified mappings/marks are excluded; false flags are not verified negatives. Native staff spacing below 8 px remains rejected.','',
           f'Strict: {stats["strict_seconds"]:.2f} s, {stats["strict_notes"]} notes. Scoring: {stats["scoring_seconds"]:.2f} s, {stats["scoring_notes"]} notes.','',
           f'Strict systems rejected for timing reasons only: {stats.get("timing_only_systems",0)}. Matches dense.py reason prefixes; its attack-assignment, verified-label and dense-run filters still apply.','',
           'Audio sanity (audio-sanity.json): '+('; '.join(stats.get('audio_flags') or [])or 'no flags')+'. Phase: notated kicks/snares must sit on low/mid onset bands better than every rhythmic alias (alignment.json `phase`).','',
           '| System | Moments | Mode | Held-out 20 / 40 ms | Kick/snare band agreement (best alias) | Strict / scoring | Reasons |',
           '|---|---:|---|---|---|---|---|']
    for s in a['systems']:
        q=s.get('held_out',{});p=' / '.join(f'{100*q[k]:.1f}%' for k in ('within_20ms','within_40ms')) if q else '—'
        ph=s.get('phase') or {}
        band=(f'{100*ph["pooled"]:.0f}% of {ph["count"]}'+(f' ({100*ph["best_shift_pooled"]:.0f}% at {ph["best_shift"]:+.2f} s)' if ph.get('best_shift') is not None else '')) if ph.get('pooled') is not None else '—'
        lines.append(f'| [{s["system"]:03d}](check/system_{s["system"]:03d}.png) | {s["moments"]} | {s.get("mode","unread")} | {p} | {band} | {s["usable"]} / {s["scoring_usable"]} | '+ '; '.join(s['reasons'])+' |')
    (folder/'QUALITY.md').write_text('\n'.join(lines)+'\n')


def worker(song,out,profile_path,reuse=None):
    import harvest_scan, strip, color_timing, tiers
    from align import build_truth
    profile=json.loads(profile_path.read_text());layout=profile.get('layout')
    out.mkdir(parents=True,exist_ok=True)
    scroll_probe=profile.get('probe_scrolling',layout in ('drumscribe','dailychops'))
    y_min=profile.get('staff_y_min_fraction',0)
    reject_overlays=profile.get('reject_chromatic_overlays',False)
    native_fallback=profile.get('native_frame_fallback',False)
    sig=scan_signature(song,layout,scroll_probe,y_min,reject_overlays,native_fallback)
    reused=False
    if reuse and (reuse/'result.json').exists():
        old=json.loads((reuse/'result.json').read_text())
        if old.get('scan_signature')==sig and (reuse/'scan.json').exists():
            for p in [reuse/'scan.json',*reuse.glob('system_*.png')]:shutil.copy2(p,out/p.name)
            reused=True
    if not reused:
        moving=False
        if scroll_probe:
            import harvest_scroll
            from layouts import detect
            geometry=detect(song/'video.mp4',layout,y_min)
            moving,probes=harvest_scroll.probe(song/'video.mp4',geometry)
            dump(out/'motion-probe.json',dict(scrolling=moving,probes=probes))
        if moving:
            data=harvest_scroll.write(song/'video.mp4',out,geometry,reject_overlays=reject_overlays)
            if not profile.get('scrolling_verified'):
                for r in data['records']:r['scroll_reasons'].append('scrolling reading not validated for this channel')
                dump(out/'scan.json',data)
        else:harvest_scan.write(song/'video.mp4',out,layout,y_min,
                               reject_overlays=reject_overlays,native_fallback=native_fallback)
    systems=strip.read(out,profile_path)
    # Detector-free audio sanity: a near-silent stem or an attack deficit is
    # recorded with the export; the alignment gates still decide every system.
    import sanity
    audio=sanity.check(song,systems);dump(out/'audio-sanity.json',audio)
    if profile.get('color_notes') and not json.loads((out/'scan.json').read_text()).get('reconstructed_scroll'):
        color_timing.attach(out,page_notes=profile.get('page_color_notes',False))
    build_truth(song,out);tiers.write(out)
    stats=summarize(out);stats['audio_flags']=audio['flags'];quality_report(out,profile,stats)
    systems=json.loads((out/'strip.json').read_text())
    dump(out/'label-coverage.json',dict(
        limitation='Partial positive labels. Unverified/missed heads and false flags are not negative training examples. Generic full-family P/R on these intervals can penalize unlabelled instruments; use coverage masks.',
        verified_flags=profile.get('verified_flags',[]),
        verified_rules=[k for k,v in profile['rules'].items() if v.get('verified')],
        systems=[dict(system=s['system'],excluded_heads=[dict(x=n['x'],y=n['y'],pos=n['pos'],shape=n['shape'])
                 for n in s['notes'] if not n.get('exportable',True)]) for s in systems]))
    dump(out/'legend.json',profile)
    dump(out/'result.json',dict(complete=True,signature=signature(song,profile),scan_signature=sig,
                               channel=profile['name'],enabled=profile.get('enabled',False),**stats))


def process(root,row,profile,path,timeout):
    song=song_path(root,row)
    with lock(song/'.truth.lock') as acquired:
        if not acquired:return None
        # Re-read after lock acquisition to avoid a second worker duplicating a
        # completed attempt. A stale fetched snapshot never overwrites read.
        current=manifest_read(root/'manifest.jsonl').get(row['video'],{})
        if current.get('status')=='compacted':return None
        sig=signature(song,profile);out=song/'truth'
        if (out/'result.json').exists():
            old=json.loads((out/'result.json').read_text())
            if old.get('complete') and old.get('signature')==sig:
                result=old
                status='read' if result['scoring_systems'] and result['scoring_notes'] else 'unreadable'
                manifest_append(root/'manifest.jsonl',video=row['video'],status=status,
                                truth_signature=sig,truth_reason=None if status=='read' else result.get('rejection_counts',{}))
                return result
        for name in ('video.mp4','attacks.htdemucs.npy'):
            if not (song/name).is_file():
                manifest_append(root/'manifest.jsonl',video=row['video'],status='unreadable',truth_reason=f'missing {name}')
                return None
        attempt=song/('.truth-work-'+uuid.uuid4().hex[:12]);attempt.mkdir()
        snapshot=attempt/'profile.json';dump(snapshot,profile)
        args=[sys.executable,str(HERE/'batch.py'),'--worker',str(song),str(attempt),str(snapshot)]
        if out.exists():args.extend(['--reuse',str(out)])
        print(f'[{row.get("channel")}] {row["video"]}: reading',flush=True)
        try:
            with (attempt/'run.log').open('w') as log:
                run=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
            if run.returncode:
                detail=(attempt/'run.log').read_text(errors='replace').splitlines()
                raise RuntimeError(f'reader exit {run.returncode}: {detail[-1] if detail else "no log"}; see run.log')
            result=json.loads((attempt/'result.json').read_text())
            # Worker fingerprints its immutable legend snapshot, not live JSON.
            if result['signature']!=sig:raise RuntimeError('reader inputs changed during attempt; retry required')
            live,_=channel_profile(row.get('channel'))
            if live!=profile:raise RuntimeError('channel profile changed during attempt; retry required')
        except (subprocess.TimeoutExpired,RuntimeError,ValueError) as e:
            dump(attempt/'failure.json',dict(reason=str(e)))
            # Publish an explicit empty diagnostic export, retaining failed
            # candidate files and prior successful attempts for review.
            for name in ('truth.json','truth.scoring.json','truth.additional.json','alignment.json','alignment.scoring.json'):
                if (attempt/name).exists():os.rename(attempt/name,attempt/('candidate.'+name))
            empty=dict(version=2,systems=[],intervals=[],trustworthy_seconds=0)
            for name in ('alignment.json','alignment.scoring.json'):dump(attempt/name,empty)
            for name in ('truth.json','truth.scoring.json','truth.additional.json'):dump(attempt/name,[])
            result=dict(complete=False,signature=sig,channel=profile['name'],enabled=profile.get('enabled',False),
                        systems=0,strict_systems=0,scoring_systems=0,strict_seconds=0,scoring_seconds=0,
                        strict_notes=0,scoring_notes=0,strict_labels={},scoring_labels={},rejection_counts={str(e):1})
            dump(attempt/'result.json',result);dump(attempt/'legend.json',profile)
            (attempt/'QUALITY.md').write_text('# Unreadable attempt\n\n'+str(e)+'\n\nNo truth exported. See run.log and failure.json. Prior attempts retained in truth-history/.\n')
        if out.exists():
            history=song/'truth-history';history.mkdir(exist_ok=True)
            os.rename(out,history/attempt.name.removeprefix('.truth-work-'))
        os.rename(attempt,out)
        status='read' if result['scoring_systems'] and result['scoring_notes'] else 'unreadable'
        manifest_append(root/'manifest.jsonl',video=row['video'],status=status,truth_signature=sig,
                        truth_reason=None if status=='read' else (result['rejection_counts'] or {'no readable systems':1}),
                        truth_strict_systems=result['strict_systems'],truth_scoring_systems=result['scoring_systems'],
                        truth_audio_flags=result.get('audio_flags',[]))
        print(f'  {status}: strict {result["strict_seconds"]:.2f}s/{result["strict_notes"]} notes; scoring {result["scoring_seconds"]:.2f}s/{result["scoring_notes"]}',flush=True)
        return result


def report(root):
    rows=manifest_read(root/'manifest.jsonl');channels={}
    config=json.loads((HERE.parent/'harvest/channels.json').read_text())['channels']
    for c in config:
        profile,_=channel_profile(c['name'])
        channels[c['name']]=dict(layout=c['layout'],enabled=profile.get('enabled',False),
            legend_confidence=profile.get('legend_confidence','unverified'),reason=profile.get('reason',''),
            available=0,attempted=0,read=0,strict_systems=0,scoring_systems=0,systems=0,
            strict_seconds=0.,scoring_seconds=0.,strict_notes=0,scoring_notes=0,
            strict_labels=Counter(),scoring_labels=Counter(),rejection_counts=Counter(),audio_flags=Counter(),audio_flagged=[])
    for row in rows.values():
        c=channels.get(row.get('channel'))
        if c is None:continue
        if row.get('status') in ('fetched','read','unreadable','compacted'):c['available']+=1
        try:p=song_path(root,row)/'truth/result.json'
        except ValueError:continue
        if not p.exists():continue
        r=json.loads(p.read_text());c['attempted']+=1;c['read']+=bool(r['scoring_notes'])
        for k in ('strict_systems','scoring_systems','systems','strict_seconds','scoring_seconds','strict_notes','scoring_notes'):c[k]+=r[k]
        for k in ('strict_labels','scoring_labels','rejection_counts'):c[k].update(r[k])
        if r.get('audio_flags'):
            c['audio_flags'].update(r['audio_flags']);c['audio_flagged'].append(row['video'])
    audit_path=root/'validation/audit/accuracy.json'
    audits=json.loads(audit_path.read_text()) if audit_path.exists() else {}
    lines=['# Harvest channel validation','',
        'Scoring counts include strict. Zero exported truth for disabled channels. A missing audit is not 0% accuracy. Labels describe notation conventions; physical kit and original performance fidelity are not independently verified.','',
        '| Channel | Layout | Enabled / legend confidence | Songs attempted / available (read) | Systems strict / scoring / found | Seconds strict / scoring | Notes strict / scoring | Audited labels |',
        '|---|---|---|---|---|---|---|---|']
    for name,c in channels.items():
        audit=audits.get(name);c['audit']=audit
        accuracy=f'{audit["correct"]}/{audit["inspected"]}' if audit else 'unverified'
        lines.append(f'| {name} | {c["layout"]} | {c["enabled"]} / {c["legend_confidence"]} | {c["attempted"]}/{c["available"]} ({c["read"]}) | {c["strict_systems"]}/{c["scoring_systems"]}/{c["systems"]} | {c["strict_seconds"]:.2f}/{c["scoring_seconds"]:.2f} | {c["strict_notes"]}/{c["scoring_notes"]} | {accuracy} |')
        print(f'{name:14} {c["attempted"]}/{c["available"]} attempted; strict {c["strict_seconds"]:.2f}s/{c["strict_notes"]}; scoring {c["scoring_seconds"]:.2f}s/{c["scoring_notes"]}; {dict(c["scoring_labels"])}',flush=True)
    labels=['kick','snare','tom_pos_3','tom_pos_7','hihat_closed','hihat_open','hihat_pedal','crash','ride','ride_bell','side_stick','ghost','accent']
    lines+=['','Scoring-tier label counts (tom positions are retained, not guessed sizes):','',
            '| Channel | '+' | '.join(labels)+' |','|---|'+'---:|'*len(labels)]
    for name,c in channels.items():lines.append('| '+name+' | '+' | '.join(str(c['scoring_labels'].get(k,0)) for k in labels)+' |')
    for name,c in channels.items():
        lines+=['',f'## {name}',c['reason'],f'Rejections: {dict(c["rejection_counts"])}']
        if c['audio_flags']:lines.append(f'Audio sanity flags: {dict(c["audio_flags"])} on {c["audio_flagged"]}')
        if c['audit']:lines.append('Per-label visual audit: '+json.dumps(c['audit'].get('labels',{}),ensure_ascii=False))
    dump(root/'channel-results.json',channels)
    (root/'CHANNEL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    return channels


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'truth/harvest')
    p.add_argument('--only',nargs='+');p.add_argument('--validate',action='store_true',help='attempt disabled channels; exports remain gated')
    p.add_argument('--per-channel',type=int,help='maximum songs per channel in this invocation')
    p.add_argument('--limit',type=int);p.add_argument('--watch',action='store_true');p.add_argument('--poll',type=float,default=60)
    p.add_argument('--timeout',type=float,default=1200);p.add_argument('--min-free-gb',type=float,default=2)
    p.add_argument('--retry-unreadable',action='store_true');p.add_argument('--report',action='store_true')
    p.add_argument('--refresh-read',action='store_true',help='rebuild read exports only when code, input, or legend fingerprint changed')
    p.add_argument('--worker',nargs=3,type=Path,help=argparse.SUPPRESS);p.add_argument('--reuse',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker:worker(*a.worker,a.reuse);return
    if a.report:report(a.root);return
    attempted=set();counts=Counter()
    try:
        while True:
            if a.limit is not None and len(attempted)>=a.limit:report(a.root);return
            rows=manifest_read(a.root/'manifest.jsonl')
            # Round-robin by channel makes validation diverse while fetch grows.
            queues=defaultdict(list)
            for row in rows.values():
                if row['video'] in attempted:continue
                if row.get('status')!='fetched' and not(a.retry_unreadable and row.get('status')=='unreadable') and not(a.refresh_read and row.get('status')=='read'):continue
                name=row.get('channel')
                if a.only and name not in a.only:continue
                profile,path=channel_profile(name)
                if not(profile.get('enabled') or a.validate):continue
                queues[name].append((row,profile,path))
            while queues:
                for name in list(queues):
                    if not queues[name] or (a.per_channel is not None and counts[name]>=a.per_channel):
                        del queues[name];continue
                    if a.limit is not None and len(attempted)>=a.limit:report(a.root);return
                    if shutil.disk_usage(a.root).free/1e9<a.min_free_gb:
                        print('Disk reserve reached; no deletion performed',flush=True);report(a.root);return
                    row,profile,path=queues[name].pop(0)
                    # Reload profile before each song so disabling is immediate.
                    profile,path=channel_profile(name)
                    if not(profile.get('enabled') or a.validate):continue
                    attempted.add(row['video']);counts[name]+=1
                    try:
                        outcome=process(a.root,row,profile,path,a.timeout)
                        if outcome is None and manifest_read(a.root/'manifest.jsonl').get(row['video'],{}).get('status')=='fetched':
                            # Another reader held the lock. Retry on a later
                            # watch poll if it exits without committing status.
                            attempted.discard(row['video']);counts[name]-=1
                    except (OSError,ValueError) as e:print(f'{row["video"]}: {e}',file=sys.stderr,flush=True)
            report(a.root)
            if not a.watch:return
            time.sleep(max(1,a.poll))
    except KeyboardInterrupt:
        print('Stopped; incomplete attempts retained and no terminal status committed',flush=True)


if __name__=='__main__':main()
