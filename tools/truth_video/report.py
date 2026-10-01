"""Report selection as well as fit quality so rejected time never looks like verified truth."""
import json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from align import quality
from audit import label


def pair(q):
    return f"{100*q['within_20ms']:.1f}% / {100*q['within_40ms']:.1f}%" if q else '—'


def weighted(systems,key):
    chosen=[s for s in systems if key in s]
    n=sum(s['moments'] for s in chosen)
    return {f'within_{ms}ms':sum(s[key][f'within_{ms}ms']*s['moments'] for s in chosen)/n for ms in (20,40)} if n else None


def reading_report(root):
    folder=root/'audit'
    samples=json.loads((folder/'samples.json').read_text());manual=json.loads((folder/'manual.json').read_text())
    scores=defaultdict(lambda:dict(verified=0,correct=0,unconfirmed=[]))
    for q in samples:
        pred=label(q['predicted']);r=scores[pred];r['verified']+=1
        r['correct']+=manual[str(q['id'])]['label']==pred
        if manual[str(q['id'])]['label']!=pred:r['unconfirmed'].append(q['id'])
    old=json.loads((folder/'flag_samples.json').read_text());actual=json.loads((folder/'flag_manual.json').read_text())
    flag_scores=defaultdict(lambda:dict(original_positives=0,original_confirmed=0,retained_positives=0,retained_confirmed=0))
    for q in old:
        r=flag_scores[q['flag']];confirmed=actual[str(q['id'])]['flag_present']
        r['original_positives']+=1;r['original_confirmed']+=confirmed
        systems=json.loads((root/q['video']/'strip.json').read_text())
        notes=next(s for s in systems if s['system']==q['system'])['notes']
        near=[n for n in notes if abs(n['x']-q['predicted']['x'])<2 and abs(n['y']-q['predicted']['y'])<2]
        if near and near[0].get(q['flag']):r['retained_positives']+=1;r['retained_confirmed']+=confirmed
    data=dict(heads=dict(scores),flags=dict(flag_scores),head_count=len(samples),confirmed=sum(s['correct'] for s in scores.values()))
    (folder/'accuracy.json').write_text(json.dumps(data,indent=2))
    return data


def write(root):
    root=Path(root);videos=[]
    if (root/'jd-run/SUMMARY.md.before-jd').exists():
        from jd_report import build
        return build(root)
    for p in sorted(root.glob('*/alignment.json')):
        folder=p.parent;r=json.loads(p.read_text());systems=r['systems'];accepted=[s for s in systems if s['usable']]
        events=json.loads((folder/'truth.json').read_text());strip=json.loads((folder/'strip.json').read_text())
        before=None;old_moments=0
        old=root/'baseline'/folder.name/'truth.json'
        if old.exists():
            old_events=json.loads(old.read_text());times=sorted({round(n['time'],3) for n in old_events})
            before=quality(times,np.load(Path(r['song'])/'attacks.htdemucs.npy'));old_moments=len(times)
        video=dict(name=folder.name,before=before,old_moments=old_moments,raw=r['before'],after=r['after'],
                   accepted_quality=weighted(accepted,'after'),held_out=weighted(accepted,'held_out'),
                   seconds=r['trustworthy_seconds'],systems=len(systems),accepted=len(accepted),
                   moments=sum(s['moments'] for s in accepted),events=len(events),
                   counts=dict(Counter(label(n) for n in events)),
                   read_counts=dict(Counter(label(n) for s in strip for n in s['notes'])),
                   flags={k:sum(bool(n.get(k)) for n in events) for k in ('ghost','accent','flam','roll')},
                   dropped=[dict(system=s['system'],reasons=s['reasons']) for s in systems if not s['usable']])
        videos.append(video)
        lines=[f'# {folder.name}: system quality','',
               '20 / 40 ms columns use one-to-one matches to class-agnostic attacks. “After” is fitted agreement, not an independent accuracy measurement. Held-out moments were excluded from offset fitting and warp anchors. Dropped systems contribute no events or scoring interval.','',
               '| System | On-screen / active interval (s) | Mode | Moments | Initial 20 / 40 | Fitted 20 / 40 | Held-out 20 / 40 | Status / reasons |',
               '|---|---|---|---:|---|---|---|---|']
        for s in systems:
            lines.append(f"| [{s['system']:03d}](check/system_{s['system']:03d}.png) | {s['interval'][0]:.2f}–{s['interval'][1]:.2f} | {s.get('mode','unread')} | {s['moments']} | {pair(s.get('before'))} | {pair(s.get('after'))} | {pair(s.get('held_out'))} | {'KEEP' if s['usable'] else '; '.join(s['reasons'])} |")
        lines+=['','## Reading warnings','']
        for s in strip:
            if s['staff']['warnings']:lines.append(f"- System {s['system']:03d}: "+'; '.join(s['staff']['warnings']))
        (folder/'QUALITY.md').write_text('\n'.join(lines)+'\n')
    audit=reading_report(root)
    report=dict(videos=videos,trustworthy_seconds=sum(v['seconds'] for v in videos),events=sum(v['events'] for v in videos),audit=audit)
    (root/'summary.json').write_text(json.dumps(report,indent=2))
    lines=['# Transcription ground truth rebuild','',
           f"**{report['trustworthy_seconds']:.2f} seconds, {report['events']} exported notes**, in {sum(v['accepted'] for v in videos)} accepted systems. All 11 requested videos were rebuilt. This is the duration passing the stated timing and staff checks; it is not independently hand-timed audio truth.",'',
           'No network, detector drum labels, or port 8765 were used for alignment. Existing activations are used only by the separate detector evaluation. The original six truth exports and eight scans/strips were saved under `baseline/` before rebuilding.','',
           '## Timing before / after','',
           'Before is the preserved old truth (old global dx/dt fit). New fitted agreement includes **all readable candidate systems**, including rejected ones. Accepted agreement and held-out prediction refer only to the exported subset. Denominators differ: new cymbal moments are included, simultaneous heads are clustered, and rejected systems are excluded from export. A high fitted score alone never grants acceptance.','',
           '| Video | Old moments | Before 20 / 40 ms | New candidate moments | New fitted 20 / 40 ms | Exported 20 / 40 ms | Held-out 20 / 40 ms | Systems kept / read | Seconds |',
           '|---|---:|---|---:|---|---|---|---:|---:|']
    for v in videos:
        lines.append(f"| [{v['name']}]({v['name']}/QUALITY.md) | {v['old_moments'] or '—'} | {pair(v['before'])} | {v['after']['moments']} | {pair(v['after'])} | {pair(v['accepted_quality'])} | {pair(v['held_out'])} | {v['accepted']}/{v['systems']} | {v['seconds']:.2f} |")
    lines+=['','Acceptance requires a valid staff, at least six moments, strictly increasing fitted times, fitted agreement ≥85% at 20 ms and ≥95% at 40 ms, held-out agreement ≥55% / 80%, and at least 15 percentage points over shifted controls at either tolerance. No-cursor systems additionally need ≥70% / 90% held-out agreement. These fixed gates deliberately favor rejecting doubtful systems. The controls measure density/alias risk; they are not proof of the correct rhythmic phase.','',
            '## Exported label counts','']
    labels=['kick','snare','tom','hihat_closed','hihat_open','hihat_pedal','ride','ride_bell','crash','side_stick','cymbal_unknown','unknown']
    lines+=['| Video | '+' | '.join(labels)+' |','|---|'+'---:|'*len(labels)]
    for v in videos:lines.append('| '+v['name']+' | '+' | '.join(str(v['counts'].get(l,0)) for l in labels)+' |')
    lines+=['','| Video | Ghost | Accent | Flam | Roll |','|---|---:|---:|---:|---:|']
    for v in videos:lines.append('| '+v['name']+' | '+' | '.join(str(v['flags'][l]) for l in ('ghost','accent','flam','roll'))+' |')
    lines+=['','## All read labels, including rejected timing','',
            'These counts are for notation inspection, not usable timed truth. Side stick and the one clear roll were read even though their systems failed timing gates.','',
            '| Video | '+' | '.join(labels)+' |','|---|'+'---:|'*len(labels)]
    for v in videos:lines.append('| '+v['name']+' | '+' | '.join(str(v['read_counts'].get(l,0)) for l in labels)+' |')
    lines+=['','## Visual reading verification','',
            f"Inspected **{audit['head_count']} sampled heads** from enlarged source crops and numbered overlays. {audit['confirmed']}/{audit['head_count']} ({100*audit['confirmed']/audit['head_count']:.2f}%) had a confirmed matching label; one stacked cymbal glyph was left ambiguous. This is sampled label correctness among detected heads, **not recall or an independent human audit**. No detector output was consulted. Sampling is stratified, so the pooled rate is not a prevalence-weighted estimate.",'',
            'The reproducible sample is `audit/samples.json` (seed 20261001); `audit/blind_01.png` through `blind_07.png` hide predicted labels. Full-height `details_01.png` through `details_05.png` resolve staff positions and o marks outside smaller crops. `manual.json` records visual judgments. Template source systems were excluded from the head sample. `accuracy.json` contains counts. The ambiguous native glyph is enlarged in `stacked_cross.png`; its Tension system 000 is not exported.','',
            '| Predicted label | Confirmed / inspected | Correctness |','|---|---:|---:|']
    for lab,a in sorted(audit['heads'].items()):lines.append(f"| {lab} | {a['correct']}/{a['verified']} | {100*a['correct']/a['verified']:.1f}% |")
    lines+=['','A separate 37-positive flag audit found adjacent ink mistaken for parentheses, beamed small glyphs mistaken for graces, and one accent whose note ownership was unclear. Parenthesis/grace templates and a conservative accent distance limit now suppress those cases. The retained counts below are a **regression recheck on the same samples**, not a fresh validation set; flag recall was not measured.','',
            '| Flag | Original confirmed / sampled | Retained confirmed / positive after fixes |','|---|---:|---:|']
    for flag,a in audit['flags'].items():lines.append(f"| {flag} | {a['original_confirmed']}/{a['original_positives']} | {a['retained_confirmed']}/{a['retained_positives']} |")
    lines+=['','## Dropped systems','']
    for v in videos:
        ids=', '.join(f"{s['system']:03d}" for s in v['dropped']) or 'none'
        reasons=Counter(reason for s in v['dropped'] for reason in s['reasons'])
        lines.append(f"- **{v['name']}**: {ids}. [Every system and exact reason]({v['name']}/QUALITY.md). "+'; '.join(f'{reason} ({n})' for reason,n in reasons.items()))
    lines+=['','## Limits and unreadable material','',
            '- Benny Greb and Larnell Lewis have no reliable cursor. Interval/x-based alignment runs, but none of their systems pass held-out validation. Their `truth.json` files are intentionally empty; inspection labels remain in `strip.json`.',
            '- Michael Alba has readable notation and a box cursor, but its timing correspondence fails validation throughout. It is not safe ground truth yet. It needs an independently checked rhythmic/time anchor or a richer rhythm model; increasing snap radius would conceal the problem.',
            '- CJ Thompson is natively 360×640 with roughly 4 px staff spacing. It is segmented into page rows, and its faint red cursor is followed across rows. Upscaled row crops run through the same head reader but retain an explicit native-resolution rejection. A genuine 720×1280 source would reach the 8 px minimum at the same framing; **1080×1920 is recommended** for about 12 px spacing. Upscaling this download is not sufficient.',
            '- Unmapped cymbal positions stay `cymbal_unknown`, with position and low confidence; they are not guessed as splash/china. Layout JSON maps notation conventions, not proof of the physical instrument played. Filled round heads outside the supported map stay unknown. Triangle cymbals, ambiguous stacked crosses, tied/continuing rolls, and ghost parentheses around non-filled heads are not comprehensively read.',
            '- Open/closed state is local to a printed mark; an unmarked hat defaults to closed. Articulation is not inferred from audio or propagated beyond the marked note. Explicit + and o override the default, and circled hats are open.',
            '- Accent/flam/roll flags are conservative detections, not exhaustive negatives. The clear three-slash roll is supported; other slash/buzz glyphs are not validated. Flam main-note time is exported; grace strokes are not assigned separate invented onset times.',
            '- Timing has been checked against provided attack times and held-out predictions, not manual listening/onset annotation. Some soft notated hits have no attack; some attacks are spurious. Rejected regions are excluded from precision/recall, and the family scores on selected regions are not whole-song detector scores.',
            '- Regression checks and actual detector scoring logs are in `tests.log` and each accepted video’s `evaluation.txt`. Rebuild commands, legends, confidence semantics, and implementation details are in `../tools/truth_video/README.md`.']
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(f"{report['trustworthy_seconds']:.2f}s, {report['events']} events")


if __name__=='__main__':
    import sys
    write(sys.argv[1] if len(sys.argv)>1 else 'truth')
