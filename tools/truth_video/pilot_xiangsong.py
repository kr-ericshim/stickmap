"""Offline sealed-VAULT native crop audits and artifact-derived pilot report.

Only notation, supplied audio/stem attacks and bands are read. No detector,
scorer, network or training admission runs. Manual decisions must be entered
after inspecting the native/reader pairs, never copied from predictions.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

import cv2
import numpy as np

from pilot_new import PILOTS, folders, HERE
from pilot_shorts import target_tile, blockers
from pilot_report import exported_fast
from strip import label_heads

ROOT = PILOTS / 'xiangsong'
PROFILE = HERE / 'legends/xiangsong.json'
KEYS = ('filled:1', 'filled:3', 'filled:5', 'filled:6', 'filled:7', 'filled:8',
        'cross:8', 'cross:9', 'cross:10', 'cross:-1', 'circled:9', 'diamond:8')


def pool():
    profile = json.loads(PROFILE.read_text())
    # This is a diagnostic flag prediction only, pending a separate crop audit.
    profile['verified_flags'] = ['ghost']
    entries = []
    ignored = {'no verified channel labels', 'channel disabled: reading / legend validation incomplete',
               'page row not visited by cursor'}
    for folder in folders(['xiangsong']):
        scan = json.loads((folder / 'notation/scan.json').read_text())
        for r in scan['records']:
            gray = cv2.imread(str(folder / f'notation/system_{r["system"]:03d}.png'), 0)
            try:
                notes, staff = label_heads(gray, profile)
            except ValueError:
                continue
            if any(q not in ignored for q in staff['reasons']): continue
            for n in notes:
                entries.append(dict(video=folder.name, head=n, staff=staff, record=r, scan=scan))
    return entries


def balanced(entries, quota, rng):
    entries = list(entries); rng.shuffle(entries); by = defaultdict(list)
    for e in entries: by[e['video']].append(e)
    result = []
    while any(by.values()) and len(result) < quota:
        for video in sorted(by):
            if by[video] and len(result) < quota: result.append(by[video].pop())
    return result


def audit(kind='labels', name=None, exclude=None):
    cv2.setNumThreads(1)
    out = ROOT / (name or f'audit-{kind}')
    if (out / 'samples.json').exists():
        raise FileExistsError(f'preserve existing audit: {out}')
    entries = pool(); rng = np.random.default_rng(20261003); chosen = []
    if exclude:
        previous=json.loads((ROOT/exclude/'samples.json').read_text())
        identities={(s['video'],s['system'],round(s['head']['x']),round(s['head']['y'])) for s in previous}
        entries=[e for e in entries if (e['video'],e['record']['system'],round(e['head']['x']),
                                       round(e['head']['y'])) not in identities]
    seen = set(); unique = []
    for e in entries:
        n = e['head']; r = e['record']
        identity = (e['video'], r['start'], n['shape'], n['pos'], round(n['x']), round(n['y']))
        if identity not in seen: unique.append(e); seen.add(identity)
    if kind == 'labels':
        for key in KEYS:
            chosen += balanced([e for e in unique if f'{e["head"]["shape"]}:{e["head"]["pos"]}' == key], 8, rng)
    else:
        snares = [e for e in unique if e['head']['shape'] == 'filled' and e['head']['pos'] == 5]
        for prediction in (True, False):
            chosen += balanced([e for e in snares if bool(e['head']['ghost']) == prediction], 24, rng)
    rng.shuffle(chosen); out.mkdir(parents=True, exist_ok=True); samples=[]; tiles=[]; caps={}
    try:
        for i, e in enumerate(chosen, 1):
            n, staff, r, scan = (e[k] for k in ('head', 'staff', 'record', 'scan'))
            gray = cv2.imread(str(ROOT / e['video'] / f'notation/system_{r["system"]:03d}.png'), 0)
            if e['video'] not in caps: caps[e['video']] = cv2.VideoCapture(scan['video'])
            cap = caps[e['video']]; frame_no = (r['start']+r['end'])//2
            cap.set(1, frame_no); ok, frame = cap.read()
            if not ok: raise ValueError('native frame unavailable')
            base = scan['top']+r['crop'][0]; x=n['x']+scan.get('left', 0); y=n['y']+base
            native = dict(frame=frame_no, x=x, y=y, lines=[a+base for a in staff['lines']])
            a=target_tile(frame, x, y, native['lines'], staff['spacing'])
            b=target_tile(gray, n['x'], n['y'], staff['lines'], staff['spacing'])
            cell=np.full((270,360,3),255,np.uint8)
            cv2.putText(cell, f'{i} {e["video"]} s{r["system"]}', (5,18),0,.45,(0,0,0),1)
            cv2.putText(cell, 'native                 reader', (5,40),0,.4,(0,0,0),1)
            cell[46:266,5:180]=a; cell[46:266,185:360]=b; tiles.append(cell)
            prediction = f'{n["shape"]}:{n["pos"]}' if kind == 'labels' else bool(n['ghost'])
            samples.append(dict(id=i, video=e['video'], system=r['system'], native=native,
                                spacing=staff['spacing'], head=n, predicted=prediction))
    finally:
        for cap in caps.values(): cap.release()
    for page in range((len(tiles)+11)//12):
        sheet=np.full((810,1440,3),255,np.uint8)
        for k, cell in enumerate(tiles[page*12:(page+1)*12]):
            row,col=divmod(k,4);sheet[row*270:(row+1)*270,col*360:(col+1)*360]=cell
        cv2.imwrite(str(out/f'blind_{page+1:02d}.png'),sheet)
    (out/'samples.json').write_text(json.dumps(samples,indent=2)+'\n')
    print(len(samples), 'native pairs:', out)


def audit_report(kind='labels', name=None):
    out=ROOT/(name or f'audit-{kind}')
    samples=json.loads((out/'samples.json').read_text());manual=json.loads((out/'manual.json').read_text())
    groups=defaultdict(Counter); errors=[]; confusion=Counter()
    for s in samples:
        actual=manual[str(s['id'])];prediction=s['predicted']
        if kind == 'ghost' and type(actual) is not bool: raise ValueError('ghost decisions must be explicit booleans')
        for key in (str(prediction),s['video']):
            groups[key]['inspected']+=1;groups[key]['correct']+=actual==prediction
        if actual!=prediction:errors.append(dict(id=s['id'],predicted=prediction,actual=actual))
        if kind=='ghost':confusion[('tp' if actual else 'fp') if prediction else ('fn' if actual else 'tn')]+=1
    result=dict(inspected=len(samples),correct=len(samples)-len(errors),groups={k:dict(v) for k,v in groups.items()},
                errors=errors,confusion=dict(confusion),
                scope='Assistant native-source/reader visual target audit; detected-head sample, not full-score recall, independent human review, physical cymbal key or acoustic fidelity')
    (out/'accuracy.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


def write_markdown(data):
    by={r['video']:r for r in data['videos']}
    ordered=[by[v] for v in ('Dd_S08chzqQ','Dd8wm3ThGTL','DMPkSQZyj76','DNz6yi-QMWp')]
    lines=['# Xiangsong 阿嵩鼓谱 @xiangsong54: offline sealed-VAULT pilots — 2026-10-03','',
        'The native vertical page reader and partial core/ghost legend are enabled. Routine fetching remains **disabled**: only 1/4 supplied reels has accepted scoring positives, and the current harvest fetch path invokes drum transcription. This channel is sealed VAULT test material: **never run a drum detector/scorer on it and never add it to `vault.json` train_channels**. The `scoring` name below refers only to the existing additive timing-reference tier; no detector evaluation ran.','',
        'Every input is the supplied local `video.mp4`, with embedded audio. Cached local Demucs weights separate a drum stem; missing weights fail instead of downloading. Only class-agnostic stem attacks and onset bands support timing. All artifacts remain under `.eval-cache/pilot_new/xiangsong/`; no canonical songs, manifest or training split was published. Preparation/read used two workers, with one OpenCV/Torch/BLAS thread per worker. One notation worker overlapped preparation, so at most three CPU workers ran. No network request ran.','',
        '## Per-reel results','',
        'Scoring includes strict. Found systems include repeated page visits, rejected fragments and unvisited rows; they are not unique musical systems. Clean means staff/reading gates pass before acoustic timing. Observed heads include repeated/unvisited candidates; only the strict/scoring columns are timed positive exports.','',
        '| Reel | Native / seconds | Found / clean | Observed heads | Native spacing | Strict / scoring systems | Strict / scoring notes | Strict / scoring seconds | Scoring fast kick / snare / tom | Label audit |',
        '|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for r in ordered:
        n=r['native'];fast=r['fast']['scoring']['core_neighbour'];audit=data['audits']['labels']['groups'][r['video']]
        folder=f'../../.eval-cache/pilot_new/xiangsong/{r["video"]}/{Path(r["artifact"]).name}'
        lines.append(f'| [{r["video"]}]({folder}/QUALITY.md) | {n["width"]}×{n["height"]} / {n["frames"]/n["fps"]:.3f} | {r["systems"]} / {r["clean_staff"]} | {r["observed_heads"]} | {r["spacing"][0]:.3f}–{r["spacing"][1]:.3f} px | {r["strict_systems"]} / {r["scoring_systems"]} | {r["strict_notes"]} / {r["scoring_notes"]} | {r["strict_seconds"]:.3f} / {r["scoring_seconds"]:.3f} | {fast["kick"]} / {fast["snare"]} / {fast["tom"]} | {audit["correct"]}/{audit["inspected"]} |')
    lines+=['',
        'Fast means distinct class/time positives with a nearest **core kick/snare/tom** neighbour gap strictly <90 ms inside each accepted system. Unisons and multiple toms at one time are deduplicated; rejected gaps are never bridged. All strict fast counts are zero. DNz6yi-QMWp has scoring same-class fast counts **0 kick / 11 snare / 0 tom**, separately from its 2/13/0 core-neighbour counts. A zero does not establish absence in the performance.','',
        'DNz6yi-QMWp scoring systems **16 and 23** contain **14 kick, 30 snare, 3 tom** notes and **9 confirmed snare ghost positives**. Their held-out 20/40 ms rates are **40.9/90.9%** and **48.4/96.8%**: both fail strict 20 ms >=55%, while passing the unchanged scoring, controls and band-phase gates. There is no strict ghost export. These are partial notation positives, not independent acoustic or complete transcription truth.','',
        '## Native layout and notation key','',
        'Ollie Woods Shorts page detection, native row crops, box-arrival priors, head-local colour timing and visited-row restrictions are reused. Xiangsong has its own `layout: xiangsong` and `cursor_style: blue_box`. Its four page tops are approximately **1200, 1199, 1085, 605 px** in the table order. DMPkSQZyj76 has seven visible rows in the sampled page; the other tall pages usually have six, while the short 1080×1166 page shows four complete rows and a partial bottom row. There is no hard-coded six-row assumption.','',
        'Dense DMPkSQZyj76 first-row beams and heads create >31 consecutive dark rows at the paper edge. The original Shorts survey therefore placed the boundary inconsistently at 1085–1153 px. This opt-in profile additionally checks the inspected white outer margins on **both** sides; all seven sampled boundaries then agree at 1085 px. Cropping uses source pixels without resizing. Per-row 8 px, line support, uneven spacing and printed paired percussion-clef gates remain mandatory; no clef-omission or scrolling waiver was added.','',
        'The blue cursor reader requires one tall, filled blue plate (>=5 native staff spaces) in the page. Blue heads/stems alone and multiple plate candidates fail closed. Thirty-one predictions match the real plate in 32 inspected frames; one real Dd8wm3ThGTL plate is missed. These spotchecks do not measure full-video recall or audio onset accuracy. [Source sheets and decisions](../../.eval-cache/pilot_new/xiangsong/cursor-audit-accuracy.json).','',
        'Positions are half-spaces above the bottom staff line. Native source/reader target decisions approve the following limited instrument convention:','',
        '| Shape / position | Convention | Correct / inspected | Export |',
        '|---|---|---:|---|']
    meanings={'filled:1':'kick','filled:3':'tom, staff position only','filled:5':'snare',
              'filled:6':'tom, staff position only','filled:7':'tom, staff position only',
              'filled:8':'observed diagonal slash misread as filled; physical identity unresolved',
              'cross:8':'upper cymbal candidate','cross:9':'hi-hat-position candidate',
              'cross:10':'upper cymbal candidate','cross:-1':'pedal-position candidate',
              'circled:9':'circled hi-hat-position candidate'}
    for key,meaning in meanings.items():
        row=data['audits']['labels']['groups'][key]
        approved=key in ('filled:1','filled:3','filled:5','filled:6','filled:7')
        lines.append(f'| {key} | {meaning} | {row["correct"]}/{row["inspected"]} | {"yes" if approved else "withheld"} |')
    lines+=['',
        '**Core rules: 40/40. Overall detected shape/position audit: 80/81.** The withheld DNz6yi-QMWp `filled:8` target is an angular slash glyph, so it must not become a tom. Upper crosses/circles have correct spatial samples but no printed complete physical-kit key; **hi-hat, open/closed/pedal articulation, crash, ride and bell names are not approved**. The audit answers where their glyphs are, without claiming which physical cymbal sounds. Physical tom sizes, accents, flams and rolls remain withheld.','',
        'Bar numbers, dense sextuplet "6" glyphs, ghost parentheses, ties/rests and 2/4↔4/4 changes remain visible in native crops. The harvest reader does not infer a fixed meter or manufacture note durations/hits from these marks. Timing comes from actual cursor/colour traversal and the audio gates. The crop audit samples detected heads; it does not establish whole-score recall, exhaustive ties/tuplets/meter transcription, independent human review or performance fidelity.','',
        '## Ghost-parenthesis audit','',
        'The generic initial 48-target check was first reported as 46/48. Wider original-frame inspection found two reviewer target errors: #35 followed the next parenthesized note instead of the red-marked ordinary snare, and #39 overlooked both parentheses. Initial decisions/results and the corrected 48/48 contextual result are all retained. **This correction is not fresh blind accuracy.** [Initial and contextual evidence](../../.eval-cache/pilot_new/xiangsong/audit-ghost/recheck.json).','',
        'The final positive reader uses four native-resolution source-rasterization templates and the existing complete-glyph/independent-side checks: **>=0.95 whole-glyph correlation and >=0.65 on each parenthesis side**, limited to filled snare position 5. A disjoint sample of 48 system/head target records confirms **24/24 predicted positives**, with **10 actual ghosts missed among 24 negative predictions**. Per reel, all 6 predicted positives are confirmed; missed ghosts in the six negative controls are 5, 1, 2 and 2 in table order. The balanced diagnostic sample is not a population recall estimate.','',
        'The final templates deliberately lose ghost yield. Only `ghost=true` is approved evidence; `ghost=false` is **not** a verified non-ghost stroke. No false ghost positive appeared in the fresh sample, which remains a small assistant within-source visual check. It is not independent human/audio verification. [Fresh audit](../../.eval-cache/pilot_new/xiangsong/audit-ghost-fresh/accuracy.json), [template provenance](../../tools/truth_video/harvest_templates/xiangsong/README.md).','',
        '## Timing and reading blockers','',
        'Each column counts systems with that strict rejection family, including unvisited diagnostic rows. Counts overlap and must not be added. The two accepted scoring rows still appear in held-out strict failures. Exact reasons, source crops, priors, fitted/held-out/control rates and phase aliases remain in each QUALITY.md/alignment JSON.','',
        '| Reel | Fitted | Held-out | Controls | No cursor | Band/phase | Unvisited | Clef | <8 px / geometry / overlay |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in ordered:
        b=r['blockers']
        lines.append(f'| {r["video"]} | {b["fitted"]} | {b["held_out"]} | {b["controls"]} | {b["no_cursor"]} | {b["phase"]} | {b["unvisited"]} | {b["clef"]} | {b["below8"]} / {b["geometry"]} / {b["overlay"]} |')
    lines+=['',
        'Even among clean visited staffs, fitted/held-out/control or instrument-band evidence usually fails. Clear notation and good spatial audits do not establish acoustic timing. All four stems pass detector-free audio sanity (no silent-stem/attack-deficit flags). Visual colour edges retain one-frame quantization at 30 fps (~33 ms), **not** total acoustic-onset uncertainty. Soft ghosts, dense attacks and rhythmic aliases are possible limitations, not proven causes of every rejection. No gate was relaxed to rescue yield.','',
        '## Listing, enablement and verification','',
        '`channels.json` has `name: xiangsong`, `listing: file`, `url_template: https://www.instagram.com/reel/{id}/`, `video_format: bv*[width<=1080]/b`, `sleep: 25`, the separate layout/legend and `reader_enabled: true`. `fetch_enabled` stays **false** for limited timed yield and the sealed detector-free boundary. Enabling ordinary fetch would execute `groovemap.cli transcribe` in the current fetcher. A source-only fetch path is owner work outside this task; no fetch/network command was run.','',
        'The current pre-existing list has **118 distinct rows, all NA duration**, not the 73 in the request. It was used as is and not edited or refreshed. NA remains unknown rather than an invented zero-duration filter. Only the four supplied media files were processed. Xiangsong is absent from vault train_channels, and `vault.json` is unchanged.','',
        'Native regression fixtures cover the dense boundary, real blue plate/ambiguity, snare positive/plain/one-parenthesis controls, withheld angular/cymbal rules, unchanged small-staff/clef gates, embedded-audio metadata and sealed fetch/training scope. Native fixture crops have SHA256/source-frame provenance. Export integrity checks find no nonfinite/unapproved/tied notes, ordering violations or strict/scoring subset failures; final positives come only from accepted systems.','',
        '```bash',
        '.venv/bin/python tools/truth_video/pilot_new.py prepare --channels xiangsong --jobs 2',
        '.venv/bin/python tools/truth_video/pilot_new.py notation --channels xiangsong --jobs 1',
        '.venv/bin/python tools/truth_video/pilot_xiangsong.py audit --kind labels',
        '# Inspect native/reader sheets and enter manual.json; preserve existing audits.',
        '.venv/bin/python tools/truth_video/pilot_xiangsong.py audit-report --kind labels',
        '.venv/bin/python tools/truth_video/pilot_xiangsong.py audit --kind ghost --name audit-ghost-fresh --exclude audit-ghost',
        '.venv/bin/python tools/truth_video/pilot_xiangsong.py audit-report --kind ghost --name audit-ghost-fresh',
        '.venv/bin/python tools/truth_video/pilot_new.py read --channels xiangsong --jobs 2 --output truth-reviewed',
        '.venv/bin/python tools/truth_video/pilot_xiangsong.py report --output truth-reviewed',
        '.venv/bin/python -m pytest -q tools/truth_video tools/harvest',
        '```','',
        'Existing completed reads are retained; choose a new `--output` for another attempt. Never combine independent runs beyond four active CPU workers. Missing Demucs weights cannot trigger a download. The prepare path extracts embedded MP4 audio itself when audio.m4a is absent; metadata falls back to the retained channel list and records `test_only: true`.','',
        f'**Required tests: {data["test_result"]}.** [Test log](../../.eval-cache/pilot_new/xiangsong/{data["test_log"]}). [Machine-readable results and export integrity](../../.eval-cache/pilot_new/xiangsong/results.json).']
    (HERE.parents[1]/'truth/harvest/XIANGSONG.md').write_text('\n'.join(lines)+'\n')


def report(output='truth'):
    rows=[];integrity=[]
    profile=json.loads(PROFILE.read_text())
    for folder in folders(['xiangsong']):
        out=folder/output;result=json.loads((out/'result.json').read_text())
        strip=json.loads((out/'strip.json').read_text());alignment=json.loads((out/'alignment.json').read_text())
        spacing=[s['staff']['spacing'] for s in strip if 'spacing' in s['staff']]
        exports={tier:json.loads((out/f'{name}.json').read_text())
                 for tier,name in (('strict','truth'),('scoring','truth.scoring'))}
        scoring_alignment=json.loads((out/'alignment.scoring.json').read_text())
        for tier,notes in exports.items():
            allowed={s['system'] for s in (alignment if tier=='strict' else scoring_alignment)['systems']
                     if s['usable' if tier=='strict' else 'scoring_usable']}
            assert all(np.isfinite(n['time']) and n.get('exportable') and not n.get('tied_continuation')
                       and n['system'] in allowed for n in notes), (folder,tier,'export integrity')
            assert all(a['time']<=b['time'] for a,b in zip(notes,notes[1:])), (folder,tier,'order')
            assert all(profile['rules'][f'{n["shape"]}:{n["pos"]}']['verified'] for n in notes)
            assert all(n['inst']=='snare' and n['pos']==5 and n['ghost_confidence']>=.95
                       for n in notes if n.get('ghost'))
        def key(n): return n['time'],n['inst'],n['pos'],n['shape'],n['x']
        assert {key(n) for n in exports['strict']}<={key(n) for n in exports['scoring']}
        integrity.append(dict(video=folder.name,strict_notes=len(exports['strict']),
                              scoring_notes=len(exports['scoring']),issues=[]))
        meta=json.loads((folder/'meta.json').read_text())
        rows.append(dict(video=folder.name,native=meta['native_video'],test_only=meta['test_only'],
                         **result,clean_staff=sum(not s['staff']['reasons'] for s in strip),
                         observed_heads=sum(len(s['notes']) for s in strip),
                         spacing=[min(spacing),max(spacing)] if spacing else [],
                         fast={tier:exported_fast(notes) for tier,notes in exports.items()},
                         blockers=blockers(alignment['systems']),artifact=str(out)))
    audits={kind:json.loads((ROOT/f'audit-{kind}/accuracy.json').read_text())
            for kind in ('labels','ghost','ghost-fresh')}
    data=dict(videos=rows,audits=audits,integrity=integrity,
              detector_execution=False,network_execution=False,
              fast_definition='Distinct class/time positives with a core kick/snare/tom neighbour gap <90 ms inside each accepted system; chords deduplicated; rejected gaps never bridged')
    logs=[p for p in (ROOT/'pytest-final.log',ROOT/'pytest.log') if p.exists()]
    summaries=[(p,re.search(r'\d+ passed in [\d.]+s',p.read_text())) for p in logs]
    completed=next(((p,m.group()) for p,m in summaries if m),None)
    data.update(test_command='.venv/bin/python -m pytest -q tools/truth_video tools/harvest',
                test_log=completed[0].name if completed else 'pytest.log',
                test_result=completed[1] if completed else 'pending')
    (ROOT/'results.json').write_text(json.dumps(data,indent=2)+'\n')
    write_markdown(data)
    for r in rows:print(r['video'],r['systems'],r['strict_systems'],r['scoring_systems'],r['strict_notes'],r['scoring_notes'],r['fast']['scoring']['core_neighbour'])
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('audit','audit-report','report'))
    p.add_argument('--kind',choices=('labels','ghost'),default='labels')
    p.add_argument('--name');p.add_argument('--exclude');p.add_argument('--output',default='truth')
    args=p.parse_args()
    if args.phase=='audit':audit(args.kind,args.name,args.exclude)
    elif args.phase=='audit-report':audit_report(args.kind,args.name)
    else:report(args.output)
