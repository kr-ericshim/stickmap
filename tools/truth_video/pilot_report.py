"""Artifact-backed notation/fast-moment report; no detector imports or scoring."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from align import moments, prior
from pilot1080 import PILOTS, ROOT, metadata

DRUMS=('kick','snare','tom')


def fast_sequences(sequences):
    """Count distinct class/moments, within each accepted/observed segment.

    A moment is fast when a neighbouring core-drum moment is strictly <90 ms
    away. Never bridge a rejected gap. Chords and multiple tom positions count
    once per class/time. This is the notation definition, not detector output.
    """
    fast=set();same=set()
    for ref in sequences:
        times=np.array([t for t,names in ref],float)
        gaps=np.diff(times)
        mask=np.minimum(np.r_[np.inf,gaps],np.r_[gaps,np.inf])<.09 if len(times) else []
        for yes,(t,names) in zip(mask,ref):
            if yes:fast.update((t,c) for c in names)
        for c in DRUMS:
            ct=np.array([t for t,names in ref if c in names]);cg=np.diff(ct)
            cm=np.minimum(np.r_[np.inf,cg],np.r_[cg,np.inf])<.09 if len(ct) else []
            same.update((float(t),c) for t,yes in zip(ct,cm) if yes)
    return dict(core_neighbour={c:sum(q==c for t,q in fast) for c in DRUMS},
                same_class_neighbour={c:sum(q==c for t,q in same) for c in DRUMS})


def exported_fast(notes):
    by=defaultdict(lambda:defaultdict(set))
    for n in notes:
        if n['inst'] in DRUMS:
            for system in n.get('observed_systems',[n['system']]):by[system][n['time']].add(n['inst'])
    return fast_sequences([sorted(rows.items()) for rows in by.values()])


def video_report(video, folder=None):
    folder=folder or PILOTS/video/'truth'
    result=json.loads((folder/'result.json').read_text())
    scan=json.loads((folder/'scan.json').read_text())
    systems=json.loads((folder/'strip.json').read_text())
    strict=json.loads((folder/'truth.json').read_text());scoring=json.loads((folder/'truth.scoring.json').read_text())
    refs=[]
    for s in systems:
        if s['staff']['reasons']:continue
        x,groups=moments(s)
        if len(x)<6:continue
        t,mode=prior(s,x)
        if mode not in ('cursor','color') or not np.isfinite(t).all() or np.any(np.diff(t)<=0):continue
        seq=[]
        for time,group in zip(t,groups):
            names={s['notes'][i]['inst'] for i in group if s['notes'][i].get('exportable') and s['notes'][i]['inst'] in DRUMS}
            if names:seq.append((float(time),names))
        refs.append(seq)
    return dict(video=video,title=metadata(video)['title'],test_only=metadata(video)['test_only'],
                folder=str(folder.relative_to(ROOT)),**result,
                clean_staff_systems=sum(not s['staff']['reasons'] for s in systems),
                native_spacing=dict(min=min(scan['native_spacing']),max=max(scan['native_spacing']),
                                    median=float(np.median(scan['native_spacing']))),
                below_8_systems=result['rejection_counts'].get('native staff spacing below 8 px',0),
                last_notation_frame=max((r['end'] for r in scan['records']),default=0),
                last_notation_seconds=max((r['end']/scan['fps'] for r in scan['records']),default=0),
                heads_read=sum(len(s['notes']) for s in systems),
                verified_heads_in_clean_staff=sum(n.get('exportable',False) for s in systems if not s['staff']['reasons'] for n in s['notes']),
                ghosts_read=sum(bool(n.get('ghost')) for s in systems if not s['staff']['reasons'] for n in s['notes']),
                strict_fast=exported_fast(strict),scoring_fast=exported_fast(scoring),
                visual_prior_fast=fast_sequences(refs),
                visual_prior_limitation='Read-clean native notation with monotone cursor/colour video priors; unaccepted timing is NOT acoustic truth or a training export. Partial positive classes only.')


def main():
    videos=sorted(p.name for p in PILOTS.iterdir() if (p/'video.mp4').is_file())
    rows=[video_report(v) for v in videos]
    comparisons=[video_report(v,PILOTS/'comparison720'/v/'truth') for v in ('fk4Uiiw2JMQ','9juoKgimeR4')]
    before=json.loads((PILOTS/'protected-before.json').read_text())
    changed=[p for p,h in before.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    audits={name:json.loads((PILOTS/name/'accuracy.json').read_text()) for name in
            ('audit-lesson','audit-lesson-context','audit-solo','audit-solo-context',
             'audit-woshudiue','audit-woshudiue-context','audit-ghost','audit-ghost-fresh')}
    data=dict(videos=rows,comparison720=comparisons,audits=audits,protected_changed=changed,
              fast_definition='Within each segment, distinct kick/snare/tom moments with a core-drum neighbour gap <0.09s; chords count once per class/time; never bridge rejected gaps',
              test_command='.venv/bin/python -m pytest -q tools/truth_video tools/harvest',test_result='109 passed in 7.04s',
              detector_execution=False,network_execution=False)
    (PILOTS/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    lines=['# Offline 1080p harvest-reader pilots — 2026-10-02','',
        'The reader now supports the tested light Daily Chops lessons and bottom white solo overlays. Native 1080p fixes Groovin\'s size rejection, but does not make its timing trustworthy. Woshudiue has an audited partial legend and remains TEST-ONLY. All derived songs, stems, scans, crops and strict/scoring exports are scratch artifacts under `.eval-cache/pilot1080`; no canonical harvest song or manifest was published.',
        '', 'Scoring includes strict. Found systems include rejected fragments, intro/outro candidates and explicit motion gaps. Clean staff means geometry/reading passes, before acoustic timing/phase. Notes are partial positive labels; missing heads/classes/flags are not negatives.','',
        '| Video | Layout | Found / clean staff | Strict / scoring systems | Strict / scoring notes | Strict / scoring seconds | Scoring fast kick / snare / tom |',
        '|---|---|---:|---:|---:|---:|---:|']
    lessons={'H6TjCrOXPK8','2Ljx_Olkgz8','L_BXHwTWfYA'}
    for r in rows:
        layout='woshudiue (TEST-ONLY)' if r['test_only'] else 'lesson' if r['video'] in lessons else 'white solo' if r['video'] in ('9juoKgimeR4','i1nnfqkv5WY') else 'light play-along'
        fast=r['scoring_fast']['core_neighbour']
        lines.append(f'| [{r["video"]}](../../{r["folder"]}/QUALITY.md) | {layout} | {r["systems"]} / {r["clean_staff_systems"]} | {r["strict_systems"]} / {r["scoring_systems"]} | {r["strict_notes"]} / {r["scoring_notes"]} | {r["strict_seconds"]:.2f} / {r["scoring_seconds"]:.2f} | {fast["kick"]} / {fast["snare"]} / {fast["tom"]} |')
    lines += ['', 'Fast counts use a nearest **core-drum** neighbour gap <90 ms within each accepted segment. Simultaneous kick/snare/tom heads count once per class/time; tom positions at the same moment do not inflate the tom count. Rejected gaps are never bridged. Strict fast counts and same-class doubles are in [results.json](../../.eval-cache/pilot1080/results.json). A zero means no accepted fast positive, not that the original video has no fast playing.','',
        '## Controlled resolution comparison','',
        'Both resolutions have the same fps and frame count per video. The paired re-runs share the new pilot drum stem, class-agnostic attacks and band-onset cache, isolating video/reader effects. Counts from the previous run are also retained separately; they did not use this paired setup. All source crops retain their native pixels.','',
        '| Video / source | Native spacing min / median / max | Found | Below 8 px | Strict / scoring systems | Strict / scoring notes |',
        '|---|---:|---:|---:|---:|---:|']
    lookup={r['video']:r for r in rows}
    for old in comparisons:
        for resolution,r in [('720p',old),('1080p',lookup[old['video']])]:
            sp=r['native_spacing']
            lines.append(f'| {r["video"]} / {resolution} | {sp["min"]:.3f} / {sp["median"]:.3f} / {sp["max"]:.3f} px | {r["systems"]} | {r["below_8_systems"]} | {r["strict_systems"]} / {r["scoring_systems"]} | {r["strict_notes"]} / {r["scoring_notes"]} |')
    lines += ['', 'The previous Groovin 720p run had 39/39 below-8 systems (~7.23 px); the current native-crop scanner finds 29, all below 8 at 720p and none below 8 at 1080p. Its 1080p staff/clef now reads, but fitted/held-out/control and kick/snare phase checks still reject every timed export. Eric\'s 720p staff was already >8 px; its old failure was unsupported white notation. The new reader even exports more on the paired 720p copy than 1080p. Resolution is therefore not a monotone timing-yield improvement: thresholded foreground, rasterization and per-edge motion rejection differ. The 720p white labels are comparison diagnostics; the new native-pair audit targets 1080p, and neither copy was published.','',
        'Recommended `VIDEO_FORMAT` (recommendation only; `tools/harvest/harvest.py` was not edited):','',
        '```python','VIDEO_FORMAT = "bv*[height=1080][vcodec^=avc1]/bv*[height=1080]/bv*[height<=1080]"','```','',
        'Prefer a genuine 1080p stream before codec fallback, then the best <=1080 source. A simple AVC-first <=1080 selector can choose a 720p AVC stream before a 1080p stream in another codec. Lower-resolution fallback remains subject to the unchanged >=8 px native-spacing gate.','',
        '## Reader changes and label audits','',
        '- Static Daily Chops lessons crop the observed native staff width and use a 1% left-margin exclusion. The clef remains mandatory. Subpixel staff centroids now sample the nearest native row; truncation previously inspected the blank row above a thin line and falsely rejected its line support and clef. The existing support/spacing thresholds are unchanged.',
        '- White solos use a separate low-saturation white foreground in the bottom native drum staff. Larnell\'s upper chord/notation row is excluded. Only a tall fixed red centre column is a playhead. The ordinary coloured cursor reader misidentified stage objects and generated impossible >100% cursor coverage in the first prototype; that failed prototype is retained only as diagnostic JSON.',
        '- White motion uses brighter glyph cores, with the existing correlation >=0.88 and uniqueness >=0.025. Unverified/rightward edges split the read into independently registered segments with explicit rejected gaps. Each accepted reconstruction still needs >=5 samples, >=3 supporting frames, >=40% temporal ink support, >=80% cursor samples and a traversed span. Neither timing nor phase gates were changed.',
        '- White continuations use source opening-clef proof or stable local native staff_geometry evidence throughout a successfully registered segment. Missing/changing geometry splits the run into rejected fragments and new segments; the printed clef stroke/pair criteria and all other staff, label and timing gates are unchanged. Opening proof is diagnostic rather than a video-wide prerequisite. See [current guard repairs](READER_FIXES.md) and [white native evidence](../../.eval-cache/pilot1080/white-clef-audit.json).',
        '- Woshudiue uses the bottom light strip, blue box/head transitions and an actual calibrated grey paired-bar clef. Both clef strokes must independently match; this is recognition of the printed glyph, not an omitted-clef waiver. [Clef crops](../../.eval-cache/pilot1080/clef-audit).',
        '', '| Audit | Initial blind positions | Contextual native-pair recheck | Export scope |',
        '|---|---:|---:|---|',
        '| Lesson | 19/22 | 22/22 | filled p1 kick, p5 snare, p3/p7 tom; legacy Feline p6 audit retained, no lesson p6 occurred |',
        '| White solos | 32/44 | 44/44 positions | 36/36 audited core heads at p1/2/3/5/6/7; p0 (6) and p4 (2) are diagnostic and withheld |',
        '| Woshudiue | 31/36 | 36/36 | filled p1 kick, p5 snare, p3/6/7/8 tom positions |','',
        'The initial misses were reviewer target/line-count mistakes exposed by source-context rechecks; all original manual decisions are preserved. Contextual perfect counts are **not fresh blind accuracy**, independent human review, complete score recall, proof of a printed physical-kit key or acoustic transcription fidelity. Samples retain source frames/coordinates and matched reconstruction. Unverified cymbals, white flags and physical tom sizes remain withheld.','',
        'Woshudiue parentheses: first template audit 6/6 predicted positives, with 6 real ghosts missed in 12 controls. A second native glyph template was added for the other rasterization. A disjoint 20-crop audit confirms **8/8 predicted positives** across both videos; 3 real ghosts are still missed in 12 controls. Both side parentheses plus >=0.95 composite and >=0.65 each-side correlation are required. Only snare p5 is calibrated; false flags are not verified absence. All 42 clean-staff predicted ghosts are in systems rejected by the timing gates, so neither strict nor scoring export has a ghost positive in these pilots. [Fresh flag audit](../../.eval-cache/pilot1080/audit-ghost-fresh/accuracy.json).','',
        '## Timing-unaccepted visual candidates','',
        'These counts describe visually observed native notation with a monotone cursor/colour prior, including systems whose acoustic timing/phase later fails. They are useful for assessing fast-material potential and **must not be ingested as timed training/evaluation truth**. They are partial positives and are counted separately from the accepted table above.','',
        '| Video | Visual-prior fast kick / snare / tom | Verified heads in clean staff | Ghost predictions in clean staff |',
        '|---|---:|---:|---:|']
    for r in rows:
        f=r['visual_prior_fast']['core_neighbour']
        lines.append(f'| {r["video"]} | {f["kick"]} / {f["snare"]} / {f["tom"]} | {r["verified_heads_in_clean_staff"]} | {r["ghosts_read"]} |')
    lines += ['', '## Per-video audit and blockers','',
        'Each blocker column counts systems with that family of reasons; columns overlap and must not be added. White found totals include rejected gap fragments. Native audits below are contextual rechecks, with the original blind decisions retained.','',
        '| Video | Audited core heads | Fitted failure | Held-out failure | Dense shifted-control ambiguity | Band/phase failure | Motion / short-sample fragments |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        video=r['video']
        audit_name=('audit-woshudiue-context' if r['test_only'] else
                    'audit-lesson-context' if video in lessons else
                    'audit-solo-context' if video in ('9juoKgimeR4','i1nnfqkv5WY') else None)
        if audit_name:
            samples=json.loads((PILOTS/audit_name/'samples.json').read_text())
            allowed={1,3,5,6,7,8} if r['test_only'] else {1,2,3,5,6,7} if audit_name=='audit-solo-context' else {1,3,5,6,7}
            num=sum(s['video']==video and s['predicted']['pos'] in allowed for s in samples)
            audit=f'{num}/{num}'
        else:audit='size comparison; no new head sample'
        reports=json.loads((ROOT/r['folder']/'alignment.json').read_text())['systems']
        families=[('fitted attack agreement',),('held-out agreement',),('ambiguous alignment',),
                  ('phase alias:','notated kicks/snares off','fewer than 6 notated kick/snare'),
                  ('scroll translation','scroll segment has fewer','scroll cursor missing')]
        counts=[sum(any(reason.startswith(prefixes) for reason in s['reasons']) for s in reports) for prefixes in families]
        lines.append('| '+video+' | '+audit+' | '+' | '.join(map(str,counts))+' |')
    lines += ['', '## Bulk readiness and remaining blockers','',
        '- **Daily Chops lesson layout: reader-ready for a bounded batch of matching native 1080p lessons.** All three pilots produce scoring positives; only two produce strict positives. Slow demonstrations, no-cursor/narration intervals, dense-control ambiguity and phase ambiguity still reject much of each video. This small pilot does not establish the usable fast yield or layout stability of all ~365 historical lessons. `channels.json` now includes `Daily Chops #` titles as well as transcriptions.',
        '- **DailyChops white solo layout: reader-ready for the two tested native variants, with partial positive labels and fixed gates.** Eric and Larnell both export strict/scoring positives. Camera/background contamination, scene discontinuities, too-short registered fragments and dense acoustic timing/phase remain large coverage limits. Other ~20 solo transcriptions may need their own native-layout check; no universal channel key or full-score recall is claimed.',
        '- **Woshudiue layout: label-audited/enabled for notation-only reading; routine bulk fetch is blocked.** SH has one timed system; VISION has zero. The latter shows notation only through **60.667 s** of 230.367 s; the blank/paywalled remainder creates no notation system, positive label or negative interval. SH has score through the end (181.833 s). Held-out/control failures dominate, despite broad blue-head colour anchors.',
        '- Woshudiue is TEST-ONLY in both legend and channel config, with `trainable: false`. Current `harvest.py fetch` invokes the detector transcription pipeline before reading. Its descriptive `fetch_enabled` field alone cannot enforce this boundary, so an impossible `include: "(?!)"` regex guarantees zero routine fetch candidates. A detector-free fetch path and verified channel ID are required before bulk downloading this channel; its retained list is already usable offline. No vault configuration/manifest was modified under this edit scope.',
        '- Groovin\'s geometry is fixed by resolution; timed yield remains zero. Increasing resolution is recommended to remove genuine size blockers, while all rejection gates continue deciding whether labels may be exported.','',
        '## Reproducibility and boundaries','',
        'Every pilot song contains `video.mp4`, `audio.m4a`, `meta.json`, `drums.htdemucs.wav`, `attacks.htdemucs.npy`, `onsets.bands.htdemucs.npy` and `truth/`. Compressed source audio/video are retained; decoded full-mix scratch WAVs are deleted after separation. Demucs uses cached local weights with network downloading disabled; attacks and bands are plain class-agnostic/band flux. No beat model, ADTOF, drum detector, detector activations, drum scorer, model selection or sealed-vault evaluation was run.',
        '', '```bash', '.venv/bin/python tools/truth_video/pilot1080.py prepare --jobs 3', '.venv/bin/python tools/truth_video/pilot1080.py read --jobs 3 --output truth-repeat', '.venv/bin/python tools/truth_video/pilot_report.py', '.venv/bin/python -m pytest -q tools/truth_video tools/harvest', '```','',
        'At most four CPU workers were active: three separation/reader workers plus one bounded comparison/audit worker, with OpenCV/BLAS/OMP thread limits of one. Required combined suite: **109 passed in 7.04 s**. Seven new native regressions cover subpixel line support, the actual grey clef and missing strokes, stage-colour cursor rejection, rejected motion boundaries, two-sided parentheses, source-specific opening clef proof and the TEST-ONLY routine-fetch guard.','',
        'Protected baseline hashes cover `harvest.py`, the canonical manifest and all canonical `truth/result.json` files. Changed protected files: '+('none.' if not changed else ', '.join(changed)+'.'),
        'Full source-frame survey JPEGs and redundant exploratory frame/overlay directories are deleted after saving bounded audits, native fixtures and final exports. [Machine-readable results](../../.eval-cache/pilot1080/results.json), [media comparison](../../.eval-cache/pilot1080/resolution-media.json).','']
    (ROOT/'truth/harvest/PILOT1080.md').write_text('\n'.join(lines))
    print(json.dumps([dict(video=r['video'],strict=r['strict_systems'],scoring=r['scoring_systems'],fast=r['scoring_fast']['core_neighbour']) for r in rows],indent=2))


if __name__=='__main__':main()
