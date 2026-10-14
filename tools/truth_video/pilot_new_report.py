"""Generate the new-channel report from completed local exports and manual audits."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import pilot_report
from pilot_new import PILOTS, ROOT, folders, metadata


def main():
    selected=folders();lookup={p.name:metadata(p) for p in selected}
    pilot_report.metadata=lambda video:lookup[video]
    audits={p.parent.name:json.loads((PILOTS/f'audit-{p.parent.name}/accuracy.json').read_text())
            for p in selected}
    rows=[]
    for p in selected:
        row=pilot_report.video_report(p.name,p/'truth')
        audit=audits[p.parent.name]['by_video'].get(p.name,dict(inspected=0,correct=0))
        row['audit']=audit
        row['native_video']=json.loads((p/'meta.json').read_text())['native_video']
        alignment=json.loads((p/'truth/alignment.json').read_text())['systems']
        families=dict(fitted=('fitted attack agreement',),held_out=('held-out agreement',),
            controls=('ambiguous alignment',),no_cursor=('no-cursor alignment',),
            phase=('phase alias:','notated kicks/snares off','fewer than 6 notated kick/snare'),
            clef=('percussion clef',),spacing=('native staff spacing below',),
            geometry=('uneven staff lines','weak staff line support','expected one staff'))
        row['blocker_families']={name:sum(any(r.startswith(prefixes) for r in s['reasons'])
            for s in alignment) for name,prefixes in families.items()}
        rows.append(row)
    protected=json.loads((PILOTS/'protected-before.json').read_text())
    changed=[p for p,h in protected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    tests=(PILOTS/'pytest.log').read_text().strip()
    duplicates=json.loads((PILOTS/'duplicates.json').read_text())
    duplicates+=json.loads((PILOTS/'tony-duplicate-check.json').read_text())
    speed_checks=json.loads((PILOTS/'speed-duplicate-check.json').read_text())
    cursor_audits=dict(dark=json.loads((PILOTS/'cursor-audit.json').read_text()),
                      gray=json.loads((PILOTS/'gray-cursor-audit-final.json').read_text()))
    audit_history={str(p.relative_to(PILOTS)):json.loads(p.read_text()) for p in
        list(PILOTS.glob('audit-*/accuracy-initial.json'))+list(PILOTS.glob('audit-*-initial/accuracy*.json'))}
    result=dict(videos=rows,audits=audits,duplicates=duplicates,protected_changed=changed,
        audit_history=audit_history,
        cursor_audits=cursor_audits,
        duplicate_speed_checks=speed_checks,
        totals={key:sum(row[key] for row in rows) for key in
                ('systems','strict_systems','scoring_systems','strict_notes',
                 'scoring_notes','strict_seconds','scoring_seconds')},
        scoring_fast_totals={c:sum(row['scoring_fast']['core_neighbour'][c] for row in rows)
                             for c in ('kick','snare','tom')},
        detector_execution=False,network_execution=False,test_result=tests,
        proposed_same_song=[['N0PXFlgigjw','ZsMtIfjkubo','K_AtZoNIoKo']],
        duplicate_group_scope='Conservative shared Change Your Mind source-reference group: Alex full track, Ivannikov solo excerpt, Alfio embedded original plus his overdub. Native footage supports Alex/Alfio; full-mix/stem envelope correlations remain below the strict diagnostic rule for most windows. No timing transfer or perfect audio identity claimed.',
        fast_definition='Distinct kick/snare/tom moments with a nearest core-drum neighbour strictly below 90 ms within each accepted system; chords count once per class/time, no bridging rejected gaps.')
    (PILOTS/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# Eight new channels: offline native-source pilots — 2026-10-02','',
        'All 15 supplied videos were inspected and processed locally. Native media contradicts the 1080p description for both Nyenty files (1280×720); Keep Up is 1216.917 seconds. Supplied titles have rounded durations; metadata retains actual native fps/frame counts. No online channel listing or resolution availability was checked.','',
        'Derived harvest song folders remain under `.eval-cache/pilot_new/<channel>/<video>/`: compressed source, metadata, drum stem, class-agnostic attack/band caches and `truth/`. No canonical song, manifest or training split was published. Scoring includes strict. Found counts include rejected artwork/fragments and repeated display candidates; they are not unique musical systems.','',
        '| Channel / video | Native | Found / clean staff | Strict / scoring systems | Strict / scoring notes | Strict / scoring seconds | Scoring fast kick / snare / tom | Native label audit |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        v=r['video'];c=r['channel'];fast=r['scoring_fast']['core_neighbour'];a=r['audit'];n=r['native_video']
        lines.append(f'| {c} / [{v}](../../{r["folder"]}/QUALITY.md) | {n["width"]}×{n["height"]} | {r["systems"]} / {r["clean_staff_systems"]} | {r["strict_systems"]} / {r["scoring_systems"]} | {r["strict_notes"]} / {r["scoring_notes"]} | {r["strict_seconds"]:.2f} / {r["scoring_seconds"]:.2f} | {fast["kick"]} / {fast["snare"]} / {fast["tom"]} | {a["correct"]}/{a["inspected"]} |')
    lines+=['','Fast counts follow the existing pilot convention: a nearest core-drum neighbour gap <90 ms, deduplicated per class/time inside each accepted system. Strict counts and same-class doubles are separately retained in results.json. A zero means no accepted fast positive; rejected material never becomes a negative.','',
        '## Layouts, conventions and export scope','',
        '| Channel | Observed native layout / cursor | Audited filled positions | Ghost / accent / tom / key conventions and remaining limits |',
        '|---|---|---|---|',
        '| timbuell | Bottom white/gray notation on dark paper; red sweeping block, static score changes | See per-position audit | Sticking below; accents above; no complete physical-kit key. PASIC and Zildjian excerpts include talking/end-card intervals. |',
        '| alexblack | Bottom white/gray strip; thin magenta sweeping cursor; static score changes | See per-position audit | Accent, sticking, ties and multi-measure slash/count bars occur. Slash bars are not expanded into invented hits. No complete printed kit key. |',
        '| baquetazo | White strip on black/purple; thin green cursor, static score changes | See per-position audit | Tony opening explicitly reverses 12-inch/10-inch tom ordering. Only staff-position tom labels retained, never a shared physical-size mapping. Upper sticking and accent rows preserved. |',
        '| copydrum | Bottom light strip; thin green cursor, static score changes | See per-position audit | Small native notation and missing/uncertain left headers remain per-system blockers. Upper o/+ and circled cymbals are withheld. |',
        '| gwon | Bottom light strip; gray rectangular box arrival | See per-position audit | Mixed meters, sticking, o/+ and ghost-style heads; gray playhead is opt-in and still must pass acoustic gates. 20-minute Keep Up is fully attempted. |',
        '| nyenty | Bottom light inset; pale blue box, short source excerpts | See per-position audit | Actual files are 720p. Gergo includes score-free original-footage intervals; those create no truth. Kick/snare/tom conventions only; physical cymbal key unresolved. |',
        '| yeom | Bottom light strip only late in lesson; thin green cursor | See per-position audit | Earlier demonstrations have no displayed score; no earlier truth inferred. Grace notes, sextuplets, ties and accents remain inspectable; unaudited technique flags withheld. |',
        '| bukey | Bottom static light lick score; cursor explicitly disabled | See per-position audit | Explicit ghost parentheses and L/R sticking, sextuplet beams, accents, repeats. Timing uses notated order and class-agnostic attacks with all no-cursor/phase gates; repeated demonstrations are not assigned fabricated onsets. |','',
        'The dark-paper extractor preserves native gray staff intensity only inside each inspected ROI. Achromatic geometry never establishes channel identity or labels. Every original native-spacing, line-support, paired-clef, timing, monotonicity, shifted-control and band-phase threshold remains mandatory. No missing-clef exception or unverified scrolling export was added. Unknown cymbals, physical tom sizes and unaudited technique flags stay withheld.','',
        'Native cursor checks: Tim red blocks 8/8 detections match the real printed block; dim-line predictions 5/5 match the real line, with 3/8 checked real lines missed. Gwon gray boxes were initially merged with anti-aliased staff rows. Sustained column support and enclosed native stem-hole repair preserve the rectangle edges; all 8/8 checked native boxes are located in the final probe, across both videos. These are assistant within-source visual checks, not independent acoustic onset verification or full-video cursor recall. Bounded cursor sheets/JSON and native regression fixtures are retained.','',
        '## Crop audit','',
        '| Channel | Initial diagnostic / earlier cohort | Final native audit | Per-position correct / inspected | Enabled rules |',
        '|---|---:|---:|---|---|']
    for c,a in sorted(audits.items()):
        profile=json.loads((ROOT/f'tools/truth_video/legends/{c}.json').read_text())
        detail=', '.join(f'{k}: {v["correct"]}/{v["inspected"]}' for k,v in a['by_position'].items())
        enabled=', '.join(k for k,v in profile['rules'].items() if v.get('verified'))
        initial_path=PILOTS/f'audit-{c}/accuracy-initial.json'
        if not initial_path.exists():
            initial_path=PILOTS/f'audit-{c}-initial/accuracy-initial.json'
        if not initial_path.exists():initial_path=PILOTS/f'audit-{c}-initial/accuracy.json'
        initial=json.loads(initial_path.read_text()) if initial_path.exists() else None
        initial_text=f'{initial["correct"]}/{initial["inspected"]}' if initial else 'same as final'
        lines.append(f'| {c} | {initial_text} | {a["correct"]}/{a["inspected"]} | {detail} | {enabled or "none"} |')
    lines+=['','Audits compare explicit visually entered target decisions with channel predictions; native and reader crops are retained together. Sampling measures correctness among detected filled heads, not whole-score recall, an independent human audit, a printed full-kit legend or acoustic transcription fidelity. Tim/Alex/Baquetazo/Bukey retain the first decisions and then a contextual recheck after adjacent-head/line-count mistakes. These corrected perfect counts are not fresh blind accuracy. COPYDRUM first had 16 targets in a single clef-confirmed Ndola system; Nyenty first had 25 Gergo targets. Their final 30-target samples include clear-geometry but clef-rejected systems to inspect both videos. This expands label evidence only: all missing-clef reasons remain mandatory and cannot be waived by the audit.','',
        '## Per-video gate blockers','',
        'Counts below overlap. Each counts systems with that reason family; do not add columns. All exact reasons are retained in each QUALITY.md/alignment.json.','',
        '| Channel / video | Fitted | Held-out | Controls | No cursor | Band/phase | Clef | <8 px | Geometry |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        b=r['blocker_families'];lines.append('| '+r['channel']+' / '+r['video']+' | '+' | '.join(str(b[k]) for k in ('fitted','held_out','controls','no_cursor','phase','clef','spacing','geometry'))+' |')
    lines+=['','## Bulk readiness and offline lists','',
        'Channel configs contain the user-supplied handle/playlist listing URL and the crop-audited reader scope. They do not invent UC channel IDs. The existing fetcher constructs its URL from `id` and does not consume `listing_url`; online listing/refresh needs owner work outside this edit scope. An impossible routine-fetch title regex guards the incomplete offline pilot lists until that fetch integration is repaired. Only supplied pilot IDs are stored in the channel lists, explicitly marked as an incomplete offline pilot list in configuration. No unseen online fetch-list IDs were added.','',
        '| Channel | Reader audit | Timed pilot yield | Bulk status |','|---|---|---:|---|']
    for c,a in sorted(audits.items()):
        cr=[r for r in rows if r['channel']==c];profile=json.loads((ROOT/f'tools/truth_video/legends/{c}.json').read_text())
        readable=sum(r['scoring_systems']>0 for r in cr)
        status='Bounded batch of matching inspected layouts only; full-channel stability unverified' if readable and profile['enabled'] else 'Blocked: no accepted timed yield or incomplete reading audit'
        if c=='copydrum':status='Blocked for bulk: most systems lack a confirmed clef; one scoring-only system is insufficient readiness evidence'
        lines.append(f'| {c} | {"enabled partial legend" if profile["enabled"] else "disabled"} | {readable}/{len(cr)} videos with scoring positives | {status} |')
    lines+=['','All eight channels are currently absent from finetune FOLDS. Training admission requires explicit fold assignment and same-performance exclusions; splits.py was deliberately not edited. External solo evaluation duplicates require global training exclusion as well as cross-fold SAME_SONG handling.','',
        '## Same-performance checks','',
        'Local 6-second waveform-envelope windows search the entire retained comparison recording. At least three windows with correlation >=0.8 and lag range <=50 ms establish a shared audio passage in this narrow diagnostic; failed matches alone do not prove different performances. Frame/context evidence and duplicate candidates are discussed in the final response. No truth was transferred or scored.','',
        '| Query video | Reference | Shared audio confirmed | Window correlations / lag seconds |','|---|---|---|---|']
    for d in duplicates:
        kept=d.get('consistent_high_windows') or sorted(d['windows'],key=lambda w:w['correlation'],reverse=True)[:4]
        detail=', '.join(f'{w["correlation"]:.3f}@{w["lag_seconds"]:+.3f}' for w in kept)
        lines.append(f'| {Path(d["query"]).name} | {Path(d["reference"]).name} | {d["shared_audio_confirmed"]} | {detail} |')
    lines+=['','Proposed conservative SAME_SONG group (not applied):','',
        '```python','("N0PXFlgigjw", "ZsMtIfjkubo", "K_AtZoNIoKo"),','```','',
        'Alex/Alfio visibly share the green-shirt Yamaha studio Change Your Mind original. Alfio also shows/plays an overdub, so their audio is not identical. Their highest full-mix query windows follow a stable approximately -254.575 s excerpt offset despite sub-0.8 correlations. Ivannikov is a score-only short solo excerpt with the same title; the diagnostic points to approximately -260.865 s, with drift/mix differences preventing strict audio confirmation. Group these conservatively to prevent leakage; do not treat this as an independently verified audio transfer.','',
        'The user-supplied duplicate assumptions are partly contradicted by current sources. Existing truth/jrod (bd17mxbLaNE) shows a yellow shirt and burgundy cap in a bright room; new Alex 2AzwR38fHKU shows a black got-pocket shirt against a dark background. Tim PASIC shows a red cap/striped jersey and PASIC venue; DailyChops Eric 9juoKgimeR4 shows a green/gray top in an acoustic-panel studio. Nyenty Eric has an overhead red-cymbal/black-head kit view and no retained audio match to DailyChops; their same-performance relationship remains unconfirmed and should be quarantined if owner knowledge says they share a source. Tim Zildjian eik7Wk03o-8, the external Drumeo solo XUdFq81ZSRQ and DailyChops City Lights i1nnfqkv5WY show three different shoots/outfits. No same-performer-only SAME_SONG group is proposed for those pairs.','']
    lines+=['The Nyenty Eric/DailyChops Eric, new Alex J-Rod/external J-Rod, and Tim Zildjian/external Drumeo pairs were additionally probed over a coarse 0.5–2.0 query-time scale grid; none produced three consistent high-correlation windows. The additional El Baquetazo Tony/external Tony Qo5aDGoU7sQ candidate also lacks a retained full-mix or stem match. These negative diagnostics do not prove every possible original-source relationship is absent, and no full online originals were checked.','']
    lines+=['','## Verification and reproducibility','',
        '```bash','.venv/bin/python tools/truth_video/pilot_new.py prepare --jobs 2',
        '.venv/bin/python tools/truth_video/pilot_new.py notation --jobs 1',
        '.venv/bin/python tools/truth_video/pilot_new_audit.py prepare --channels <channel>',
        '# Enter visual decisions in audit-<channel>/manual.json, then report and verify legend rules.',
        '.venv/bin/python tools/truth_video/pilot_new.py read --jobs 2',
        '.venv/bin/python tools/truth_video/pilot_new_duplicates.py',
        '.venv/bin/python -m pytest -q tools/truth_video tools/harvest',
        '.venv/bin/python tools/truth_video/pilot_new_report.py','```','',
        'Demucs uses cached local weights and disables weight downloading; CPU/OpenCV/BLAS/OMP worker thread limits are one. No ADTOF, beat/drum detector, detector activations, scorer or network request ran. Decoded full-mix scratch WAVs and exploratory full frames are deleted; compressed source, stems, bounded audit sheets and final scans remain.','',
        '**Required pytest:** `'+tests.splitlines()[-1]+'`.','',
        'Protected paths changed since the captured baseline: '+(', '.join(changed) if changed else 'none')+'.',
        'The pilot session never wrote harvest.py; its WSL/owner changes arrived concurrently. Canonical manifest/results, finetune splits and timing/phase source hashes remain unchanged. A transient combined-suite failure came from synthetic vault fixtures seeing the real WSL mirror; the owner fixture now isolates its OUT path and the final combined suite passes. See pytest-intermediate-failure.log and pytest-final.log for both outcomes.',
        '[Machine-readable evidence](../../.eval-cache/pilot_new/results.json).','']
    (ROOT/'truth/harvest/PILOT_NEW.md').write_text('\n'.join(lines))
    print(json.dumps([dict(channel=r['channel'],video=r['video'],systems=r['systems'],strict=r['strict_systems'],scoring=r['scoring_systems']) for r in rows],indent=2))


if __name__=='__main__':main()
