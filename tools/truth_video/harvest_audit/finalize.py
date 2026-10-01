"""Rebuild validation summaries from actual exports and visual annotations."""
import json
from pathlib import Path
import sys
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from batch import report,manifest_read,song_path,dump
from reader import label


def audit(folder,scope):
    samples=json.loads((folder/'samples.json').read_text());manual=json.loads((folder/'manual.json').read_text());result={}
    for item in samples:
        a=manual.get(str(item['id']))
        if not a:continue
        c=result.setdefault(item['channel'],dict(inspected=0,correct=0,labels={},errors=[],videos=[],scope=scope))
        pred=label(item['predicted']);correct=pred==a['label'];c['inspected']+=1;c['correct']+=int(correct)
        v=c['labels'].setdefault(pred,dict(inspected=0,correct=0));v['inspected']+=1;v['correct']+=int(correct)
        if not correct:c['errors'].append(dict(id=item['id'],predicted=pred,actual=a['label']))
        if item['video'] not in c['videos']:c['videos'].append(item['video'])
    dump(folder/'accuracy.json',result)
    return result


def main(root):
    root=root.resolve()
    validation=root/'validation'
    final=audit(validation/'export-audit','Final exported heads, stratified by channel and label; assistant visual review, not independent human review, recall, audio-label fidelity or population precision.')
    prior={}
    for name in ('audit','audit-wave2','audit-wave3'):
        prior.update(audit(validation/name,'Diagnostic candidate audit before fixes; includes unresolved labels and rejected staff candidates.'))
    for channel,c in final.items():c['diagnostic_audit']=prior.get(channel)
    # The production report reads this one current channel-audit index.
    for channel,c in prior.items():
        if channel not in final:c['exported_audit']=False;final[channel]=c
    dump(validation/'audit/accuracy.json',final)
    channels=report(root)
    totals={k:sum(c[k] for c in channels.values()) for k in ('attempted','read','strict_systems','scoring_systems','strict_seconds','scoring_seconds','strict_notes','scoring_notes')}
    counts=Counter()
    for c in channels.values():counts.update(c['scoring_labels'])
    stamp=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')
    lines=['# Harvest validation results','',f'Generated {stamp}. Scoring includes strict; durations are unions of accepted moment spans.','',
        f'{totals["attempted"]} songs attempted; {totals["read"]} exported at least one verified timed note. Strict: {totals["strict_seconds"]:.2f} s / {totals["strict_notes"]} notes in {totals["strict_systems"]} systems. Scoring: {totals["scoring_seconds"]:.2f} s / {totals["scoring_notes"]} notes in {totals["scoring_systems"]} systems.','',
        'The validation cohort is the first three available songs of each of the ten channels inspected during this run. Later arrivals remain fetched. Other channel profiles stay disabled pending their own three-song/key audit; this is not approval of all twenty channels.','',
        '## Reading evidence','',
        '530 diagnostic head crops exposed wrong default keys, open marks, diamond/filled confusion and room/artwork false candidates. Fixes are documented in the individual audit manifests. A further 242 stratified crops from the final strict-plus-scoring exports were visually inspected: 242/242 labels matched their notation conventions. This is an assistant visual audit, not an independent human review, exhaustive score recall, acoustic fidelity, or a population-level 100% accuracy claim. Template-development system jfFi5GeGzUg/003 was excluded.','',
        'Supplemental evidence: 38 side-stick/accent/diamond crops (diamonds remained physically unverified), 16 separate-o open-hat positives, and a channel-specific ghost glyph check from a supplied native still. Ghost calibration uses one development glyph plus ten other positive locations and regular-note controls; its positive check is within the same frame, not an independent video. The three timed MY DRUM SHEET pilots still have no confirmed ghost exports.','',
        'MY DRUM SHEET printed key: [native intro](validation/mydrumsheet-key.png), video j3jmresIot4 at 6 s. Hi-hat is position 8 and ride position 9. Yeol circled:9 is crash, supported by [full frame](validation/open-example-259.jpg); a separate o over a cross is open hat. Do not share generic KBdrums keys across these channels.','',
        '| Channel | Exported-label audit | Per-label confirmed / inspected |', '|---|---:|---|']
    for name,c in channels.items():
        a=final.get(name)
        if not a or not c['enabled']:lines.append(f'| {name} | unverified for export | — |');continue
        parts=', '.join(f'{k}: {v["correct"]}/{v["inspected"]}' for k,v in sorted(a['labels'].items()))
        lines.append(f'| {name} | {a["correct"]}/{a["inspected"]} | {parts} |')
    lines+=['','## Timing evidence','',
        'Strict and scoring gates are unchanged. Alignment uses class-agnostic attacks and held-out prediction/shifted controls. It does not use detector instrument labels and is not independent hand-timed audio truth. Color threshold crossings are quantized to a frame; rendering-fade and acoustic error remain separate.','',
        'An initial 23-edge before/edge/after visual audit confirmed 22 same-head color transitions. One transition in p7J0T8SLumc was a changed page/glyph. This prompted cursor-wrap segmentation for near-identical pages and a local-ink stability guard. The invalid edge was checked for removal in color-regression.json. Post-fix inspection also found a gradual blue tint before one threshold crossing: the stored one-frame uncertainty is quantization of that criterion, not total rendering-fade/onset error. Final acoustic timing must still pass the unchanged alignment gates; no subframe or independent acoustic accuracy is claimed.','',
        '| Channel | Strict / scoring systems | Strict / scoring seconds | Strict / scoring notes |', '|---|---:|---:|---:|']
    for name,c in channels.items():lines.append(f'| {name} | {c["strict_systems"]}/{c["scoring_systems"]} | {c["strict_seconds"]:.2f}/{c["scoring_seconds"]:.2f} | {c["strict_notes"]}/{c["scoring_notes"]} |')
    lines+=['','## Scoring label totals','',json.dumps(dict(counts),ensure_ascii=False,sort_keys=True),'',
        '## Per-song quality and provenance','', '| Channel / video | Strict / scoring systems | Strict / scoring notes | Quality |', '|---|---:|---:|---|']
    for row in manifest_read(root/'manifest.jsonl').values():
        try:folder=song_path(root,row)/'truth'
        except ValueError:continue
        if not (folder/'result.json').exists():continue
        r=json.loads((folder/'result.json').read_text());rel=folder.relative_to(root)
        lines.append(f'| {row.get("channel")} / {row["video"]} | {r["strict_systems"]}/{r["scoring_systems"]} | {r["strict_notes"]}/{r["scoring_notes"]} | [QUALITY.md](<{rel}/QUALITY.md>) |')
    lines+=['','## Limits and unresolved work','',
        '- No end-to-end validation yet for Drumscribe, full pages (MoYin/Swartz), DailyChops, Ollie Woods or the other disabled channel profiles. Their geometry/cursor/crop tests and generic scrolling code do not establish their physical notation keys.',
        '- Drumclass remains disabled: inverted transparent score extraction and upper-head reading vary across pilots. Its rejected promotional artwork produced false candidate heads, never exported truth.',
        '- DRUMMATE has two zero-yield pilots: scrolling/bar-highlight cursor timing is not sufficiently established. Production scrolling reconstruction requires a separately validated scrolling_verified flag. No fallback threshold is relaxed to make it pass.',
        '- Native staff spacing below 8 px stays rejected in every layout. The real 7 px fixture and inspection upscaling do not justify small-glyph production labels.',
        '- Ghost reading now has an explicit MY DRUM SHEET composite-glyph calibration, but no timed ghost examples in this cohort. Ride, bell and pedal-hat counts may remain zero: unverified physical cymbal identity is withheld.',
        '- This is partial positive notation truth. label-coverage.json identifies unverified rules/heads. Missed heads and false flags are not verified negatives. Use verified-reference recall or coverage-aware evaluation; unmasked full-family precision can penalize unlabelled instruments. Classifier splits should be by video/performance (preferably channel/song groups), not random notes from the same audio.',
        '- Original/cover/backing-track fidelity and simplified arrangements were not independently listened through. A printed note is not proof that the original recording contains that articulation.',
        '- Compaction was tested on tiny disposable fixtures and was not run on harvested sources. Existing legacy/JD exports were not rebuilt. No downloads, root application changes or port 8765 access were made.','',
        '## Verification','',
        'See reader-tests-final.log, tier-tests-final.log, harvest-tests-final.log and batch-tests-final.log under validation/. They cover legacy behavior, timing tiers, real native crops, cursor colors, page/cursor resets, stable color edges, ghost/open glyphs, partial journal writes, lock retry/recovery, timeouts and truth-preserving compaction.']
    (root/'VALIDATION.md').write_text('\n'.join(lines)+'\n')
    dump(root/'validation-results.json',dict(generated_at=stamp,totals=totals,scoring_labels=dict(counts),channels=channels))
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main(Path(sys.argv[1] if len(sys.argv)>1 else 'truth/harvest'))
