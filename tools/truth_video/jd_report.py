"""Write auditable JD results and separate strict/scoring coverage for every set."""
import json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from align import quality,match

ORDER=['jd-growing-older','jd-zildjian-drums','jd-zildjian-full','jd-vfjams','jd-namm',
       'jd-whatup','jd-all-caps','jd-two-shrimps','jd-tx','jd-solo','jd-alfio','jd-vfjams-short','jd-sniff']
READING={
 'jd-growing-older':'Partly legible at 1080p, but native staff spacing is only 7.5–7.75 px. The unchanged 8 px gate rejects every system. Some rows/heads remain missed. First frame explicitly says cymbal bell; ending caption claims only about 90% transcription accuracy. No accepted album-bell truth.',
 'jd-zildjian-drums':'Readable 1080p bottom staff. No full kit key found in inspected opening 0–8 s: logo followed by “cross stick play over cymbal on head”. Upper crosses and triangles stay cymbal_unknown. Thick-line centroids, local glyphs and complete 16th/32nd beam groups repair reading/alignment. Unparsed rests/tuplets and nonuniform spacing reject most pages.',
 'jd-zildjian-full':'Same reference as drums-only, transferred by independent waveform-envelope alignment. Not a second independent performance or reading audit. 11 windows, correlations 0.892–0.971; fitted offset = 2.197 ms + 37.879 ppm × time; largest leave-one-window-out residual 7.74 ms.',
 'jd-vfjams':'Readable 1080p white scrolling notation, after image-motion reconstruction over the video. 48 four-second blocks from 12–204 s; other time not exported. Native 25% temporal ink support preserves thin strokes that a median erased. Eight missed crosses in accepted blocks were added from visual coordinates. Two otherwise scoring candidates are rejected for incomplete tied/diamond-head interpretation.',
 'jd-namm':'Readable 1080p paged strip. Peter Campion layout; hollow triangle heads are now read. Filled 1/5/3/7 use kick/snare/tom notation conventions. No explicit physical cymbal key was verified: triangles and upper crosses stay cymbal_unknown.',
 'jd-whatup':'Full score with keyboard/piano staves. Drum rows are visible, but low native spacing, mixed clefs and no row-level timing cursor prevent a validated timed export. Candidate non-drum rows are not treated as drum truth.',
 'jd-all-caps':'Readable 720p bottom strip; no cursor. One page passes the separate scoring tier, none strict. No complete physical cymbal key verified; upper crosses stay cymbal_unknown.',
 'jd-two-shrimps':'Readable 720p multipage instructional score with blue cursor. Local text explicitly refers to hi-hat. Two visually missed hat heads in system 023 are added in reading-corrections.json. This instructional video is reported separately; its audio provenance as original JD performance versus playback was not independently established.',
 'jd-tx':'No readable notation in inspected intro and 25/50/75/90% frames; only a promotional title. This is a sampled inspection, not a claim to have watched every frame.',
 'jd-solo':'Readable full drum-score pages, but no cursor or validated active interval per row. On-screen page visibility alone cannot time five concurrent rows.',
 'jd-alfio':'Performance/cover footage in inspected intro and 25/50/75/90% frames, with no score visible. This is a sampled inspection, not proof about every intervening frame.',
 'jd-vfjams-short':'Performance footage in inspected intro and 25/50/75/90% frames, with no score visible. This is a sampled inspection, not proof about every intervening frame.',
 'jd-sniff':'Full piano-and-drum score pages at 720p. Small native staff spacing and absent row timing; no system qualifies. Piano notes are not exported as drum truth.',
}


def load(path):return json.loads(path.read_text())
def dump(path,obj):path.write_text(json.dumps(obj,indent=2))
def labels(notes):return dict(Counter(n['inst']+('_'+n.get('articulation','closed') if n['inst']=='hihat' else '') for n in notes))
def counts(d):return ', '.join(f'{k}={v}' for k,v in sorted(d.items())) or '0'
def fmt(q):return '—' if not q else f'{q["within_20ms"]:.1%} / {q["within_40ms"]:.1%}'
def pr(s):
    if not s:return '—'
    p='—' if s['precision'] is None else f'{s["precision"]:.1%}'
    r='—' if s['recall'] is None else f'{s["recall"]:.1%}'
    return f'{p} / {r} ({s["tp"]}/{s["predicted"]}/{s["reference"]})'


def build(root):
    records=[]
    for path in sorted(root.glob('*/alignment.scoring.json')):
        f=path.parent;r=load(path);strict=load(f/'truth.json');allnotes=load(f/'truth.scoring.json');extra=load(f/'truth.additional.json')
        old=load(f/'alignment.json');systems=r['systems'];accepted=[s for s in systems if s['scoring_usable']]
        row=dict(name=f.name,systems=len(systems),strict_systems=sum(s['usable'] for s in systems),
                 additional_systems=sum(s['tier']=='scoring' for s in systems),strict_seconds=old['trustworthy_seconds'],
                 scoring_seconds=r['trustworthy_seconds'],strict_notes=len(strict),additional_notes=len(extra),
                 strict_counts=labels(strict),additional_counts=labels(extra),counts=labels(allnotes),
                 reading=READING.get(f.name,'Existing reader/audit; strict export preserved unchanged.'),
                 accepted_system_ids=[s['system'] for s in accepted],
                 additional_system_ids=[s['system'] for s in systems if s['tier']=='scoring'])
        diag=f/'diagnostics.json'
        if diag.exists():row['detector']=load(diag)
        records.append(row)
        lines=[f'# {f.name}: reading and timing quality','',row['reading'],'',
               'Strict truth.json and alignment.json keep the original gates. truth.scoring.json is the union of strict plus separately marked scoring-only systems; truth.additional.json contains only the additions. No rejected gap contributes detector false positives.','',
               f'Strict: {row["strict_systems"]} systems, {row["strict_seconds"]:.3f} s, {len(strict)} notes. Additional: {row["additional_systems"]} systems, {r["trustworthy_seconds"]-old["trustworthy_seconds"]:.3f} s, {len(extra)} notes.','',
               'Strict labels: '+counts(row['strict_counts'])+'.','Additional labels: '+counts(row['additional_counts'])+'.','',
               '## Evidence and limits','',
               'Overlays are reading checks, not independent human adjudication. Timing uses class-agnostic provided attack caches, with held-out moments and shifted controls; no listening/onset audit was performed. Dense rhythms can still contain phase aliases. A missing flag means unconfirmed, not an exhaustive negative.','']
        if f.name.startswith('jd-'):
            lines+=['The accepted JD overlays were visually inspected in this run. This is not a blinded statistical audit; the legacy 190-head statistic does not apply to these new sets. Corrections and rejected readings are retained as JSON where applicable.','']
        if 'transferred_from' in r:lines+=['See [source reading overlays](../jd-zildjian-drums/QUALITY.md) and [audio-transfer.json](audio-transfer.json).','']
        lines+=['| System | Interval s | Mode | Moments | Fitted 20/40 | Held out 20/40 | Gain 20/40 | Tier | Reasons |',
                '|---|---|---|---:|---|---|---|---|---|']
        for s in systems:
            target=f'check/system_{s["system"]:03d}.png' if 'transferred_from' not in r else f'../jd-zildjian-drums/check/system_{s["system"]:03d}.png'
            gain=f'{s.get("held_out_gain_20ms",0):.1%}/{s.get("held_out_gain_40ms",0):.1%}'
            reason='; '.join(s.get('reasons',[]))
            if s['tier']=='rejected':reason+='; scoring: '+'; '.join(s.get('scoring_reasons',[]))
            lines.append(f'| [{s["system"]:03d}]({target}) | {s["interval"][0]:.2f}–{s["interval"][1]:.2f} | {s.get("mode","unread")} | {s["moments"]} | {fmt(s.get("after"))} | {fmt(s.get("held_out"))} | {gain} | {s["tier"]} | {reason} |')
        (f/'QUALITY.md').write_text('\n'.join(lines)+'\n')
    byname={v['name']:v for v in records};jd=[byname[n] for n in ORDER];legacy=[v for v in records if not v['name'].startswith('jd-')]
    totals={}
    for name,chosen in [('legacy',legacy),('jd_evaluated_audio_versions',jd),('jd_unique_reference',[v for v in jd if v['name']!='jd-zildjian-full'])]:
        totals[name]={k:sum(v[k] for v in chosen) for k in ['strict_systems','additional_systems','strict_seconds','scoring_seconds','strict_notes','additional_notes']}
    dump(root/'jd-results.json',dict(totals=totals,videos=jd,legacy=legacy))
    lines=['# JD Beck transcription truth and detector results','',
           'These are timed references to other people’s notation, not a guarantee that the notation exactly describes the recording. The Growing Older source itself claims about 90% accuracy and provides no accepted truth at the unchanged resolution gate. Source audio provenance of the instructional Two Shrimps video is not independently verified.','',
           'No network, source edits, or port 8765 were used. Detector source was copied read-only into jd-run/source; source-manifest.json records hashes and capture time. The relevant toms/models/fills source remained unchanged during this run; unrelated server/library/musicxml edits by the other worker are recorded separately.','',
           '## Coverage by video','',
           'S = strict; +G = scoring-only additions. Duration is union of first-to-last accepted moment spans. The full-band Zildjian row duplicates a reference and must not inflate unique truth totals.','',
           '| Video | S / +G systems / candidates | S seconds + G | S notes + G | Labels, S+G |',
           '|---|---|---|---|---|']
    for v in jd:
        lines.append(f'| [{v["name"]}]({v["name"]}/QUALITY.md) | {v["strict_systems"]} / +{v["additional_systems"]} / {v["systems"]} | {v["strict_seconds"]:.2f} + {v["scoring_seconds"]-v["strict_seconds"]:.2f} | {v["strict_notes"]} + {v["additional_notes"]} | {counts(v["counts"])} |')
    for v in jd:lines+=['',f'**{v["name"]}:** {v["reading"]}']
    lines+=['','## Separate scoring tier, including old sets','',
            'The strict thresholds are unchanged. The additional tier requires fitted 40 ms agreement ≥90%, held-out 40 ms ≥85% with a cursor or ≥90% without one (including rhythm priors), ≥15 percentage points above shifted controls at either tolerance, ≥6 moments, valid reading/staff and strictly increasing times. It drops the held-out 20 ms requirement. This is a screening tier for ±50 ms evaluation, not a claim of strict onset accuracy.','',
            'Evidence: All Caps system 005 has held-out 20/40 ms of 63.6%/97.0%, failing the no-cursor 70% 20 ms gate while supporting 50 ms scoring. VFJams system 024 has 54.1%/89.2%, narrowly failing strict 20 ms while satisfying the cursor scoring rule. Old Ken Jezer system 000 has 46.2%/92.3%. Fitted agreement is still checked and misses are not hidden by a larger snapping radius.','',
            '| Existing set | Strict systems | +G systems | Strict s | +G s | Strict notes | +G notes |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for v in legacy:lines.append(f'| {v["name"]} | {v["strict_systems"]} | {v["additional_systems"]} | {v["strict_seconds"]:.2f} | {v["scoring_seconds"]-v["strict_seconds"]:.2f} | {v["strict_notes"]} | {v["additional_notes"]} |')
    lines+=['',f'Coverage totals: `{json.dumps(totals)}`','',
            'J-Rod remains 3/11: its rejected systems do not meet the chosen fitted/held-out 40 ms criteria. The new tier is not tuned to make J-Rod pass. Legacy strict truth/alignment files were not rewritten.','',
            'Empirical added-tier prediction counts (jd-inspection/scoring-tier-evidence.json): old sets match135/205 held-out moments at20ms,185/205 at40ms and188/205 at50ms; JD additions match41/70,65/70 and66/70 respectively. Thus the JD additions give94.3% held-out agreement at the scoring tolerance versus58.6% at20ms. These remain same-performance prediction checks, not independently hand-timed truth.','',
            '## Detector scores','',
            'P / R (TP / predictions / reference), one-to-one within ±50 ms. Scores are selected-region scores, not whole-song performance. Fine upper-cymbal identities are unresolved in several layouts; use combined hat/cymbal to assess metal-family detection. A zero separate hihat score against cymbal_unknown is not evidence that a physical hi-hat was misclassified. Separate family rows are retained in evaluation logs.','']
    for tier in ['strict','scoring']:
        lines+=[f'### {tier}'+(' (strict plus additional)' if tier=='scoring' else ''),'',
                '| Video | Kick | Snare | Tom | Hat/cymbal combined | Ghost snare recall |',
                '|---|---|---|---|---|---|']
        for v in jd:
            d=v.get('detector',{}).get(tier,{})
            if not d.get('scores'):continue
            g=d['ghost'];lines.append(f'| {v["name"]} | '+' | '.join(pr(d['scores'].get(c)) for c in ['kick','snare','tom','hat/cymbal'])+f' | {g["matched"]}/{g["reference"]} |')
    lines+=['','## What pre-processing tom candidates coincide with','',
            'Every row counts original pre-tom-stage detector candidates in accepted intervals. Truth labels within ±50 ms are shown as exclusive label-set combinations; simultaneous instruments and nearby fast strokes cannot be causally separated. “nothing” means no exported reference in that window, not proven silence. These diagnostic correspondences are not the one-to-one P/R calculation.','',
            'The code calls this head energy, but _Stem.spectrum returns root-mean-channel power amplitude, and _headless sums amplitudes. The reported share is exactly that implementation: summed struck spectral amplitude in 60–450 Hz divided by 60–12000 Hz, using the 60 ms ring window. It is not an integrated squared-energy ratio.','',
            'Actual _pitch calls are recorded. Hits removed before pitch classification get a separately marked counterfactual call to the same _pitch function. All raw hertz values and source kind are in correspondence.*.json.','']
    for tier in ['strict','scoring']:
        lines += [f'### {tier} candidate share/pitch counts','', '| Video | Share bin | Pitch bin | Truth label sets: count |','|---|---|---|---|']
        for v in jd:
            d=v.get('detector',{}).get(tier,{})
            for key,row in d.get('tom_candidates',{}).get('share_pitch',{}).items():
                share,pitch=key.split(' / ');lines.append(f'| {v["name"]} | {share} | {pitch} | {counts(row)} |')
    lines+=['','## Bell renaming, low toms, and missed snares','',
            '| Video / tier | ride_bell output → notation | tom_low output → notation | Unmatched notated snare → nearby detector labels |',
            '|---|---|---|---|']
    for v in jd:
        for tier in ['strict','scoring']:
            d=v.get('detector',{}).get(tier,{})
            if not d.get('scores'):continue
            lines.append(f'| {v["name"]} / {tier} | {counts(d["bells"]["truth"])} | {counts(d["tom_low"]["truth"])} | {counts(d["snare_misses"])} |')
    lines+=['','## Findings supported by this selected truth','',
            'The tom-candidate population is the input to tom_sizes, after models.pick_hits and tom_runs, not every raw ADTOF peak. There are 34 candidates in strict regions and 40 in strict-plus-scoring regions (NAMM 20 and VFJams 20). Zildjian, All Caps and Two Shrimps contribute zero such candidates in their accepted regions.','',
            '- Across both tiers, the 16 candidates with head-share<0.1 coincide with tom 6, snare 5, unknown-cymbal 2, snare+unknown-cymbal 2, and no reference 1. Thus 11/16 coincide only with notated drums, directly contradicting a blanket “no drum head rang” interpretation. Strict-only already contains tom 5 and snare 4 among 14 low-share candidates.','- All 12 ride_bell promotions in accepted regions are from NAMM: tom 3, snare 4, unknown-cymbal 2, snare+unknown-cymbal 2, no reference 1. Zero is verified as a notated bell. Seven are drum-only correspondences. VFJams removes another 4 low-share candidates (tom 3,snare 1) without renaming them.','- tom_low: strict-only 6 outputs correspond to tom 3/kick3. With scoring additions, 9 outputs correspond to tom 3/kick 4/snare 2. VFJams alone is 1 tom / 4 kick / 2 snare; NAMM contributes 2 real notated toms. Reclassifying every low tom as kick would therefore also be wrong.','- NAMM snare recall is 5/42, with 16 of 37 missed snares having a nearby kick output, 6 ride_bell, 2 tom_high, 1 hi-hat, 1 other cymbal and 11 nothing. The ADTOF-only baseline is also 5/42. Sensitivity 1.7 raises recall only to 7/42 while precision falls to 7/20; sensitivity 3 gives 13/42 at 13/45 precision. This is not solved by a global sensitivity increase.','- Zildjian snare recall is 41/44 drums-only versus 38/44 full mix. Their 3 and 6 missed snares respectively have only a nearby hi-hat label. VFJams is 17/25 strict, 29/42 including scoring additions. All Caps is 9/9 scoring-only. The instructional Two Shrimps reference is 100/123 and must be kept separate from conclusions about the original performance.','- Seven pre-tom candidates have a _pitch result>=300Hz: five coincide with snare and two with tom. Only two of these are actual downstream pitch calls; five are explicitly marked counterfactual calls for candidates removed beforehand. This does not establish a 420 Hz acoustic fundamental, particularly because Growing Older yielded no accepted truth.','',
            'Concrete detector experiments supported by these counts: preserve low-share candidates for instrument competition instead of unconditional deletion/bell promotion; require separate evidence of a metal instrument before emitting ride_bell; discriminate kick/snare versus tom before assigning floor-tom size. The current count-based bell rule is not sufficient. These are recommendations only: no src code was edited.','']
    lines+=['## Pitch provenance','', '| Tier | Pitch call provenance | Bin | Notation label sets |','|---|---|---|---|']
    for tier in ['strict','scoring']:
        grouped=defaultdict(Counter)
        for v in jd:
            if v['name']=='jd-zildjian-full':continue
            path=root/v['name']/f'correspondence.{tier}.json'
            if not path.exists():continue
            for row in load(path)['toms']:
                grouped[(row['pitch_source'],row['pitch_bin'])]['+'.join(row['truth_labels'])]+=1
        for (source,pitch),row in sorted(grouped.items()):lines.append(f'| {tier} | {source} | {pitch} | {counts(row)} |')
    lines+=['','## Confusion and ghost agreement','',
            'Confusion rows ask which detector family is nearby, preferring a matching family. They can reuse a nearby event and therefore need not equal one-to-one recall. Ghost false means no confirmed printed ghost mark; these are not exhaustive technique annotations.','']
    for v in jd:
        d=v.get('detector',{}).get('scoring',{})
        if not d.get('scores'):continue
        lines+=[f'### {v["name"]}','', '| Notated | Detector nearby |','|---|---|']
        for lab,row in d['confusion'].items():lines.append(f'| {lab} | {counts(row)} |')
        lines+=['','Matched snare ghost flags: '+counts(d['ghost']['agreement'])+'.','']
    lines+=['','## Verification artifacts','',
            '- All systems, including rejects: per-set QUALITY.md and check/system_*.png.','- Visual decisions and scope: jd-inspection/reading-review.json. Source frames, native coordinates and development crops are retained in jd-inspection/.','- Actual evaluate.py invocations: per-set evaluation.strict.txt and evaluation.scoring.txt, including ADTOF baseline, sensitivities, confusion and ghost agreement.','- Instrumented current detector: per-set diagnostics.json, detector-hits.json, tom-candidates.all.json and correspondence.strict/scoring.json.','- Reader regression checks: jd-inspection/tests-final.log (12 reader + 5 tier tests).','- No independent listening/onset annotation or full physical kit-key validation. No inference about a 420 Hz studio-snare fundamental is established by this timed subset.']
    (root/'JD_RESULTS.md').write_text('\n'.join(lines)+'\n')
    # Preserve the prior complete report rather than extending its 190-head audit
    # statistic to the new material.
    previous=(root/'jd-run/SUMMARY.md.before-jd').read_text()
    overview=['# Ground truth status after JD Beck pass','',
              'Detailed per-video reading, tier coverage, detector scores, confusion and tom diagnostics: [JD_RESULTS.md](JD_RESULTS.md). Machine-readable results: [jd-results.json](jd-results.json).','',
              f'Unique JD references: **{totals["jd_unique_reference"]["strict_seconds"]:.2f} strict seconds / {totals["jd_unique_reference"]["strict_notes"]} notes**, plus **{totals["jd_unique_reference"]["scoring_seconds"]-totals["jd_unique_reference"]["strict_seconds"]:.2f} scoring-only seconds / {totals["jd_unique_reference"]["additional_notes"]} notes**. Full-band Zildjian is an additional audio evaluation of transferred truth, not new independent notation.','',
              f'Existing 11 sets retain **{totals["legacy"]["strict_seconds"]:.2f} strict seconds / {totals["legacy"]["strict_notes"]} notes**; scoring tier adds **{totals["legacy"]["scoring_seconds"]-totals["legacy"]["strict_seconds"]:.2f} seconds / {totals["legacy"]["additional_notes"]} notes**. Strict timing gates and existing strict exports are unchanged.','',
              'Growing Older has no accepted truth: the 1080p frame still contains a staff below the 8 px native-spacing gate. Its printed cymbal-bell mapping is recorded, but cannot validate the detector bell hypothesis in that album excerpt.','',
              '## Previous strict rebuild report (preserved; its audit applies only to the older sets)','',previous]
    (root/'SUMMARY.md').write_text('\n'.join(overview))
    oldsummary=load(root/'jd-run/summary.json.before-jd');oldsummary['jd_pass']=dict(totals=totals,videos=jd,legacy_tiers=legacy)
    dump(root/'summary.json',oldsummary)
    print(json.dumps(totals,indent=2))


if __name__=='__main__':
    import sys
    build(Path(sys.argv[1] if len(sys.argv)>1 else 'truth'))
