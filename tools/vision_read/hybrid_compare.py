"""Same frozen scorer for CV, hybrid, and the two retained Gemini baselines."""
import argparse
import collections
import copy
import json
import statistics
import sys
sys.dont_write_bytecode = True
from fractions import Fraction
from common import BENCH, DATA, confined, dump, flatten, load, sha, validate
from hybrid import HERE, HYBRID, cv_reading, unsealed
from score import aggregate, frac_equal, match_heads, metric, score_system


def extra_metrics(item, gold, reading):
    notes = [n for n in gold.get('notes', []) if not n.get('uncertain')]
    pred = flatten(reading) if reading else []
    lines = gold.get('staff_lines') or [0, 12, 24, 36, 48]
    pairs = dict(match_heads(notes, pred, (lines[-1]-lines[0])/4))
    eligible = [i for i, n in enumerate(notes) if n.get('inst') is not None and n.get('onset_q') is not None]
    joint = metric(sum(i in pairs and pred[pairs[i]].get('instrument') == notes[i]['inst'] and
                       pred[pairs[i]]['bar'] == notes[i]['bar'] and frac_equal(pred[pairs[i]].get('onset'), notes[i]['onset_q'])
                       for i in eligible), len(eligible))
    complete = gold.get('complete', False)
    def key(n, target, inst=False, all_fields=False):
        onset = n.get('onset_q' if target else 'onset')
        base = (n.get('bar'), n.get('pos'), n.get('shape' if target else 'notehead'), n.get('tied_continuation' if target else 'tie_continuation'),
                Fraction(onset)/(4 if target else 1) if onset is not None else None)
        if all_fields:
            duration = n.get('duration_q' if target else 'duration')
            tuplet = n.get('tuplet')
            base += (Fraction(duration)/(4 if target else 1) if duration is not None else None,
                     n.get('ghost'), n.get('accent'), tuplet if target else tuplet.get('ratio') if tuplet else None)
        if inst:
            base += (n.get('inst' if target else 'instrument'),)
        return base
    targets = collections.Counter(key(notes[i], True, inst=True) for i in eligible) if complete else collections.Counter()
    predictions = collections.Counter(key(p, False, inst=True) for p in pred) if complete else collections.Counter()
    joint_symbolic = metric(sum((targets & predictions).values()), sum(targets.values()))
    # Position/shape/onset define an instrument-independent head; inventory
    # equality rejects unlocated extra misses too, unlike region-only XY FP.
    full_gold = collections.Counter(key(n, True, all_fields=True) for n in notes)
    full_pred = collections.Counter(key(n, False, all_fields=True) for n in pred)
    known_inst_gold = collections.Counter(key(notes[i], True, inst=True, all_fields=True) for i in eligible)
    known_inst_pred = collections.Counter(key(n, False, inst=True, all_fields=True) for n in pred)
    symbolic_exact = complete and reading is not None and full_gold == full_pred and sum((known_inst_gold & known_inst_pred).values()) == len(eligible)
    return {'onset_instrument': joint, 'symbolic_onset_instrument': joint_symbolic,
            'symbolic_known_fields_system_exact': metric(symbolic_exact, int(complete))}


def evaluate(name, systems, items, gold, hybrid_run):
    rows = []
    for sid in systems:
        item = items[sid]
        unsealed(item['video'])
        reading, result = None, {'status': 'missing', 'wall_seconds': None}
        if name == 'cv-only':
            cv = load(HYBRID/'jobs'/sid/'cv.json')
            reading = cv_reading(sid, cv)
            result = {'status': 'ok', 'wall_seconds': cv['wall_seconds']}
        else:
            folder = DATA/'runs'/(hybrid_run if name == 'hybrid' else name)/sid
            if (folder/'result.json').exists():
                result = load(folder/'result.json')
                if result['status'] == 'ok':
                    reading = validate(load(folder/'reading.json'), sid)
                    if result['reading_sha256'] != sha(folder/'reading.json'):
                        raise ValueError('reading digest mismatch: '+str(folder))
        row = score_system(item, gold[sid], reading, result)
        row['metrics'].update(extra_metrics(item, gold[sid], reading))
        rows.append(row)
    completed = [r for r in rows if r['status'] == 'ok']
    full = [r for r in rows if r['complete']]
    measured = not any(r['status'] in ('missing', 'quota_cancelled', 'quota_exhausted') for r in rows)
    return {'all': aggregate(rows), 'complete_gold': aggregate(full), 'completed_only': aggregate(completed),
            'accuracy_measured': measured, 'systems': rows,
            'interpretation': 'Missing/quota jobs are operational failures, not zero model accuracy. Completed-only results can have selection bias.'}


def pct(value):
    return 'N/A' if value is None else f'{value*100:.1f}%'


def count(m):
    return f"{m['correct']}/{m['total']} ({pct(m['value'])})"


def should_extend(expected_ids, complete_rows, hybrid, baselines):
    if not expected_ids or {r['id'] for r in complete_rows} != set(expected_ids) or any(r['status'] != 'ok' for r in complete_rows):
        return False
    return ((hybrid['symbolic_onset_instrument']['value'] or 0) >= .85 or
            hybrid['known_fields_system_correct']['correct'] > max(b['known_fields_system_correct']['correct'] for b in baselines) or
            (hybrid['full_symbolic_onset']['value'] or 0) >= max(b['full_symbolic_onset']['value'] or 0 for b in baselines)+.10)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default='hybrid-medium-look')
    ap.add_argument('--systems', nargs='+')
    ap.add_argument('--out', default='HYBRID_REPORT.md')
    a = ap.parse_args()
    configfile = DATA/'runs'/a.run/'config.json'
    systems = a.systems or (load(configfile)['systems'] if configfile.exists() else load(DATA/'selection.json')['gold'])
    items = {it['id']: it for it in load(BENCH/'manifest.json')['systems']}
    for sid in systems:
        unsealed(items[sid]['video'])
    gold = load(BENCH/'gold.json')['systems']
    conditions = {name: evaluate(name, systems, items, gold, a.run) for name in ('cv-only', 'high-look', 'medium-look', 'hybrid')}
    hs = conditions['hybrid']['complete_gold']['metrics']
    cs = conditions['cv-only']['complete_gold']['metrics']
    ms = conditions['medium-look']['complete_gold']['metrics']
    high = conditions['high-look']['complete_gold']['metrics']
    complete_rows = [r for r in conditions['hybrid']['systems'] if r['complete']]
    complete_ids = {sid for sid, g in gold.items() if g.get('complete')}
    success = should_extend(complete_ids, complete_rows, hs, [cs, ms, high])
    report = {'version': 'hybrid-comparison-v1', 'run': a.run, 'systems': systems, 'conditions': conditions,
              'manifest_sha256': sha(BENCH/'manifest.json'), 'gold_sha256': sha(BENCH/'gold.json'),
              'scorer_sha256': sha(HERE/'score.py'), 'comparison_sha256': sha(__file__),
              'extension_recommended': success, 'success_rule': 'all complete gold readings completed; symbolic onset+instrument >=85%, or whole-system improvement over all baselines, or >=10 point symbolic onset gain over all baselines',
              'scope': 'provisional same-reviewer gold; 21 partial/6 complete viewports; no audio or vault scoring'}
    dump(DATA/'hybrid_summary.json', report)
    lines = ['# Hybrid notation reader evaluation', '', report['scope'], '',
             '| Condition | XY heads recall / precision | XY onset | XY instrument | Coordinate-free onset, complete | Joint onset+instrument, complete XY / symbolic | Whole systems XY / symbolic | mean min/system |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, condition in conditions.items():
        if not condition['accuracy_measured']:
            lines.append(f"| {name} | pending / quota censored | pending | pending | pending | pending | pending | N/A |")
            continue
        metrics = condition['all']['metrics']; full = condition['complete_gold']['metrics']
        sec = condition['all']['time_seconds']['mean']
        lines.append(f"| {name} | {pct(metrics['head']['value'])} / {pct(metrics['head']['precision'])} | {count(metrics['onset'])} | {count(metrics['instrument'])} | {count(full['full_symbolic_onset'])} | {count(full['onset_instrument'])} / {count(full['symbolic_onset_instrument'])} | {count(full['known_fields_system_correct'])} / {count(full['symbolic_known_fields_system_exact'])} | {sec/60:.3f} |")
    lines += ['', 'XY matching and symbolic multisets are the existing score.py, unchanged. Joint accuracy is an additional metric, evaluated on known instrument+onset gold (112 heads in complete inventories); missing heads count as errors. Symbolic joint includes bar, position, shape and tie state. Symbolic exact additionally requires duration, ghost/accent, positive tuplet ratios and all known instruments, and counts every added head even when unlocated.',
              '', 'Complete inventories have 159 heads; verified instruments cover 112 of them. The 21-system XY onset denominator is 280 and instrument denominator 222. Do not compare conditional detected-head onset (91.5% for medium) with end-to-end recall.',
              '', '## Positive technique and tuplets', '', '| Condition | Tuplet ratio | Tie positives | Ghost positives | Accent positives |', '|---|---:|---:|---:|---:|']
    for name, condition in conditions.items():
        mm = condition['all']['metrics']
        if not condition['accuracy_measured']:
            lines.append(f'| {name} | pending | pending | pending | pending |')
            continue
        values = [count(mm['tuplet_ratio'])]+[f"{mm[k].get('positive_correct', 0)}/{mm[k].get('positive_total', 0)}" for k in ('tie', 'ghost', 'accent')]
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    lines += ['', '## Per complete system', '', '| System | gold heads | CV heads | hybrid status / heads | hybrid symbolic onset | hybrid joint symbolic |', '|---|---:|---:|---:|---:|---:|']
    cv_rows = {r['id']: r for r in conditions['cv-only']['systems']}
    for r in complete_rows:
        mm = r['metrics']
        lines.append(f"| {r['id']} | {r['gold_notes']} | {cv_rows[r['id']]['predicted_heads']} | {r['status']} / {r['predicted_heads']} | {count(mm['full_symbolic_onset']) if r['status']=='ok' else 'pending'} | {count(mm['symbolic_onset_instrument']) if r['status']=='ok' else 'pending'} |")
    lines += ['', '## Decision', '', f'Extension criterion met: {success}.', '',
              'CV instrument/geometry are frozen. Gemini-only misses always have instrument=null and unverified-instrument; approximate or absent coordinates are retained without gold-guided snapping. False positives are preserved in rhythm.json and rejected_cv_heads. No output is strict truth until independent notation/anchor review and existing alignment/tier gates pass.',
              '', 'Look is a prompt policy: original once, annotated once, no other tools. Text CLI logs do not certify an exhaustive tool trace. Inference shares agy.lock with deferred and baseline jobs, uses at most four workers, and stops pending/in-flight calls on first quota error.',
              '', 'Raw artifacts: hybrid/jobs/<system>/{original.png,annotated.png,cv.json,annotation.json}; runs/'+a.run+'/<system>/{rhythm.json,reading.json,result.json,*log}; hybrid_summary.json.']
    confined(DATA/a.out).write_text('\n'.join(lines)+'\n')
    # Append the pre-reviewed rollout plan only when real complete-gold results
    # meet the rule. A pending/quota result can never trigger this branch.
    if success:
        readme = HERE/'README.md'
        marker = '## Hybrid WSL rollout after measured improvement'
        if marker not in readme.read_text():
            readme.write_text(readme.read_text().rstrip()+'\n\n'+(HERE/'HYBRID_WSL_PLAN.md').read_text())
    print(json.dumps({'report': str(DATA/a.out), 'extension_recommended': success, 'hybrid_statuses': conditions['hybrid']['all']['statuses']}))


if __name__ == '__main__':
    main()
