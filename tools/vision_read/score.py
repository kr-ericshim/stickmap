"""Score frozen reconciled note gold; never treat unreviewed ink as negative."""
import argparse
import collections
import csv
import datetime
import json
import statistics
import sys
sys.dont_write_bytecode = True
from fractions import Fraction
from pathlib import Path
from common import BENCH, DATA, VERSION, confined, dump, flatten, load, sha, validate


def metric(correct, total, **extra):
    return {'correct': int(correct), 'total': int(total), 'value': correct/total if total else None, **extra}


def inside(head, regions):
    return head.get('x') is not None and head.get('y') is not None and any(a <= head['x'] <= c and b <= head['y'] <= d for a, b, c, d in regions)


def match_heads(gold, pred, spacing, relaxed=False):
    # Instrument, staff position, shape, bar and rhythm are NOT used to match.
    # Thus a wrong label/onset cannot select a more convenient gold head.
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    valid_pred = [i for i, p in enumerate(pred) if p.get('x') is not None and p.get('y') is not None]
    if not gold or not valid_pred:
        return []
    dx = np.abs(np.array([n['x'] for n in gold])[:, None]-np.array([pred[i]['x'] for i in valid_pred])[None, :])
    dy = np.abs(np.array([n['y'] for n in gold])[:, None]-np.array([pred[i]['y'] for i in valid_pred])[None, :])
    allowed = (dx <= max(8 if relaxed else 4, spacing*(1.3 if relaxed else .65))) & (dy <= max(5 if relaxed else 3, spacing*(.6 if relaxed else .4)))
    cost = np.where(allowed, dx/spacing+dy/spacing, 1e6)
    aa, bb = linear_sum_assignment(cost)
    return [(int(i), valid_pred[int(j)]) for i, j in zip(aa, bb) if allowed[i, j]]


def frac_equal(value, quarter_gold):
    if value is None or quarter_gold is None:
        return False
    return Fraction(value)*4 == Fraction(quarter_gold)


def beat_diagnostics(reading):
    """Check claimed beat sums and detected same-voice event overlaps independently."""
    checks = reading.get('beat_checks', [])
    mismatches = []
    for check in checks:
        try:
            total = Fraction(check['sounded_duration'])+Fraction(check['implied_rest_duration'])
            if total != Fraction(check['beat_length']):
                mismatches.append(check)
        except (ValueError, KeyError, TypeError, ZeroDivisionError):
            mismatches.append(check)
    overlaps = 0
    overflows = 0
    signature = reading.get('time_signature') or reading.get('inferred_time_signature')
    length = Fraction(signature['numerator'], signature['denominator']) if isinstance(signature, dict) and signature.get('numerator') and signature.get('denominator') else None
    for bar in reading['bars']:
        voices = collections.defaultdict(list)
        for event in bar['events']:
            if event['onset'] is None or event['duration'] is None or all(h.get('grace') for h in event['heads']):
                continue
            onset, duration = Fraction(event['onset']), Fraction(event['duration'])
            if length is not None and onset+duration > length and not bar['partial_right']:
                overflows += 1
            voices[str(event['voice'])].append((onset, onset+duration))
        for spans in voices.values():
            spans = sorted(set(spans))  # simultaneous identical-duration heads are a chord
            overlaps += sum(b[0] < a[1] for a, b in zip(spans, spans[1:]))
    return {'reported_checks': len(checks), 'claim_sum_mismatches': len(mismatches), 'same_voice_overlaps': overlaps, 'bar_overflows': overflows,
            'limitation': 'Without a visible/declared meter, missing beat checks are not automatically verified. Implied rests cannot prove the image was read correctly.'}


def score_system(item, gold, reading, result=None, relaxed=False):
    notes = [n for n in gold.get('notes', []) if not n.get('uncertain')]
    pred = flatten(reading) if reading else []
    lines = gold.get('staff_lines')
    spacing = (lines[-1]-lines[0])/4 if lines else 12
    pairs = match_heads(notes, pred, spacing, relaxed)
    mapping = dict(pairs)
    matched = {j for _, j in pairs}
    regions, ignored = gold.get('exhaustive_regions', []), gold.get('uncertain_regions', [])
    reviewed = lambda p: inside(p, regions) and not inside(p, ignored)
    fp_indices = [j for j, p in enumerate(pred) if j not in matched and reviewed(p)]
    precision_tp = sum(reviewed(pred[j]) for _, j in pairs)
    head = metric(len(pairs), len(notes), tp=len(pairs), fn=len(notes)-len(pairs), fp=len(fp_indices), precision_tp=precision_tp,
                  precision=precision_tp/(precision_tp+len(fp_indices)) if precision_tp+len(fp_indices) else None)
    metrics = {'head': head}
    failures = {'missing_head': [n['id'] for i, n in enumerate(notes) if i not in mapping], 'extra_head': [pred[j]['id'] for j in fp_indices]}
    definitions = [('pos', 'position', 'pos'), ('inst', 'instrument', 'instrument'), ('shape', 'shape', 'notehead'),
                   ('duration_q', 'duration', 'duration'), ('onset_q', 'onset', 'onset'), ('tuplet', 'tuplet_ratio', 'tuplet'),
                   ('tied_continuation', 'tie', 'tie_continuation'), ('ghost', 'ghost', 'ghost'), ('accent', 'accent', 'accent')]
    for field, name, pfield in definitions:
        total = correct = conditional_correct = conditional_total = positive_total = positive_correct = false_positive = 0
        failed = []
        for i, note in enumerate(notes):
            if note.get(field) is None:
                continue
            total += 1
            p = pred[mapping[i]] if i in mapping else {}
            if field in ('duration_q', 'onset_q'):
                okay = frac_equal(p.get(pfield), note[field]) and (field != 'onset_q' or p.get('bar') == note.get('bar'))
            elif field == 'tuplet':
                okay = isinstance(p.get(pfield), dict) and p[pfield].get('ratio') == note[field]
            else:
                okay = i in mapping and p.get(pfield) == note[field]
            correct += okay
            if i in mapping:
                conditional_total += 1
                conditional_correct += okay
            if type(note[field]) is bool:
                positive_total += note[field]
                positive_correct += note[field] and okay
                false_positive += i in mapping and not note[field] and p.get(pfield) is True
            if not okay:
                failed.append({'gold_id': note['id'], 'expected': note[field], 'predicted': p.get(pfield), 'pred_bar': p.get('bar')})
        metrics[name] = metric(correct, total, conditional_correct=conditional_correct, conditional_total=conditional_total)
        if field in ('tied_continuation', 'ghost', 'accent'):
            metrics[name].update(positive_correct=int(positive_correct), positive_total=int(positive_total), false_positive=int(false_positive))
        failures[name] = failed
    tuplets = [i for i, n in enumerate(notes) if n.get('tuplet') is not None]
    metrics['tuplet_membership'] = metric(sum(i in mapping and pred[mapping[i]].get('tuplet') is not None for i in tuplets), len(tuplets))
    attacks = [i for i, n in enumerate(notes) if n.get('onset_q') is not None and not n.get('tied_continuation') and not n.get('grace')]
    metrics['attack_onset'] = metric(sum(i in mapping and frac_equal(pred[mapping[i]].get('onset'), notes[i]['onset_q']) and pred[mapping[i]]['bar'] == notes[i]['bar'] and pred[mapping[i]].get('tie_continuation') is False for i in attacks), len(attacks))
    staff = reading['staff'] if reading else {}
    plines = staff.get('lines_y') or []
    metrics['staff_found'] = metric(lines is not None and staff.get('line_count') == 5, int(lines is not None))
    metrics['staff_lines'] = metric(lines is not None and len(plines) == 5 and max(abs(a-b) for a, b in zip(lines, plines)) <= max(1.5, spacing*.12), int(lines is not None))
    target = gold.get('clef_visible')
    okay = target is not None and reading is not None and staff.get('clef_visible') == target
    metrics['clef'] = metric(okay, int(target is not None), positive_correct=int(okay and target is True), positive_total=int(target is True), false_positive=int(staff.get('clef_visible') is True and target is False))
    complete = gold.get('complete', False)
    expected_bars = max(n['bar'] for n in notes if n.get('bar') is not None)+1 if complete else None
    metrics['bar_count'] = metric(reading is not None and reading['bar_count'] == expected_bars, int(expected_bars is not None), gold=expected_bars, predicted=reading['bar_count'] if reading else None)
    # Geometry gold lists barlines, not bar counts. Score the boundaries directly
    # on partial systems; never infer partial bar counts from sparse notes.
    boundaries = []
    if reading:
        width = item.get('image_width', item['variants'][item['primary']].get('width', 1920))
        for bar in reading['bars']:
            for key in ('x_start', 'x_end'):
                x = bar.get(key)
                if isinstance(x, (int, float)) and 2 < x < width and not any(abs(x-b) <= spacing*.3 for b in boundaries):
                    boundaries.append(x)
    target_boundaries = gold.get('bars')
    if target_boundaries is not None:
        available = set(range(len(boundaries)))
        hits = 0
        for x in target_boundaries:
            choices = [j for j in available if abs(x-boundaries[j]) <= max(4, spacing*.65)]
            if choices:
                available.remove(min(choices, key=lambda j: abs(x-boundaries[j])))
                hits += 1
        metrics['barline'] = metric(hits, len(target_boundaries), false_positive=len(available))
    else:
        metrics['barline'] = metric(0, 0)
    by_instrument = collections.defaultdict(lambda: {'tp': 0, 'fn': 0, 'precision_tp': 0, 'fp': 0})
    for i, note in enumerate(notes):
        instrument = note.get('inst')
        if instrument is None:
            continue
        p = pred[mapping[i]] if i in mapping else None
        okay = p is not None and p.get('instrument') == instrument
        counts = by_instrument[instrument]
        counts['tp' if okay else 'fn'] += 1
        if p is not None and reviewed(p):
            if okay:
                counts['precision_tp'] += 1
            elif p.get('instrument') is not None:
                by_instrument[p['instrument']]['fp'] += 1
    for j in fp_indices:
        instrument = pred[j].get('instrument')
        if instrument is not None:
            by_instrument[instrument]['fp'] += 1
    required = ['head', 'position', 'instrument', 'shape', 'duration', 'onset', 'tie', 'ghost', 'accent', 'tuplet_ratio', 'bar_count', 'clef', 'staff_lines']
    system_ok = all(metrics[k]['correct'] == metrics[k]['total'] for k in required) and not fp_indices and reading is not None
    metrics['known_fields_system_correct'] = metric(complete and system_ok, int(complete))
    # An independent symbolic check on COMPLETE inventories avoids making
    # pixel localization a prerequisite for assessing the musical output.
    # Multisets preserve chords and duplicate heads; no gold-guided alignment.
    for name, include_onset, include_duration in [('full_symbolic_inventory', False, False), ('full_symbolic_onset', True, False), ('full_symbolic_onset_duration', True, True)]:
        def gold_key(n):
            key = (n.get('bar'), n.get('pos'), n.get('shape'), bool(n.get('tied_continuation')))
            if include_onset:
                key += (Fraction(n['onset_q'])/4 if n.get('onset_q') is not None else None,)
            if include_duration:
                key += (Fraction(n['duration_q'])/4 if n.get('duration_q') is not None else None,)
            return key
        def pred_key(n):
            key = (n.get('bar'), n.get('pos'), n.get('notehead'), n.get('tie_continuation'))
            if include_onset:
                key += (Fraction(n['onset']) if n.get('onset') is not None else None,)
            if include_duration:
                key += (Fraction(n['duration']) if n.get('duration') is not None else None,)
            return key
        targets = collections.Counter(gold_key(n) for n in notes) if complete else collections.Counter()
        predictions = collections.Counter(pred_key(p) for p in pred) if complete else collections.Counter()
        hits = sum((targets & predictions).values())
        metrics[name] = metric(hits, sum(targets.values()), fp=sum(predictions.values())-hits,
                               precision_tp=hits, precision=hits/sum(predictions.values()) if predictions else None)
    return {'id': item['id'], 'video': item['video'], 'channel': item['channel'], 'complete': complete, 'gold_notes': len(notes),
            'status': (result or {}).get('status', 'ok' if reading else 'missing'), 'wall_seconds': (result or {}).get('wall_seconds'),
            'first_attempt_seconds': (result or {}).get('attempts', [{}])[0].get('seconds') if (result or {}).get('attempts') else None,
            'metrics': metrics, 'by_instrument': dict(by_instrument), 'matches': [[notes[i]['id'], pred[j]['id']] for i, j in pairs],
            'failures': failures, 'beat_diagnostics': beat_diagnostics(reading) if reading else None,
            'predicted_heads': len(pred), 'unlocated_heads': sum(p.get('x') is None or p.get('y') is None for p in pred),
            'uncertain_count': len(reading['uncertain']) if reading else 0}


def aggregate(rows):
    names = sorted({k for r in rows for k in r['metrics']})
    metrics = {}
    for name in names:
        mm = [r['metrics'][name] for r in rows]
        metrics[name] = metric(sum(m['correct'] for m in mm), sum(m['total'] for m in mm))
        for field in ('conditional_correct', 'conditional_total', 'positive_correct', 'positive_total', 'false_positive', 'tp', 'fn', 'fp', 'precision_tp'):
            if any(field in m for m in mm):
                metrics[name][field] = sum(m.get(field, 0) for m in mm)
        if name == 'head' or name.startswith('full_symbolic_'):
            denominator = metrics[name]['precision_tp']+metrics[name]['fp']
            metrics[name]['precision'] = metrics[name]['precision_tp']/denominator if denominator else None
    instruments = {}
    for instrument in sorted({k for r in rows for k in r['by_instrument']}):
        counts = {f: sum(r['by_instrument'].get(instrument, {}).get(f, 0) for r in rows) for f in ('tp', 'fn', 'fp', 'precision_tp')}
        counts['recall'] = counts['tp']/(counts['tp']+counts['fn']) if counts['tp']+counts['fn'] else None
        counts['precision'] = counts['precision_tp']/(counts['precision_tp']+counts['fp']) if counts['precision_tp']+counts['fp'] else None
        instruments[instrument] = counts
    seconds = [r['wall_seconds'] for r in rows if r['wall_seconds'] is not None]
    first_seconds = [r['first_attempt_seconds'] for r in rows if r.get('first_attempt_seconds') is not None]
    failure_counts = {name: sum(r['metrics'][name]['correct'] < r['metrics'][name]['total'] for r in rows) for name in names}
    return {'systems': len(rows), 'note_gold_systems': sum(r['gold_notes'] > 0 for r in rows), 'metrics': metrics, 'by_instrument': instruments,
            'time_seconds': {'mean': statistics.mean(seconds) if seconds else None, 'median': statistics.median(seconds) if seconds else None,
                             'min': min(seconds) if seconds else None, 'max': max(seconds) if seconds else None, 'total_worker': sum(seconds)},
            'first_attempt_time_seconds': {'mean': statistics.mean(first_seconds) if first_seconds else None, 'median': statistics.median(first_seconds) if first_seconds else None},
            'statuses': dict(collections.Counter(r['status'] for r in rows)), 'failure_system_counts': failure_counts,
            'uncertain_count': sum(r['uncertain_count'] for r in rows), 'unlocated_heads': sum(r['unlocated_heads'] for r in rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', required=True, type=Path, help='run name or path beneath .eval-cache/vision_pilot/runs')
    ap.add_argument('--bench', type=Path, default=BENCH)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    folder = confined(a.run if a.run.is_absolute() else DATA/'runs'/a.run)
    out = confined(a.out or folder/'scores')
    config = load(folder/'config.json')
    manifest = load(a.bench/'manifest.json')
    goldfile = a.bench/'gold.json'
    gold = load(goldfile)
    items = {it['id']: it for it in manifest['systems']}
    rows, relaxed_rows = [], []
    for sid in config['systems']:
        job = folder/sid
        reading = None
        result = load(job/'result.json') if (job/'result.json').exists() else {'status': 'missing'}
        if result.get('status') == 'ok':
            reading = validate(load(job/'reading.json'), sid)
            if result.get('reading_sha256') != sha(job/'reading.json'):
                raise ValueError(f'reading changed after run: {sid}')
        item = {**items[sid], 'image_width': load(job/'job.json')['images'][0]['width']} if (job/'job.json').exists() else items[sid]
        rows.append(score_system(item, gold['systems'][sid], reading, result))
        relaxed_rows.append(score_system(item, gold['systems'][sid], reading, result, relaxed=True))
    journals = {p.name: sha(p) for p in (a.bench/'annotations').glob('*.json')}
    summary = {'version': VERSION, 'scoring_version': 'native-xy-v1', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'config': config, 'manifest_sha256': sha(a.bench/'manifest.json'), 'gold_sha256': sha(goldfile), 'annotation_sha256': journals,
               'scorer_sha256': sha(__file__), 'common_sha256': sha(Path(__file__).with_name('common.py')), 'gold_coverage': gold['coverage'],
               'scope': 'provisional same-reviewer partial gold; primary native viewport only; no audio timing or generalization claims',
               'matching': {'method': 'geometry-only one-to-one Hungarian, independent of instrument/rhythm', 'x_tolerance': 'max(4px,0.65 staff-space)', 'y_tolerance': 'max(3px,0.4 staff-space)',
                            'precision_scope': 'exhaustive gold regions only; matches to unknown instrument gold omitted from instrument precision',
                            'onset_units': 'model whole-note fractions multiplied by 4 to compare gold quarter-note fractions',
                            'tuplet_scope': 'known positive member heads and exact ratio; group boundary correctness not independently annotated',
                            'bar_count_scope': 'six complete-visible-note systems only; partial gold barlines scored separately',
                            'known_fields_system_correct_scope': 'known scored fields only, excludes implied-rest correctness'},
               'summary': aggregate(rows), 'relaxed_geometry_sensitivity': aggregate(relaxed_rows),
               'by_channel': {c: aggregate([r for r in rows if r['channel'] == c]) for c in sorted({r['channel'] for r in rows})}, 'systems': rows}
    if (folder/'run.json').exists():
        summary['batch_wall_seconds'] = load(folder/'run.json')['batch_wall_seconds']
    dump(out.with_suffix('.json'), summary)
    instruments = sorted({inst for row in rows for inst in row['by_instrument']})
    names = ['id', 'channel', 'complete', 'gold_notes', 'status', 'wall_seconds', 'first_attempt_seconds', 'predicted_heads', 'uncertain_count',
             'head_precision', 'head_fp', 'head_fn', 'bar_count_gold', 'bar_count_predicted']+[name+'_'+f for name in sorted(rows[0]['metrics']) for f in ('correct', 'total')]
    names += [f'inst_{inst}_{field}' for inst in instruments for field in ('tp', 'fn', 'fp', 'precision_tp', 'recall', 'precision')]
    with out.with_suffix('.tsv').open('w') as stream:
        writer = csv.DictWriter(stream, names, delimiter='\t')
        writer.writeheader()
        for row in rows:
            instrument_values = {}
            for inst in instruments:
                counts = row['by_instrument'].get(inst, {'tp': 0, 'fn': 0, 'fp': 0, 'precision_tp': 0})
                counts = {**counts, 'recall': counts['tp']/(counts['tp']+counts['fn']) if counts['tp']+counts['fn'] else None,
                          'precision': counts['precision_tp']/(counts['precision_tp']+counts['fp']) if counts['precision_tp']+counts['fp'] else None}
                instrument_values.update({f'inst_{inst}_{field}': value for field, value in counts.items()})
            writer.writerow({**{k: row[k] for k in names if k in row},
                             'head_precision': row['metrics']['head']['precision'], 'head_fp': row['metrics']['head']['fp'], 'head_fn': row['metrics']['head']['fn'],
                             'bar_count_gold': row['metrics']['bar_count']['gold'], 'bar_count_predicted': row['metrics']['bar_count']['predicted'],
                             **{name+'_'+f: m[f] for name, m in row['metrics'].items() for f in ('correct', 'total')}, **instrument_values})
    print(json.dumps(summary['summary'], indent=2))


if __name__ == '__main__':
    main()
