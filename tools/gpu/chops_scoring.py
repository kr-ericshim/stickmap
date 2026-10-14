"""Comparison-local counters and routine scorer; never opens the vault.

Legacy label-dependent order DP is preserved. Matched estimates inherit the
matched reference's subset, including wrong/additional classes. Unmatched
estimates use reference-time Voronoi regions: nearest reference, earlier on
ties, slow when there are no references. Regions do not depend on predictions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import importlib
import json
from numbers import Integral
from pathlib import Path
import sys

from chops_common import ROOT, vault

sys.path.insert(0, str(ROOT / 'tools/truth_video'))
import fill_seq

CLASSES = ('stroke', *fill_seq.DRUMS)


def validate_counts(counts):
    """Fail before reporting/bootstrap even when Python assertions are disabled."""
    for cls in CLASSES:
        for subset in ('all', 'fast', 'slow'):
            tp, est, ref = [counts.get(f'{subset}.{cls}.{key}', 0) for key in ('tp', 'est', 'ref')]
            if not (all(isinstance(v, Integral) and not isinstance(v, bool) for v in (tp, est, ref))
                    and 0 <= tp <= est and tp <= ref and est >= 0 and ref >= 0):
                raise ValueError(f'invalid {subset}.{cls}: TP={tp}, est={est}, ref={ref}')
        for key in ('tp', 'est', 'ref'):
            if counts.get(f'all.{cls}.{key}', 0) != sum(
                    counts.get(f'{s}.{cls}.{key}', 0) for s in ('fast', 'slow')):
                raise ValueError(f'all != fast + slow: {cls}.{key}')


def tally(ref, est, tol):
    pairs = fill_seq.align(ref, est, tol)
    fast = fill_seq.fast_mask(ref)
    paired_ref, paired_est = dict(pairs), {j: i for i, j in pairs}
    c = Counter()
    for i, (_, labels) in enumerate(ref):
        predicted = est[paired_ref[i]][1] if i in paired_ref else frozenset()
        for subset in ('all', 'fast' if fast[i] else 'slow'):
            c[f'{subset}.stroke.ref'] += 1
            c[f'{subset}.stroke.tp'] += i in paired_ref
            for cls in fill_seq.DRUMS:
                c[f'{subset}.{cls}.ref'] += cls in labels
                c[f'{subset}.{cls}.tp'] += cls in labels and cls in predicted
        for cls in ('tom', 'snare'):
            if labels == {cls}:
                c[f"conf.{cls}->{'+'.join(sorted(predicted)) or 'nothing'}"] += 1
    for j, (time, labels) in enumerate(est):
        i = paired_est.get(j)
        if i is None and ref:
            i = min(range(len(ref)), key=lambda k: (abs(ref[k][0] - time), ref[k][0], k))
        speed = 'fast' if i is not None and fast[i] else 'slow'
        for subset in ('all', speed):
            c[f'{subset}.stroke.est'] += 1
            for cls in fill_seq.DRUMS:
                c[f'{subset}.{cls}.est'] += cls in labels
    validate_counts(c)
    return c


def score_one(task):
    scope, row, detector = task
    seal = vault.Vault()
    if scope == 'solos':
        truth = fill_seq.TRUTH / row['video']
        song = Path((truth / 'song.txt').read_text().strip())
        if seal.contains_song(song):
            return Counter()
    else:
        # Recheck membership in each worker BEFORE reading truth or predictions.
        if seal.contains_song(row):
            return Counter()
        song = vault.SONGS / row['slug']
        truth = song / 'truth'
    if scope == 'strict':
        import fill_score
        _, counts = fill_score.one((row, 'strict', 0.14, detector))
        # Legacy strict calls the non-fast partition "fill".
        for key, value in list(counts.items()):
            if key.startswith('fill.'):
                counts['slow.' + key[5:]] = value
        validate_counts(counts)
        return counts
    try:
        systems = fill_seq.notated(truth, 0.75)
    except FileNotFoundError:
        return Counter()
    if not systems:
        return Counter()
    module, function = detector.split(':')
    hits = getattr(importlib.import_module(module), function)(song)
    counts = Counter()
    for a, b, ref in systems:
        counts.update(tally(ref, fill_seq.detected(hits, a - 0.05, b + 0.05), 0.06))
        counts['systems'] += 1
        counts['seconds'] += b - a
    counts['songs'] += 1
    validate_counts(counts)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('loose', 'strict', 'solos'))
    parser.add_argument('--retrain-plan', type=Path)
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--videos', nargs='+')
    parser.add_argument('--exclude', nargs='*', default=[], help='frozen external-solo harvest peer slugs')
    parser.add_argument('--detect', default='ft_variants:tuned')
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    if args.retrain_plan:
        from chops_common import load_plan
        result = score_retrain(load_plan(args.retrain_plan), args.detect)
        args.json.write_text(json.dumps(result, indent=2) + '\n')
        return
    if args.scope is None:
        parser.error('--scope or --retrain-plan is required')
    if args.scope == 'solos':
        rows = [{'video': v} for v in (args.videos or fill_seq.SOLOS)]
    else:
        from score import songs
        rows = [r for r in songs(args.only, final=False, loose=args.scope == 'loose')
                if r['slug'] not in args.exclude]
    total = Counter()
    with ProcessPoolExecutor(args.jobs) as pool:
        for counts in pool.map(score_one, [(args.scope, row, args.detect) for row in rows]):
            total.update(counts)
    validate_counts(total)
    args.json.write_text(json.dumps(dict(total), indent=2) + '\n')
    for subset in ('all', 'fast', 'slow'):
        print(f'{args.scope} / {subset}: {fill_seq.line(total, subset)}')




def complete_intervals(alignment, strip, system_ids=None):
    """hhscore.py completeness on accepted intervals, without boundary padding.

    Only systems represented in exported truth participate. Match the frozen
    collector's withheld predicate, including tied heads and exportable's True
    default; systems without fitted spans cannot overlap an interval.
    """
    by_id = {s['system']: s for s in strip}
    cymbals = {'hihat', 'ride', 'ride_bell', 'crash', 'cymbal_unknown',
               'unknown', 'bell', 'china', 'splash'}
    bad = []
    for system in alignment['systems']:
        if system_ids is not None and system['system'] not in system_ids:
            continue
        times = system.get('times') or []
        if not times:
            continue
        source = by_id.get(system['system'])
        # A system without its strip cannot prove completeness.
        if source is None or any(not n.get('exportable', True) and n.get('inst', '') in cymbals
                                 for n in source['notes']):
            bad.append([min(times), max(times)])
    return [[a, b] for a, b in alignment['intervals']
            if not any(lo <= b and hi >= a for lo, hi in bad)]


def cymbal_counts(truth, hits, alignment, strip):
    """hhscore.py interval-local, 1-ms distinct refs and greedy 50-ms matching."""
    from align import match
    all_intervals = alignment.get('intervals') or ([[truth[0]['time'], truth[-1]['time']]] if truth else [])
    system_ids = {e['system'] for e in truth} if all('system' in e for e in truth) else None
    intervals = complete_intervals({**alignment, 'intervals': all_intervals}, strip, system_ids)
    output = {}
    for cls, names in [('hihat', {'hihat', 'hihat_open'}), ('cymbal', {'crash', 'ride', 'ride_bell'})]:
        for scope, spans in [('complete', intervals), ('exported', all_intervals)]:
            counts = [0, 0, 0]
            for a, b in spans:
                ref = sorted({round(e['time'], 3) for e in truth
                              if e['inst'] in names and a <= e['time'] <= b})
                est = sorted(h.time for h in hits if h.inst == cls and a - .05 <= h.time <= b + .05)
                for i, value in enumerate((len(match(ref, est, .05)), len(est), len(ref))):
                    counts[i] += value
            output[f'{scope}.{cls}'] = counts
    return output


def retrain_score_song(row, detector):
    """Legacy strict K/S/T counts remain byte-for-byte independent of HH/cymbal."""
    if vault.Vault().contains_song(row):
        raise ValueError('vault song in RETRAIN scorer')
    counts = score_one(('strict', row, detector))
    output = {f'{subset}.{cls}': [counts.get(f'{subset}.{cls}.{k}', 0) for k in ('tp', 'est', 'ref')]
              for subset in ('all', 'fast') for cls in ('kick', 'snare', 'tom')}
    folder = vault.SONGS / row['slug'] / 'truth'
    # No silent fallback to a different truth tier or fabricated complete span.
    truth = json.loads((folder / 'truth.json').read_text())
    alignment = json.loads((folder / 'alignment.json').read_text())
    strip = json.loads((folder / 'strip.json').read_text())
    module, function = detector.split(':')
    hits = getattr(importlib.import_module(module), function)(folder.parent)
    output.update(cymbal_counts(truth, hits, alignment, strip))
    return output


def score_retrain(plan, detector):
    from chops_common import check_plan
    check_plan(plan)
    by_slug = {r['slug']: r for r in plan['references']}
    result = {'A': {}, 'B': {}, 'dev': {}}
    for fold in plan['folds']:
        for scope, key in [('A', 'predict_a'), ('B', 'predict_b')]:
            for slug in fold[key]:
                if slug in result[scope] and scope == 'B':
                    continue  # The detector ensembles predictions from the atthedrum-training folds.
                if slug in result[scope]:
                    raise ValueError('repeated RETRAIN held-out song')
                result[scope][slug] = retrain_score_song(by_slug[slug], detector)
    for row in plan['solos']:
        c = score_one(('solos', row, detector))
        result['dev'][row['slug']] = {f'fast.{cls}': [c.get(f'fast.{cls}.{k}', 0) for k in ('tp', 'est', 'ref')]
                                    for cls in ('kick', 'snare', 'tom')}
    check_plan(plan)
    return result


def retrain_statistics(arms, *, resamples=2000, seed=20261004):
    """arms[BASE|ATH][0|1][song][metric] = integer TP/est/ref.

    Shared song draws across both seeds and arms; pool integers within each
    seed, then average seed metrics. Exported precision/F1 is never reported.
    Undefined bootstrap replicates make the CI unavailable, not optimistic.
    """
    import numpy as np
    from chops_statistics import prf
    if set(arms) != {'BASE', 'ATH'} or any(set(v) != {0, 1} for v in arms.values()):
        raise ValueError('RETRAIN statistics require both arms and seeds 0/1')
    songs = sorted(arms['BASE'][0])
    if any(set(v) != set(songs) for arm in arms.values() for v in arm.values()):
        raise ValueError('RETRAIN paired song rosters differ')
    if not songs:
        return {}
    metrics = sorted(arms['BASE'][0][songs[0]])
    arrays = {}
    for arm, seeds in arms.items():
        if any(set(row) != set(metrics) for v in seeds.values() for row in v.values()):
            raise ValueError('RETRAIN metric roster differs')
        data = np.array([[[seeds[s][song][m] for m in metrics] for song in songs] for s in (0, 1)])
        if (data.shape != (2, len(songs), len(metrics), 3) or not np.issubdtype(data.dtype, np.integer)
                or np.any(data < 0) or np.any(data[..., 0] > data[..., 1]) or np.any(data[..., 0] > data[..., 2])):
            raise ValueError('RETRAIN needs valid integer TP/est/ref')
        arrays[arm] = data
    refs = arrays['BASE'][0, ..., 2]
    if any(not np.array_equal(v[s, ..., 2], refs) for v in arrays.values() for s in (0, 1)):
        raise ValueError('RETRAIN paired reference counts differ')
    draws = np.random.default_rng(seed).multinomial(len(songs), np.full(len(songs), 1 / len(songs)), size=resamples)
    points, samples = {}, {}
    for arm, data in arrays.items():
        points[arm] = 100 * prf(data.sum(axis=1))
        samples[arm] = np.mean([100 * prf((draws @ data[s].reshape(len(songs), -1)).reshape(
            resamples, len(metrics), 3)) for s in (0, 1)], axis=0)
    result = {}
    number = lambda x: float(x) if np.isfinite(x) else None
    for j, metric in enumerate(metrics):
        entry = {'seeds': {}, 'seed_spread': {}}
        recall_only = metric.startswith('exported.')
        for arm in arrays:
            entry['seeds'][arm] = {s: {key: number(points[arm][s, j, k]) for k, key in enumerate(('P', 'R', 'F1'))
                                      if not recall_only or key == 'R'} for s in (0, 1)}
            if not recall_only:
                entry['seed_spread'][arm] = number(abs(points[arm][0, j, 2] - points[arm][1, j, 2]))
        for k, key in enumerate(('P', 'R', 'F1')):
            if recall_only and key != 'R':
                continue
            delta = samples['ATH'][:, j, k] - samples['BASE'][:, j, k]
            valid = np.isfinite(delta)
            entry[key] = {'difference': number(points['ATH'][:, j, k].mean() - points['BASE'][:, j, k].mean()),
                          'ci95': [float(v) for v in np.quantile(delta, [.025, .975])] if valid.all() else [None, None],
                          'valid_resamples': int(valid.sum())}
        result[metric] = entry
    return result


def retrain_decision(scopes):
    """Percentage-point boundaries are inclusive for gains, strict for drops."""
    primary = ['all.' + c for c in ('kick', 'snare', 'tom')] + ['complete.hihat', 'complete.cymbal']
    guards = [('A', m) for m in primary + ['fast.' + c for c in ('kick', 'snare', 'tom')]]
    guards += [('dev', 'fast.' + c) for c in ('kick', 'snare', 'tom')]
    missing, drops = [], []
    for scope, metric in guards:
        value = scopes.get(scope, {}).get(metric, {})
        f = value.get('F1', {})
        gain, ci = f.get('difference'), f.get('ci95', [None, None])
        spreads = [value.get('seed_spread', {}).get(a) for a in ('BASE', 'ATH')]
        if gain is None or any(v is None for v in ci + spreads):
            missing.append(scope + '/' + metric)
            continue
        threshold = max(1., 2 * max(spreads))
        if gain < -threshold - 1e-9 or ci[1] < -.5 - 1e-9:
            drops.append(scope + '/' + metric)
    def gains(scope):
        return [m for m in primary if scopes.get(scope, {}).get(m, {}).get('F1', {}).get('difference') is not None
                and scopes[scope][m]['F1']['difference'] >= 3. - 1e-9]
    a, b = gains('A'), gains('B')
    candidate = bool(a) and not drops and not missing
    return {'candidate': candidate, 'verdict': 'candidate' if candidate else 'inconclusive' if missing else
            'channel fit' if b and not a else 'data only', 'primary': 'held-out A', 'A_gains': a,
            'B_report_only_gains': b, 'drops': drops, 'undefined_guards': missing,
            'spread_policy': 'max(BASE |s0-s1|, ATH |s0-s1|), per metric; threshold=max(1,2*spread)'}


if __name__ == '__main__':
    main()
