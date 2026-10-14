"""Child scorer for one durable vault opening, paired A–D ensembles.

Reuses the existing loose order-DP metric. Complete K/S/T audit intervals
form the primary metric; partial notation is reported separately. No models
are loaded here, and opening IDs must belong to the current parent runner.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import json
from pathlib import Path
import sys

from chops_common import ROOT, check_performance_table, load_plan, performance_groups, promotion_policy, sha, vault
from chops_scoring import tally
from chops_statistics import CLASSES, count_array, paired_bootstrap, prf, sufficiency
from harvest_transfer import atomic

sys.path[:0] = [str(ROOT / 'tools/truth_video'), str(ROOT / 'tools/gpu/chops_recipe')]


def score_song(row, tag):
    import fill_seq
    import ft_variants
    from three_state import validate_audit
    song = vault.SONGS / row['slug']
    ft_variants.TAG = tag
    hits = ft_variants.tuned(song)
    complete, partial = Counter(), Counter()
    unique_fast = {scope: {c: set() for c in CLASSES} for scope in ('complete', 'partial')}
    groups = fill_seq.notated(song / 'truth', 0.75)
    audit_path = song / 'truth/kst_audit.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    validate_audit(audit)
    complete_intervals = [] if audit is None else audit.get('complete_intervals', [])
    # Select whole systems only: clipping a partial system could change speed
    # definitions, alignments, or make a border prediction an invented FP.
    for a, b, ref in groups:
        c = tally(ref, fill_seq.detected(hits, a - 0.05, b + 0.05), 0.06)
        audited = any(r['start'] <= a - 0.05 and b + 0.05 <= r['end'] for r in complete_intervals)
        (complete if audited else partial).update(c)
        scope = 'complete' if audited else 'partial'
        for (time, classes), fast in zip(ref, fill_seq.fast_mask(ref)):
            if fast:
                for cls in classes:
                    unique_fast[scope][cls].add(round(time, 4))
    return {'complete': dict(complete), 'partial': dict(partial),
            'unique_fast_moments': {
                **{scope: {c: len(times) for c, times in values.items()} for scope, values in unique_fast.items()},
                'all_notation': {c: len(unique_fast['complete'][c] | unique_fast['partial'][c]) for c in CLASSES}},
            'complete_systems': sum(any(r['start'] <= a - 0.05 and b + 0.05 <= r['end'] for r in complete_intervals)
                                    for a, b, _ in groups)}


def score_one(task):
    arm, row, tag = task
    return arm, row['slug'], score_song(row, tag)


def summarize(plan, rows, results, *, excluded=None):
    import numpy as np
    excluded = excluded or {}
    excluded_slugs = {r['slug'] for r in excluded.values() if r.get('slug')}
    rows = [r for r in rows if (r.get('video') or r.get('video_id')) not in excluded
            and r['slug'] not in excluded_slugs]
    selected = {r['slug'] for r in rows}
    results = {arm: {slug: counts for slug, counts in values.items() if slug in selected}
               for arm, values in results.items()}
    groups = performance_groups(rows, plan['same_performance'])
    outputs = {}
    for scope in ('complete', 'partial', 'all_notation'):
        arms = {a: {} for a in 'ABCD'}
        raw_fast = {c: 0 for c in CLASSES}
        group_fast = {}
        unique_by_video = {}
        for row in rows:
            slug, group = row['slug'], groups[row['slug']]
            for arm in 'ABCD':
                result = results[arm][slug]
                counts = (Counter(result['complete']) + Counter(result['partial'])
                          if scope == 'all_notation' else result[scope])
                values = count_array(counts)
                if values[..., 2].sum() == 0:
                    continue
                arms[arm].setdefault(group, np.zeros((2, 3, 3), dtype=np.int64))
                arms[arm][group] += values
                if arm == 'A':
                    unique = result.get('unique_fast_moments')
                    fast = np.array([unique[scope][c] for c in CLASSES]) if unique else values[1, :, 2]
                    for c, value in zip(CLASSES, values[1, :, 2]):
                        raw_fast[c] += int(value)
                    unique_by_video[slug] = {c: int(value) for c, value in zip(CLASSES, fast)}
                    group_fast[group] = np.maximum(group_fast.get(group, np.zeros(3, dtype=int)), fast)
        conservative_fast = {c: int(sum(v[k] for v in group_fast.values())) for k, c in enumerate(CLASSES)}
        if not arms['A']:
            outputs[scope] = {'available': False, 'reason': 'no eligible references',
                              'sufficiency': sufficiency({}, 0, {})}
            continue
        summaries = {}
        for arm in 'ABCD':
            counts = sum(arms[arm].values())
            scores = prf(counts)
            summaries[arm] = {s: {c: {'counts': counts[i, k].tolist(),
                                      'prf': [float(v) if np.isfinite(v) else None for v in scores[i, k]]}
                                  for k, c in enumerate(CLASSES)} for i, s in enumerate(('all', 'fast'))}
        outputs[scope] = {'available': True, 'arms': summaries, 'bootstrap': paired_bootstrap(arms),
                          'raw_fast_reference_moments': raw_fast,
                          'per_video_unique_fast_moments': unique_by_video,
                          'sufficiency': sufficiency(conservative_fast, len(arms['A']),
                              {c: sum(v[k] > 0 for v in group_fast.values()) for k, c in enumerate(CLASSES)}),
                          'count_policy': 'sum of max per-video counts within each performance group; conservative for overlapping clips'}
    return {'plan': plan.get('name', 'legacy'), 'ensemble': plan.get('final_policy', {}).get('ensemble', 'legacy single tag'),
            'primary': 'complete', 'scopes': outputs, 'per_song': results, 'excluded': excluded,
            'promotion': promotion_policy(plan),
            'performance_groups': groups,
            'caveats': ['Partial-notation P/R/F measures notation agreement; unwritten real strokes can count as FP.',
                        'Bootstrap conditions on the frozen seed/fold ensemble, not random training-seed uncertainty.',
                        'Repeated or clipped notation within a recording is correlated; unknown cross-video offsets prevent exact deduplication.',
                        '10000 resamples do not increase the number of independent performances.']}


def write_report(result, output):
    atomic(output, (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())
    lines = [f'# {result["plan"]}: single opening, paired A–D vault comparison', '',
             f'Ensemble: {result["ensemble"]}. Primary scope: complete K/S/T audited systems.', '',
             'Promotion-eligible arms: ' + ', '.join(result['promotion']['eligible_arms']) + '.',
             result['promotion']['D']['label'] + ': ' + result['promotion']['D']['reason'] + '.',
             'D contrasts are diagnostic when D is excluded from promotion. D–C changes multiple loss policies.', '',
             '| Scope | Arm | Subset | Kick P/R/F | Snare P/R/F | Tom P/R/F |',
             '|---|---|---|---|---|---|']
    exclusions = [f"{v}: {item['reason']}" for v, item in sorted(result.get('excluded', {}).items())]
    def cell(values):
        return ' / '.join('—' if v is None else f'{100 * v:.2f}' for v in values)
    unavailable = []
    def label(arm):
        return result['promotion']['D']['label'] if arm == 'D' else arm
    for scope, data in result['scopes'].items():
        if not data['available']:
            unavailable.append(f'{scope}: unavailable ({data["reason"]}).')
            continue
        for arm, subsets in data['arms'].items():
            for subset, classes in subsets.items():
                lines.append(f'| {scope} | {label(arm)} | {subset} | ' +
                             ' | '.join(cell(classes[c]['prf']) for c in CLASSES) + ' |')
    lines += ['', *exclusions, *unavailable, '', '| Scope | Difference | Subset / class | F difference, 95% CI (%p) |',
              '|---|---|---|---|']
    for scope, data in result['scopes'].items():
        if not data['available']:
            continue
        for delta, subsets in data['bootstrap']['differences'].items():
            for subset, classes in subsets.items():
                for cls in (*CLASSES, 'macro_F'):
                    d = classes[cls] if cls == 'macro_F' else classes[cls]['F']
                    lines.append(f'| {scope} | {delta.replace("D", label("D"))} | {subset} / {cls} | ' +
                                 cell([d['difference'], *d['ci95']]) + ' |')
        check = data['sufficiency']
        lines += ['', f'{scope}: {check["independent_groups"]} performance groups (target 20–30).', '',
                  '| Class | Conservative fast moments | Independent fast groups | First look ≥500 | Final ≥1000 | Target ≥2000 |',
                  '|---|---:|---:|---|---|---|']
        for cls, values in check['classes'].items():
            lines.append(f'| {cls} | {values["fast_moments"]} | {values["independent_fast_groups"]} | {values["first_look_500"]} | '
                         f'{values["final_minimum_1000"]} | {values["final_target_2000"]} |')
    lines += ['', 'P/R/F and CIs are re-pooled integer counts. Fixed seed 20261002; 10,000 paired group resamples.',
              'P/R differences, raw counts, per-group changes and leave-one-group-out sensitivity are in final.json.', '', *result['caveats']]
    atomic(output.with_suffix('.md'), ('\n'.join(lines) + '\n').encode())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--opening-id', required=True)
    p.add_argument('--model-tag', required=True)
    p.add_argument('--json', type=Path, required=True)
    p.add_argument('--jobs', type=int, choices=range(1, 5), default=4)
    a = p.parse_args()
    vault.resume_final(a.opening_id, a.model_tag)  # before ANY sealed notation
    plan = load_plan(a.plan)
    from score import songs
    rows, excluded = vault.filter_final(songs(None, final=True, loose=True), log=False)
    frozen = json.loads((a.plan.parent / 'final_inputs.json').read_text())
    for relative, digest in frozen['hashes'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f'sealed input changed during final: {relative}')
    for relative, digest in frozen['checkpoints'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f'checkpoint changed during final: {relative}')
    for relative in frozen.get('absent_audits', []):
        if (ROOT / relative).exists():
            raise ValueError(f'audit added during final: {relative}')
    rows = [r for r in rows if r['slug'] in frozen['slugs']]
    results = {arm: {} for arm in 'ABCD'}
    tasks = [(arm, row, f'{plan["name"]}_{arm}_final') for arm in 'ABCD' for row in rows]
    with ProcessPoolExecutor(a.jobs) as pool:
        for arm, slug, counts in pool.map(score_one, tasks):
            results[arm][slug] = counts
    for relative, digest in frozen['hashes'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f'sealed input changed during final scoring: {relative}')
    for relative in frozen.get('absent_audits', []):
        if (ROOT / relative).exists():
            raise ValueError(f'audit added during final scoring: {relative}')
    check_performance_table(plan)
    result = summarize(plan, rows, results, excluded=excluded)
    result.update(opening_id=a.opening_id, input_snapshot=frozen)
    write_report(result, a.json)


if __name__ == '__main__':
    main()
