"""Direct child scorer for frozen A absolute activations, strict and loose.

Only a matching durable parent opening authorizes sealed reads. ft_variants.full
is called unchanged, with its snare callback bound to the explicit final/vault
cache. No .eval-cache/ft/solos or development vision output is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import vault_absolute as absolute
from vault_absolute_io import guard, hashes, load, tree, verify, write
sys.path.insert(0, str(Path(__file__).resolve().parent / 'chops_recipe'))
import chops_vault_score as loose_scorer


def strict_counts(truth_folder, hits):
    """Exactly chops_scoring strict -> fill_score.one: load/fills/tally/rename."""
    import fill_score
    from chops_scoring import validate_counts
    files = [truth_folder / 'truth.json', truth_folder / 'alignment.json']
    exists = [p.is_file() for p in files]
    if not all(exists):  # as fill_score.one: a missing file is FileNotFoundError -> no strict truth
        return Counter(), False
    truth, intervals = absolute.harvest_score.load(truth_folder, 'strict')
    windows = fill_score.fills(truth, intervals, 0.14) if truth else []
    counts = fill_score.tally(truth, hits, windows) if windows else Counter()
    if windows:
        counts['songs'] += 1
    for key, value in list(counts.items()):
        if key.startswith('fill.'):
            counts['slow.' + key[5:]] = value
    validate_counts(counts)
    return counts, bool(truth)


def notation_groups(truth_folder):
    import fill_seq
    files = [truth_folder / 'strip.json', truth_folder / 'alignment.json']
    exists = [p.is_file() for p in files]
    # Manifest entries may have no published truth at all.
    if not (truth_folder / 'strip.json').exists():
        return []
    if not all(exists):  # as one_loose: FileNotFoundError -> no notation
        return []
    return fill_seq.notated(truth_folder, 0.75)


def unusable_mix(song):
    """A mix cache ft_variants.full cannot use (it would raise); recorded per song, never scored as negatives."""
    import numpy as np
    mix = np.load(song / 'activations.mix.npy', allow_pickle=False)
    if mix.ndim != 2 or mix.shape[1] != 5 or not len(mix) or not np.isfinite(mix).all():
        return f'unusable activations.mix.npy shape {list(mix.shape)}'
    return None


def full_hits(song, cache, tag):
    import numpy as np
    import ft_variants
    from groovemap.dynamics import snare_response
    tree(cache, absolute.ROOT)
    found = sorted(cache.glob(f'student-{tag}-fold*.npy'))
    expected = [cache / f'student-{tag}-fold{k}.npy' for k in (0, 1)]
    if found != expected:
        raise ValueError('missing/stale final fold activations')
    activations = [np.load(p, allow_pickle=False) for p in found]
    if (activations[0].shape != activations[1].shape or
            any(a.ndim != 2 or a.shape[1] != 5 or not np.isfinite(a).all() for a in activations)):
        raise ValueError('invalid ensemble activation shapes/values')
    act = np.mean(activations, axis=0)  # same ft_variants.student fold convention

    def snare(target, values, name):
        if target != song or name != f'student-{tag}':
            raise ValueError('unexpected snare response target')
        path = cache / f'snare-{name}.npy'
        if path.exists():
            raise ValueError('stale snare cache in once-only absolute evaluation')
        stem = song / 'drums.htdemucs.wav'
        response = snare_response(stem if stem.exists() else stem.with_suffix('.opus'), values)
        np.save(path, response)
        return np.load(path, allow_pickle=False)

    with patch.object(ft_variants, 'snare', snare):
        return ft_variants.full(song, act, f'student-{tag}')


def loose_counts(song, hits):
    """Invoke score_song unchanged; replace only detector with already built hits."""
    import ft_variants
    # score_song resolves vault.SONGS/slug. A single temporary parent binding
    # keeps its audit/interval/unique-fast semantics exactly identical.
    with patch.object(loose_scorer.vault, 'SONGS', song.parent), patch.object(ft_variants, 'tuned', lambda _: hits), \
            patch.object(ft_variants, 'TAG', ft_variants.TAG):
        return loose_scorer.score_song({'slug': song.name}, 'absolute-precomputed-hits')


def score_job(job):
    from chops_scoring import validate_counts
    per_song = []
    strict = {c: Counter() for c in ('manifest', 'ollie')}
    loose = {s: Counter() for s in ('complete', 'partial', 'all_notation')}
    coverage = {c: dict(considered=0, with_strict_truth=0, with_notation=0, with_strict_fills=0, unusable_inputs=0)
                for c in ('manifest', 'ollie')}
    for item in job['songs']:
        song = tree(absolute.ROOT / item['song'], absolute.ROOT)
        cache = tree(absolute.ROOT / item['cache'], absolute.ROOT)
        truth = song / 'truth'
        cohort = item['cohort']
        groups = notation_groups(truth)
        problem = unusable_mix(song)
        if problem:
            coverage[cohort]['considered'] += 1
            coverage[cohort]['unusable_inputs'] += 1
            per_song.append({'video': item['video'], 'slug': item['slug'], 'cohort': cohort, 'unusable_inputs': problem,
                             'with_strict_truth': (truth / 'truth.json').is_file(), 'notation_systems': len(groups)})
            continue
        # Build hits once per song for both metrics, even a no-truth song.
        hits = full_hits(song, cache, job['model_tag'])
        counts, has_strict = strict_counts(truth, hits)
        strict[cohort].update(counts)
        cov = coverage[cohort]
        cov['considered'] += 1
        cov['with_strict_truth'] += int(has_strict)
        cov['with_notation'] += int(bool(groups))
        cov['with_strict_fills'] += int(bool(counts.get('songs')))
        pairs = {'strict': ('truth.json', 'alignment.json'), 'notation': ('strip.json', 'alignment.json')}
        incomplete = [k for k, names in pairs.items() if len({(truth / n).is_file() for n in names}) > 1]
        row = {'video': item['video'], 'slug': item['slug'], 'cohort': cohort, 'incomplete_truth': incomplete,
               'strict_counts': dict(counts), 'with_strict_truth': has_strict, 'notation_systems': len(groups)}
        if cohort == 'ollie':
            provenance = load(song / 'provenance.json')
            row['canonical_reader'] = {key: provenance[key] for key in ('status', 'reason') if key in provenance}
        if cohort == 'manifest' and groups:
            result = loose_counts(song, hits)
            complete, partial = Counter(result['complete']), Counter(result['partial'])
            loose['complete'].update(complete)
            loose['partial'].update(partial)
            loose['all_notation'].update(complete); loose['all_notation'].update(partial)
            row['loose'] = result
        per_song.append(row)
    strict['combined'] = strict['manifest'].copy()
    strict['combined'].update(strict['ollie'])
    coverage['combined'] = {k: coverage['manifest'][k] + coverage['ollie'][k] for k in coverage['manifest']}
    for counts in (*strict.values(), *loose.values()):
        validate_counts(counts)
    return {'coverage': coverage, 'strict': {c: absolute.metrics(v) for c, v in strict.items()},
            'strict_counts': {c: dict(v) for c, v in strict.items()},
            'loose_manifest': {s: absolute.metrics(v) for s, v in loose.items()},
            'loose_counts': {s: dict(v) for s, v in loose.items()}, 'per_song': per_song,
            'reader_rejections': [{'video': r['video'], **r['canonical_reader']} for r in per_song
                                  if r.get('canonical_reader', {}).get('status') == 'reader_rejected']}


def execute(request, opening_id, model_tag):
    job = absolute.child_job(request, opening_id, model_tag)  # BEFORE any sealed read
    directory = absolute.ROOT / job['directory']
    snapshot = load(directory / 'inputs.json')
    absolute.verify_inputs(job, snapshot)
    # Freeze prediction bytes too, so a concurrently edited cache fails closed.
    predictions = hashes([absolute.ROOT / s['cache'] / f'student-{model_tag}-fold{k}.npy'
                          for s in job['songs'] for k in (0, 1)], absolute.ROOT)
    with guard(writes=[directory, absolute.NUMBA_CACHE, *[absolute.ROOT / s['cache'] for s in job['songs']]]):
        result = score_job(job)
        verify(predictions, absolute.ROOT)
        absolute.verify_inputs(job, snapshot)
        result['prediction_hashes'] = predictions
        write(directory / 'scored.json', result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--opening-id', required=True)
    parser.add_argument('--model-tag', required=True)
    args = parser.parse_args(argv)
    execute(args.request, args.opening_id, args.model_tag)


if __name__ == '__main__':
    main()
