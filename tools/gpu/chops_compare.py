"""WSL four-arm comparison: same GPU, grouped folds, budget and two seeds.

run --name NAME [--dry-run]; score --name NAME; final --name NAME.
Default: A/B/C/D x seeds 0/1 x 2 folds = 16 sequential training jobs.
RETRAIN: --arms BASE ATH --base-plan PATH --holdout PATH --batch2-vault PATH
[--dedup-groups PATH]; run --dry-run reads existing non-vault truth and prints cohorts.
No downloads. Final is a separate once-only, logged vault opening.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

from chops_common import (CACHE, ROOT, arm_includes, arm_label, arm_fold, check_plan, check_performance_table, fold_policy,
                          load_plan, load_performance_table, partitions, performance_groups, require_sealed_groups,
                          promotion_policy, sha, splits, train_roster, vault, retrain_inputs, retrain_rosters, retrain_partitions)
from harvest_transfer import atomic, lock

PYTHON = str(ROOT / '.venv/bin/python')
sys.path.insert(0, str(ROOT / 'tools/truth_video'))
METRICS = ('kick', 'snare', 'tom')
RECIPES = {'BASE': 'v2', 'ATH': 'v2', 'A': 'v2', 'B': 'v2', 'O': 'v2', 'C': 'hierarchical', 'D': 'three_state', 'E': 'weakorder', 'v5b': 'v5b'}
DEFAULT_ARMS = ('A', 'B', 'C', 'D')


def experiments(name, arms=DEFAULT_ARMS):
    return [{'tag': f'{name}_{arm}_s{seed}', 'arm': arm, 'recipe': RECIPES[arm], 'seed': seed}
            for arm in arms for seed in (0, 1)]


def training_jobs(plan, plan_path):
    return [(e, fold, [PYTHON, 'tools/gpu/chops_recipe/train_recipe.py', '--recipe', e['recipe'],
                      '--plan', str(plan_path), '--fold', str(fold), '--tag', e['tag'],
                      '--seed', str(e['seed'])])
            for e in plan['tags'] for fold in range(len(plan['folds']))]


def require_wsl():
    path = Path('/proc/sys/kernel/osrelease')
    if not path.exists() or 'microsoft' not in path.read_text().lower():
        raise RuntimeError('run/final require WSL; --dry-run is available offline on any host')
    vault.require_final_writer()


def gpu_identity():
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('comparison needs exactly one visible CUDA GPU')
    props = torch.cuda.get_device_properties(0)
    return {'visible_device': os.environ['CUDA_VISIBLE_DEVICES'], 'name': props.name,
            'uuid': str(getattr(props, 'uuid', 'unavailable')),
            'total_memory': props.total_memory, 'capability': list(torch.cuda.get_device_capability(0))}


def command(argv: list[str], log: Path, tag: str | None = None) -> None:
    env = {k: v for k, v in os.environ.items() if not (k.startswith('FT_') and k.endswith('_SCALE'))}
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1', PYTHONPATH=str(ROOT / 'tools/finetune'),
               PYTHONDONTWRITEBYTECODE='1')
    if tag:
        env['FT_TAG'] = tag
    log.parent.mkdir(parents=True, exist_ok=True)
    print(shlex.join(argv), '->', log.relative_to(ROOT), flush=True)
    with log.open('w') as stream:
        stream.write(shlex.join(argv) + '\n')
        stream.flush()
        subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def feature(song: Path) -> Path:
    """Scoring-only folders have features, never fabricated training truth."""
    import numpy as np
    from adtof_pytorch.audio import process_audio_file
    folder = CACHE / song.name if song.parent == vault.SONGS else CACHE / 'solos' / song.name
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / 'feat.npy'
    if not output.exists():
        stem = song / 'drums.htdemucs.wav'
        if not stem.exists():
            stem = stem.with_suffix('.opus')
        if not stem.exists():
            raise FileNotFoundError(f'{stem}; send baseline/solo audio with harvest_wsl.sh push --inputs')
        feat, _ = process_audio_file(str(stem))
        temp = output.with_suffix('.npy.tmp')
        with temp.open('wb') as out:
            np.save(out, feat[:, :, 0].astype(np.float16))
        temp.replace(output)
    return folder


def score_rows(channels=None, final=False, exclude=()) -> list[dict]:
    import score
    import fill_seq
    result = []
    rows = score.songs(channels, final=final, loose=True)
    seal = vault.Vault()
    for row in rows:
        if row['slug'] in exclude or seal.contains_song(row) != final:
            continue
        song = vault.SONGS / row['slug']
        try:
            strict, _ = score.load(song / 'truth', 'strict')
        except FileNotFoundError:
            strict = []
        try:
            loose = fill_seq.notated(song / 'truth', 0.75)
        except FileNotFoundError:
            loose = []
        if strict or loose:
            result.append({'slug': row['slug'], 'channel': row['channel'], 'video': row['video'],
                           'strict': bool(strict), 'loose': bool(loose)})
    return result


def solo_inputs() -> list[dict]:
    import fill_seq
    seal = vault.Vault()
    result = []
    for video in fill_seq.SOLOS:
        pointer = fill_seq.TRUTH / video / 'song.txt'
        song = Path(pointer.read_text().strip())
        if not song.is_absolute() or not song.is_dir() or not song.is_relative_to(ROOT):
            raise ValueError(f'unresolved WSL solo pointer {pointer}; push --inputs then truth_paths.py --rebase')
        if seal.contains_song(song):
            continue
        if not (song / 'activations.drums-htdemucs.npy').exists():
            raise FileNotFoundError(f'missing solo caches: {song}')
        result.append({'video': video, 'song': str(song.relative_to(ROOT)),
                       'slug': song.name, 'cache': str(feature(song).relative_to(ROOT))})
    if not result:
        raise ValueError('no non-vault external solos available')
    if len({s['slug'] for s in result}) != len(result):
        raise ValueError('solo cache slug collision')
    return result


def cohort_summary(rows, groups):
    """Strict exported frame counts, not loose notation or independent hits."""
    import numpy as np
    from chops_recipe.stratified import centres
    counts = {cls: 0 for cls in METRICS}
    performances = {cls: set() for cls in METRICS}
    seconds = {'original': 0., 'expanded': 0.}
    songs = {'original': 0, 'expanded': 0}
    for row in rows:
        bucket = 'original' if row['channel'] in splits.ORIGINAL_CHANNELS else 'expanded'
        songs[bucket] += 1
        intervals = []
        for a, b in sorted(row['intervals']):
            if intervals and a <= intervals[-1][1]:
                intervals[-1][1] = max(intervals[-1][1], b)
            else:
                intervals.append([a, b])
        seconds[bucket] += sum(b - a for a, b in intervals)
        n = len(np.load(CACHE / row['slug'] / 'feat.npy', mmap_mode='r', allow_pickle=False))
        points = centres(row, n)['fast']
        for cls in METRICS:
            count = len(points.get(cls, ()))
            counts[cls] += count
            if count:
                performances[cls].add(groups[row['slug']])
    return {'songs': songs, 'accepted_seconds': seconds, 'fast_positive_frames': counts,
            'independent_fast_performances': {cls: len(values) for cls, values in performances.items()}}


def create_plan(name: str, directory: Path, *, arms=DEFAULT_ARMS, folds=2, steps=1500, gpu=None,
                weak_lambda=.05) -> dict:
    seal = vault.Vault()
    table, table_sha = load_performance_table(ROOT)
    require_sealed_groups(seal, table)
    weak_rows, weak_channels, external_exclusions = [], [], {}
    if 'E' in arms:
        from weak_set import manifest_rows, external_solos, external_exclusions as external_peer_exclusions
        weak_rows = [r for r in manifest_rows(seal) if not seal.contains_song(r)]
        weak_channels = sorted({r['channel'] for r in weak_rows})
        external, _ = external_solos(ROOT)
        external_exclusions = external_peer_exclusions(weak_rows, external, table)
        for slug, info in external_exclusions.items():
            print(f'Excluded external-solo harvest peer: {slug} -> ' + ', '.join(info['solo_sources']))
    folds = fold_policy(set(seal.train_channels) | set(weak_channels), folds)
    channels = [c for group in folds for c in group]
    data_command = ([PYTHON, 'tools/finetune/weak_set.py', '--prepare-strict', '--exclude', *external_exclusions]
                    if 'E' in arms else [PYTHON, 'tools/finetune/data.py'])
    command([*data_command, '--jobs', '4', '--only', *channels], directory / 'data.log')
    command([PYTHON, 'tools/gpu/truth_paths.py', '--rebase'], directory / 'rebase.log')
    selected_references = (score_rows(channels, exclude=external_exclusions) if 'E' in arms else score_rows(channels))
    references = [r for r in selected_references if not seal.contains_song(r) and r['slug'] not in external_exclusions]
    songs = []
    for row in references:
        folder = feature(vault.SONGS / row['slug'])
        if not row['strict']:
            continue
        meta_path = folder / 'truth.json'
        if not meta_path.exists():
            raise FileNotFoundError(f'{meta_path}; strict data.py preparation did not produce training truth')
        meta = json.loads(meta_path.read_text())
        if (meta['slug'] != row['slug'] or meta['channel'] != row['channel'] or
                meta.get('video_id') != row['video']):
            raise ValueError(f'FT metadata differs from canonical manifest row: {meta_path}')
        if meta['channel'] in channels and not seal.contains_song(meta) and meta['intervals']:
            songs.append(meta)
    if any(arm not in ('A', 'E') for arm in arms) and not any(s['channel'] not in splits.ORIGINAL_CHANNELS for s in songs):
        raise ValueError('no strict truth from expanded TRAIN roster yet; collect/read new chops before retraining')
    solo = solo_inputs()
    groups = performance_groups(songs + references + solo, table)
    if any(s['slug'] == row['slug'] or splits.same_song(s['slug'], row['slug']) or
           groups[s['slug']] == groups[row['slug']] for s in solo for row in songs + references):
        raise ValueError('external solo overlaps cohort / SAME_SONG / SAME_PERFORMANCE')
    assignments = partitions(songs, references + weak_rows, folds, table)
    reference_slugs = {r['slug'] for r in references}
    for item in assignments:
        item['predict'] = [s for s in item['predict'] if s in reference_slugs]
    if any(not item['train'] or not item['held'] for item in assignments):
        raise ValueError('each fold must have nonempty training and held-out strict cohorts')
    hashes = {}
    sources = ['tools/finetune/train.py', 'tools/finetune/data.py', 'tools/finetune/splits.py', 'tools/harvest/vault.py',
               'tools/harvest/fill_score.py', 'tools/harvest/global_score.py', 'tools/harvest/score.py',
               'tools/finetune/ft_variants.py', 'tools/truth_video/fill_seq.py',
               'tools/truth_video/align.py', 'tools/truth_video/evaluate.py', 'tools/gpu/harvest_transfer.py',
               'tools/gpu/chops_common.py', 'tools/gpu/chops_compare.py',
               'tools/gpu/chops_statistics.py', 'tools/gpu/chops_vault_score.py', 'tools/gpu/chops_scoring.py',
               'tools/gpu/chops_recipe/train_recipe.py', 'tools/gpu/chops_recipe/train_v2frozen.py',
               'tools/gpu/chops_recipe/stratified.py', 'tools/gpu/chops_recipe/three_state.py',
               'tools/gpu/chops_recipe/train_synth.py', 'truth/harvest/manifest.jsonl', 'truth/harvest/vault.json']
    for row in songs:
        sources.append(str((CACHE / row['slug'] / 'truth.json').relative_to(ROOT)))
    for row in references:
        truth = vault.SONGS / row['slug'] / 'truth'
        sources += [str(p.relative_to(ROOT)) for p in truth.glob('*.json')]
    for row in solo:
        truth = ROOT / 'truth' / row['video']
        sources += [str(p.relative_to(ROOT)) for p in truth.glob('*.json')]
        sources.append(str((truth / 'song.txt').relative_to(ROOT)))
    for relative in dict.fromkeys(sources):
        hashes[relative] = sha(ROOT / relative)
    if table_sha is not None:
        hashes['truth/harvest/same_performance.json'] = table_sha
    for path in (ROOT / 'src/stickmap').glob('*.py'):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    audits, absent_audits, audited_negative_songs = {}, [], []
    training_meta = {r['slug']: r for r in songs}
    for row in songs + references:
        path = vault.SONGS / row['slug'] / 'truth/kst_audit.json'
        relative = str(path.relative_to(ROOT))
        if path.exists():
            sys.path.insert(0, str(ROOT / 'tools/gpu/chops_recipe'))
            from three_state import masks, validate_audit
            audit = json.loads(path.read_text())
            validate_audit(audit)
            audits[row['slug']] = relative
            hashes[relative] = sha(path)
            if row['slug'] in training_meta and row['slug'] not in audited_negative_songs:
                import numpy as np
                meta = training_meta[row['slug']]
                n = len(np.load(CACHE / row['slug'] / 'feat.npy', mmap_mode='r', allow_pickle=False))
                target, known, _ = masks(meta, n, audit)
                if ((target[:, :3] == 0) & (known[:, :3] > 0)).any():
                    audited_negative_songs.append(row['slug'])
        else:
            absent_audits.append(relative)
    cohorts = {arm: {'songs': [s['slug'] for s in songs if arm_includes(arm, s['channel'])]}
               for arm in arms}
    tags = experiments(name, arms)
    feature_stamps = {}
    for folder in [CACHE / r['slug'] for r in references] + [ROOT / r['cache'] for r in solo]:
        for filename in ('feat.npy', 'teacher0.npy'):
            path = folder / filename
            if path.exists():
                stat = path.stat()
                feature_stamps[str(path.relative_to(ROOT))] = [stat.st_size, stat.st_mtime_ns]
    plan = {'name': name, 'channels': channels, 'songs': songs, 'references': references,
            'solos': solo, 'folds': assignments, 'tags': tags, 'hashes': hashes,
            'feature_stamps': feature_stamps,
            'steps': steps, 'gpu': gpu, 'arms': cohorts, 'audits': audits,
            'audited_negative_songs': sorted(audited_negative_songs),
            'absent_audits': sorted(set(absent_audits)),
            'new_channel_policy': f'sha256(utf8 channel name) integer modulo {len(folds)}; connected channels move to lowest fold',
            'same_performance': [list(g) for g in table],
            'same_performance_sha256': table_sha,
            'performance_groups': groups,
            'same_song': [key if isinstance(key, str) else list(key) for key in splits.SAME_SONG],
            'final_policy': {'ensemble': 'mean of both seeds and all folds per arm',
                             'resamples': 10000, 'bootstrap_seed': 20261002,
                             'differences': ['B-A', 'C-B', 'D-C', 'D-A'],
                             'primary': 'fast KST macro F1; complete audited KST intervals only'}}
    if 'E' in arms:
        from weak_set import POLICY, build_fold, print_counts
        plan['weak_channels'] = weak_channels
        plan['external_solo_exclusions'] = external_exclusions
        plan['baseline_policy'] = {'reuse': False, 'source': 'chopround2',
                                   'reason': 'fresh A rerun; historical cohort/folds/seeds/steps and inputs are not proven identical'}
        plan['weak_order'] = {'policy_version': POLICY, 'lambda': weak_lambda, 'batch_size': 1,
                              'acceptance_mode': 'automatic',
                              'mix_ratio': '1 weak sample per unchanged 12-sample strict batch, every step',
                              'ctc_backend': 'cpu', 'sidecars': [], 'counts': [],
                              'primary': 'external solos fast K/S/T macro F1',
                              'stop_rule': 'both seeds: solo fast macro gain >=2pp; no strict fast/slow/all or global family F1 drop >0.5pp'}
        plan['absent_weak_reviews'] = []
        for k in range(len(assignments)):
            data = build_fold(plan, k, root=ROOT, prepare_feature=feature)
            print_counts(data)
            path = directory / f'weak-order-fold{k}.json'
            atomic(path, (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode())
            relative = str(path.relative_to(ROOT))
            hashes[relative] = sha(path)
            hashes.update(data['source_hashes'])
            plan['weak_order']['sidecars'].append(relative)
            plan['weak_order']['counts'].append(data['counts'])
            for slug, windows in data['songs'].items():
                if windows:
                    path = CACHE / slug / 'feat.npy'
                    stat = path.stat()
                    feature_stamps[str(path.relative_to(ROOT))] = [stat.st_size, stat.st_mtime_ns]
        plan['absent_weak_reviews'] = sorted(set(plan['absent_weak_reviews']))
        for filename in ('tools/finetune/weak_set.py', 'tools/finetune/weak_order.py',
                         'tools/finetune/WEAK_ORDER.md', 'tools/gpu/chops_recipe/weakorder.py'):
            if (ROOT / filename).exists():  # WEAK_ORDER.md is a local note, not in the public repository
                hashes[filename] = sha(ROOT / filename)
    plan['promotion'] = promotion_policy(plan)
    print_train_policy(plan)
    for e in tags:
        if any(not arm_fold(plan, e, k)['train'] for k in range(len(assignments))):
            raise ValueError(f'empty training cohort in arm {e["arm"]}; grouping may have collapsed a fold')
    summaries = {}
    for arm in cohorts:
        summaries[arm] = []
        for k in range(len(assignments)):
            item = arm_fold(plan, {'arm': arm}, k)
            entry = {scope: cohort_summary([r for r in songs if r['slug'] in item[scope]], groups)
                     for scope in ('train', 'held')}
            summaries[arm].append(entry)
            print(f'{arm} fold{k} strict cohort: ' + json.dumps(entry, sort_keys=True), flush=True)
    plan['cohort_diagnostics'] = summaries
    atomic(directory / 'plan.json', (json.dumps(plan, ensure_ascii=False, indent=2) + '\n').encode())
    print(f'Plan: {len(songs)} strict training songs; {len(references)} scoring songs; {len(solo)} solos', flush=True)
    for k, item in enumerate(assignments):
        print(f'fold{k}: channels={item["channels"]}, train={len(item["train"])}, held={len(item["held"])}', flush=True)
    return plan


def print_train_policy(plan):
    print('TRAIN channel roster (song-level vault exclusions always apply): ' + ', '.join(plan['channels']))
    policy = promotion_policy(plan)
    print('Promotion-eligible arms: ' + ', '.join(policy['eligible_arms']))
    if 'E' in {e.get('arm') for e in plan['tags']}:
        print('E primary: external solo fast K/S/T macro F1. Promising only if gain >= +2pp over A '
              'in BOTH seeds and strict fast/slow/all plus global family F1 do not drop >0.5pp.')
        print('A is rerun alongside E; chopround2 artifacts are not reused without proven identity.')
        print('E preflight requires nonempty automatically accepted weak windows in every training fold: '
              'reader-clean >=8 core moments, plausible cursor span, acoustic onset count and >=85% monotone assignment.')
    if 'D' in {e.get('arm') for e in plan['tags']} and not policy['D']['promotion_eligible']:
        print(f'WARNING: {policy["D"]["label"]}; excluded from promotion. {policy["D"]["reason"]}.')


def checkpoint(plan, tag, fold) -> Path:
    output = CACHE / 'models' / f'{tag}-fold{fold}.pt'
    data = json.loads(output.with_suffix('.json').read_text())
    recipe = next(item for item in plan['tags'] if item['tag'] == tag)
    expected = arm_fold(plan, recipe, fold)
    if set(data['held']) != set(expected['held']) or set(data['train']) != set(expected['train']):
        raise ValueError(f'checkpoint cohort mismatch: {output}')
    if (data['args']['steps'] != plan.get('steps', 1500) or data['args']['device'] != 'cuda' or
            data['args']['seed'] != recipe['seed'] or data['args']['doubt'] != 0.3 or
            data.get('recipe') != recipe['recipe']):
        raise ValueError(f'checkpoint recipe/device mismatch: {output}')
    if 'arms' in plan and data.get('arm') != recipe['arm']:
        raise ValueError(f'checkpoint arm mismatch: {output}')
    if recipe.get('arm') == 'E':
        weak = data.get('weak_order', {})
        relative = plan['weak_order']['sidecars'][fold]
        if (weak.get('lambda') != plan['weak_order']['lambda'] or weak.get('batch_size') != 1
                or weak.get('steps') != plan['steps'] or weak.get('sidecar_sha256') != plan['hashes'][relative]):
            raise ValueError(f'checkpoint weak-order recipe/sidecar mismatch: {output}')
    if 'arms' in plan and any(data['args'].get(k) != v for k, v in
                              {'batch': 12, 'seconds': 8.0, 'lr': 1e-4, 'distill': 0.3}.items()):
        raise ValueError(f'checkpoint training budget/options mismatch: {output}')
    if not output.exists():
        raise FileNotFoundError(output)
    return output


def predict(plan: dict, tag: str) -> None:
    import numpy as np
    import torch
    from train import original_model, run
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(1)
    for fold, item in enumerate(plan['folds']):
        model = original_model('cuda')
        model.load_state_dict(torch.load(checkpoint(plan, tag, fold), weights_only=True, map_location='cuda'))
        model.eval()
        folders = [CACHE / slug for slug in item['predict']] + [ROOT / row['cache'] for row in plan['solos']]
        for folder in folders:
            act = run(model, np.load(folder / 'feat.npy', allow_pickle=False), 'cuda').astype(np.float32)
            output = folder / f'student-{tag}-fold{fold}.npy'
            temp = output.with_suffix('.npy.tmp')
            with temp.open('wb') as out:
                np.save(out, act)
            temp.replace(output)
        del model
        torch.cuda.empty_cache()
        print(f'{tag} fold{fold}: CUDA predictions {len(folders)}', flush=True)


def check_predictions(plan: dict, tag: str) -> None:
    for fold in range(len(plan['folds'])):
        checkpoint(plan, tag, fold)
    for fold, item in enumerate(plan['folds']):
        for slug in item['predict']:
            found = list((CACHE / slug).glob(f'student-{tag}-fold?.npy'))
            expected = ({f'student-{tag}-fold{k}.npy' for k, f in enumerate(plan['folds'])
                         if slug in f['predict']} if 'retrain' in plan else {f'student-{tag}-fold{fold}.npy'})
            if {p.name for p in found} != expected:
                raise ValueError(f'expected exactly the channel-held fold prediction: {tag} {slug}')
    for solo in plan['solos']:
        found = list((ROOT / solo['cache']).glob(f'student-{tag}-fold?.npy'))
        if {p.name for p in found} != {f'student-{tag}-fold{k}.npy' for k in range(len(plan['folds']))}:
            raise ValueError(f'missing/stale solo folds: {tag} {solo["slug"]}')


def score_tag(plan: dict, tag: str, directory: Path, jobs: int) -> None:
    if 'retrain' in plan:
        return score_retrain_tag(plan, tag, directory)
    check_plan(plan)
    check_predictions(plan, tag)
    out = directory / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / 'done.json').unlink(missing_ok=True)
    for path in CACHE.glob(f'*/snare-student-{tag}.npy'):
        path.unlink()
    for path in (CACHE / 'solos').glob(f'*/snare-student-{tag}.npy'):
        path.unlink()
    # Current fill_score/fill_seq support 1..4 workers. Global is split by channel.
    worker_count = str(min(jobs, 4))
    excluded = list(plan.get('external_solo_exclusions', {}))
    exclusion_args = ['--exclude', *excluded] if excluded else []
    base = [PYTHON, 'tools/gpu/chops_scoring.py', '--detect', 'ft_variants:tuned', '--jobs', worker_count,
            *exclusion_args]
    for scope in ('loose', 'strict'):
        command([*base, '--scope', scope, '--json', str(out / f'{scope}.json'), '--only', *plan['channels']],
                out / f'{scope}.log', tag)
    command([*base, '--scope', 'solos', '--json', str(out / 'solos.json'),
             '--videos', *[s['video'] for s in plan['solos']]], out / 'solos.log', tag)

    def global_channel(channel):
        path = out / 'channels' / (channel + '.json')
        command([PYTHON, 'tools/harvest/global_score.py', '--detect', 'ft_variants:tuned', '--tier', 'strict',
                 '--json', str(path), *exclusion_args, '--only', channel], path.with_suffix('.log'), tag)
        return channel, json.loads(path.read_text())

    totals = {inst: [0, 0, 0] for inst in ('kick', 'snare', 'tom', 'hihat', 'cymbal')}
    totals['songs'] = 0
    by_channel = {}
    with ThreadPoolExecutor(jobs) as pool:
        for channel, counts in pool.map(global_channel, plan['channels']):
            by_channel[channel] = counts
            totals['songs'] += counts.get('songs', 0)
            for inst in ('kick', 'snare', 'tom', 'hihat', 'cymbal'):
                totals[inst] = [a + b for a, b in zip(totals[inst], counts.get(inst, [0, 0, 0]))]
    totals['channels'] = by_channel
    atomic(out / 'global.json', json.dumps(totals, indent=2).encode())
    check_plan(plan)
    completed = {'tag': tag, 'plan_sha256': sha(directory / 'plan.json')}
    if 'arms' in plan:
        completed['checkpoints'] = {str(checkpoint(plan, tag, k).relative_to(ROOT)): sha(checkpoint(plan, tag, k))
                                    for k in range(len(plan['folds']))}
    atomic(out / 'done.json', json.dumps(completed).encode())


def prf(tp, est, ref):
    return [tp / est if est else None, tp / ref if ref else None,
            2 * tp / (est + ref) if est + ref else None]


def weak_preflight(plan):
    if 'E' not in plan.get('arms', {}):
        return
    from weak_set import load_fold
    for fold, relative in enumerate(plan['weak_order']['sidecars']):
        data = load_fold(ROOT / relative, plan, fold, root=ROOT)
        if not any(data['songs'].values()):
            raise ValueError(f'E fold{fold} has zero accepted weak windows: {data["skipped"]}. '
                             'Inspect weak_set.py --diagnose-dense-solos on WSL and its automatic acceptance evidence, '
                             'then use a fresh comparison name after valid windows are available. No GPU training started.')


def weak_stop_rule(rows):
    """Both seeds are mandatory; missing denominators are inconclusive, not passes."""
    lookup = {(r.get('arm'), r['seed'], r['metric'], r['subset']): r for r in rows}
    results = []
    for seed in (0, 1):
        failures, gains, guard_gains = [], [], []
        for cls in METRICS:
            a, e = [lookup.get((arm, seed, 'solos', 'fast')) for arm in ('A', 'E')]
            if a is None or e is None or a['prf'][cls][2] is None or e['prf'][cls][2] is None:
                failures.append(f'undefined solo fast {cls}')
            elif a['counts'][cls][2] != e['counts'][cls][2]:
                raise ValueError('E/A solo reference cohorts differ')
            else:
                gains.append(100 * (e['prf'][cls][2] - a['prf'][cls][2]))
        macro_gain = sum(gains) / 3 if len(gains) == 3 else None
        if macro_gain is not None and macro_gain < 2 - 1e-9:
            failures.append('solo fast macro gain <2pp')
        for metric, subsets in (('strict', ('fast', 'slow', 'all')), ('global', ('all',))):
            for subset in subsets:
                a, e = [lookup.get((arm, seed, metric, subset)) for arm in ('A', 'E')]
                classes = ('kick', 'snare', 'tom', 'hihat', 'cymbal') if metric == 'global' else METRICS
                for cls in classes:
                    if a is None or e is None or a['prf'].get(cls, [None] * 3)[2] is None or e['prf'].get(cls, [None] * 3)[2] is None:
                        failures.append(f'undefined guard {metric}/{subset}/{cls}')
                        continue
                    if a['counts'][cls][2] != e['counts'][cls][2]:
                        raise ValueError('E/A strict/global reference cohorts differ')
                    gain = 100 * (e['prf'][cls][2] - a['prf'][cls][2])
                    guard_gains.append(gain)
                    if gain < -.5 - 1e-9:
                        failures.append(f'{metric}/{subset}/{cls} drop >0.5pp')
                if metric == 'global' and a is not None and e is not None:
                    af, ef = a.get('all_song_prf', [None] * 3)[2], e.get('all_song_prf', [None] * 3)[2]
                    if af is None or ef is None:
                        failures.append('undefined all-song global guard')
                    elif 100 * (ef - af) < -.5 - 1e-9:
                        failures.append('all-song global drop >0.5pp')
        results.append({'seed': seed, 'solo_fast_macro_gain_pp': macro_gain,
                        'worst_guard_gain_pp': min(guard_gains) if guard_gains else None,
                        'passes': not failures, 'failures': failures})
    return {'primary': 'solo fast K/S/T macro F1', 'promising': all(r['passes'] for r in results),
            'rule': '>=+2pp over A in both seeds; strict fast/slow/all and global no drop >0.5pp',
            'seeds': results}


def report(plan: dict, directory: Path) -> None:
    if 'retrain' in plan:
        return report_retrain(plan, directory)
    from chops_scoring import validate_counts
    policy = promotion_policy(plan)
    rows = []
    for experiment in plan['tags']:
        tag = experiment['tag']
        done = directory / tag / 'done.json'
        if not done.exists() or json.loads(done.read_text())['plan_sha256'] != sha(directory / 'plan.json'):
            raise ValueError(f'incomplete scoring: {tag}')
        for metric in ('loose', 'strict', 'solos', 'global'):
            counts = json.loads((directory / tag / f'{metric}.json').read_text())
            if metric != 'global':
                validate_counts(counts)
            for subset in ('all',) if metric == 'global' else ('all', 'fast', 'slow'):
                row = {**experiment, 'metric': metric, 'subset': subset, 'counts': {}, 'prf': {},
                       'promotion_eligible': experiment.get('arm') in policy['eligible_arms']}
                for inst in ('kick', 'snare', 'tom', 'hihat', 'cymbal') if metric == 'global' else METRICS:
                    values = counts.get(inst, [0, 0, 0]) if metric == 'global' else [
                        counts.get(f'{subset}.{inst}.{key}', 0) for key in ('tp', 'est', 'ref')]
                    row['counts'][inst] = values
                    row['prf'][inst] = prf(*values)
                if metric == 'global':
                    pooled = [sum(counts.get(inst, [0, 0, 0])[k]
                                  for inst in ('kick', 'snare', 'tom', 'hihat', 'cymbal')) for k in range(3)]
                    row['all_song_counts'] = pooled
                    row['all_song_pr'] = prf(*pooled)[:2]
                    row['all_song_prf'] = prf(*pooled)
                rows.append(row)
    labels = ' / '.join(dict.fromkeys(e.get('arm', e['recipe']) for e in plan['tags']))
    lines = [f'# {plan["name"]}: {labels}, CUDA, seeds 0 and 1', '',
             f'{len(plan["songs"])} strict training songs; {len(plan["references"])} scoring songs; '
             f'{len(plan["solos"])} external solos. Channels: ' + ', '.join(plan['channels']) + '.', '',
             'P/R/F are pooled from integer counts, not means of channel scores. Undefined denominators show —. '
             'All-song P/R pools all five families. All metrics exclude the sealed vault.', '',
             'Promotion-eligible arms: ' + ', '.join(policy['eligible_arms']) + '.',
             f'{policy["D"]["label"]}: {policy["D"]["reason"]}.', '',
             '| Arm (recipe) | Seed | Metric / subset | Kick P/R/F | Snare P/R/F | Tom P/R/F | All-song P/R/F |',
             '|---|---:|---|---|---|---|---|']

    def cell(values):
        return ' / '.join('—' if v is None else f'{100 * v:.2f}' for v in values)

    for row in rows:
        lines.append(f'| {arm_label(row["arm"], plan) if "arm" in row else row["recipe"]} ({row["recipe"]}) | {row["seed"]} | {row["metric"]} / {row["subset"]} | ' +
                     ' | '.join(cell(row['prf'][inst]) for inst in METRICS) + ' | ' +
                     (cell(row['all_song_prf']) if 'all_song_prf' in row else '—') + ' |')
    weak_result = None
    if 'E' in plan.get('arms', {}):
        weak_result = weak_stop_rule(rows)
        lines += ['', 'E primary: external solo fast K/S/T macro F1. Stop rule: ' + weak_result['rule'] + '.',
                  'E verdict: **' + ('promising' if weak_result['promising'] else 'not promising / inconclusive') + '**.',
                  '', '| Seed | Solo fast macro gain (pp) | Worst strict/global family gain (pp) | Pass |',
                  '|---|---:|---:|---|']
        for item in weak_result['seeds']:
            values = ['—' if item[k] is None else f'{item[k]:+.2f}'
                      for k in ('solo_fast_macro_gain_pp', 'worst_guard_gain_pp')]
            lines.append(f'| {item["seed"]} | {values[0]} | {values[1]} | {item["passes"]} |')
            if item['failures']:
                lines.append(f'Seed {item["seed"]}: ' + '; '.join(item['failures']) + '.')
        lines += ['', 'Global HH/C P/R/F guards:']
        for row in rows:
            if row['metric'] == 'global':
                lines.append(f'{row["arm"]} seed{row["seed"]}: HH {cell(row["prf"]["hihat"])}; C {cell(row["prf"]["cymbal"])}.')
        lines += ['', 'E uses the identical A strict cohort, sampling, initialization, steps and RNG streams, '
                  'plus independent weak crops and source-window KST KD masking. E–A tests this auxiliary '
                  'branch; a matched KD-mask lambda=0 control is needed to isolate CTC from KD policy. '
                  'This promising rule is a screening result; WEAK_ORDER.md confirmation gates still apply. '
                  'No vault evaluation is permitted for an E plan.']
    lines += ['', 'Values are percentages. Loose/solo fast = nearest written K/S/T neighbour <90 ms '
              '(order DP ±60 ms); strict fast = median gap <90 ms in a ≥4-moment, tom-containing fill '
              '(gap ≤140 ms). These are different metrics; weak supervision uses only independently '
              'reviewed source notation, never the loose scorer\'s fitted event times.', '',
              'Matched estimates inherit reference speed. Unmatched estimates use nearest-reference time regions '
              '(earlier reference on ties; no reference means slow). Legacy DP remains label-dependent. '
              'D–C changes several loss policies together; it is not a pure three-state-label effect.', '',
              'Compare the two seeds together; choosing a recipe/seed on these scores has selection optimism. ' +
              ('E plans never open the vault. ' if 'E' in plan.get('arms', {}) else
               'Final freezes all A–D arms and averages both seeds and all folds within each arm in one opening. ') +
              'SAME_PERFORMANCE joins registered video IDs transitively for splits and bootstrap; '
              'unregistered reposts remain a leakage risk. No deployment weights are changed.']
    atomic(directory / 'comparison.json', json.dumps({'plan': plan['name'], 'promotion': policy, 'rows': rows,
                                                      'weak_order_result': weak_result}, indent=2).encode())
    atomic(directory / 'comparison.md', ('\n'.join(lines) + '\n').encode())
    print('\n'.join(lines), flush=True)


def final(plan: dict, tag: str, directory: Path, jobs: int) -> None:
    if 'E' in plan.get('arms', {}):
        raise ValueError('WEAK_ORDER.md forbids vault evaluation for arm E')
    check_performance_table(plan, ROOT)
    if 'arms' in plan and not set(DEFAULT_ARMS) <= set(plan['arms']):
        raise ValueError('final paired comparison requires all A/B/C/D arms')
    if 'arms' not in plan and tag not in {e['tag'] for e in plan['tags']}:
        raise ValueError('unknown legacy final tag')
    for experiment in plan['tags']:
        done = directory / experiment['tag'] / 'done.json'
        if not done.exists() or json.loads(done.read_text())['plan_sha256'] != sha(directory / 'plan.json'):
            raise ValueError('finish the routine comparison before the final pick')
        if 'arms' in plan:
            trained = json.loads(done.read_text()).get('checkpoints')
            if not trained:
                raise ValueError('routine scoring must freeze checkpoint hashes before final')
            for relative, digest in trained.items():
                if sha(ROOT / relative) != digest:
                    raise ValueError(f'checkpoint changed after routine comparison: {relative}')
    # Persist intention before even reading vault notation; failed openings stay recorded.
    with (directory / 'final_once.json').open('x') as out:
        json.dump({'tag': tag, 'command': shlex.join(sys.argv)}, out)
        out.flush()
        os.fsync(out.fileno())
    opening_id = vault.open_final(tag)
    atomic(directory / 'final_once.json', json.dumps({
        'tag': tag, 'command': shlex.join(sys.argv), 'opening_id': opening_id,
        'policy': plan.get('final_policy'), 'plan_sha256': sha(directory / 'plan.json')}).encode())
    references = score_rows(final=True)
    excluded = vault.Vault().final_exclusions()
    check_performance_table(plan, ROOT)
    if not references:
        if excluded:
            from chops_vault_score import summarize, write_report
            result = summarize(plan, [], {a: {} for a in DEFAULT_ARMS}, excluded=excluded)
            result['opening_id'] = opening_id
            write_report(result, directory / 'final.json')
            atomic(directory / 'final.log', ''.join(
                f"{v}: {item['reason']}\n" for v, item in sorted(excluded.items())).encode())
            return
        raise ValueError('no readable sealed vault songs; final_once.json retains this attempt')
    import numpy as np
    import torch
    from train import original_model, run
    if not torch.cuda.is_available():
        raise RuntimeError('final inference requires CUDA')
    torch.set_num_threads(1)
    folders = [feature(vault.SONGS / row['slug']) for row in references]
    if 'arms' not in plan:  # interrupted historical plans retain the old one-tag protocol
        for fold in range(len(plan.get('folds', [0, 1, 2]))):
            model = original_model('cuda')
            model.load_state_dict(torch.load(checkpoint(plan, tag, fold), map_location='cuda', weights_only=True))
            model.eval()
            for folder in folders:
                np.save(folder / f'student-{tag}_final-fold{fold}.npy',
                        run(model, np.load(folder / 'feat.npy'), 'cuda').astype(np.float32))
            del model
            torch.cuda.empty_cache()
        check_performance_table(plan, ROOT)
        command([PYTHON, 'tools/harvest/fill_score.py', '--loose', '--final', '--model-tag', tag,
                 '--opening-id', opening_id, '--detect', 'ft_variants:tuned', '--jobs', str(min(jobs, 4)),
                 '--json', str(directory / 'final.json')], directory / 'final.log', tag + '_final')
    else:
        # Snapshot vault inputs only AFTER the durable opening, before prediction.
        hashes = {str(p.relative_to(ROOT)): sha(p) for row in references
                  for p in (vault.SONGS / row['slug'] / 'truth').glob('*.json')}
        snapshot = {'slugs': [r['slug'] for r in references], 'hashes': hashes, 'excluded': excluded,
                    'absent_audits': [str((vault.SONGS / r['slug'] / 'truth/kst_audit.json').relative_to(ROOT))
                                      for r in references if not (vault.SONGS / r['slug'] / 'truth/kst_audit.json').exists()],
                    'checkpoints': {}, 'opening_id': opening_id}
        for arm in DEFAULT_ARMS:
            for e in [e for e in plan['tags'] if e['arm'] == arm]:
                for fold in range(len(plan['folds'])):
                    path = checkpoint(plan, e['tag'], fold)
                    snapshot['checkpoints'][str(path.relative_to(ROOT))] = sha(path)
        atomic(directory / 'final_inputs.json', json.dumps(snapshot, indent=2).encode())
        for arm in DEFAULT_ARMS:
            selected = [e for e in plan['tags'] if e['arm'] == arm]
            if {e['seed'] for e in selected} != {0, 1}:
                raise ValueError('each final arm must contain exactly seeds 0 and 1')
            final_tag = f'{plan["name"]}_{arm}_final'
            for fold in range(len(plan['folds'])):
                for seed_index, e in enumerate(selected):
                    model = original_model('cuda')
                    model.load_state_dict(torch.load(checkpoint(plan, e['tag'], fold), map_location='cuda', weights_only=True))
                    model.eval()
                    for folder in folders:
                        output = folder / f'student-{final_tag}-fold{fold}.npy'
                        act = run(model, np.load(folder / 'feat.npy'), 'cuda').astype(np.float32) / len(selected)
                        if seed_index:
                            act += np.load(output)
                        np.save(output, act)
                    del model
                    torch.cuda.empty_cache()
            for folder in folders:
                (folder / f'snare-student-{final_tag}.npy').unlink(missing_ok=True)
                found = {p.name for p in folder.glob(f'student-{final_tag}-fold?.npy')}
                expected = {f'student-{final_tag}-fold{k}.npy' for k in range(len(plan['folds']))}
                if found != expected:
                    raise ValueError(f'stale final ensemble fold files: {folder} {final_tag}')
        check_performance_table(plan, ROOT)
        command([PYTHON, 'tools/gpu/chops_vault_score.py', '--plan', str(directory / 'plan.json'),
                 '--model-tag', tag, '--opening-id', opening_id, '--jobs', str(min(jobs, 4)),
                 '--json', str(directory / 'final.json')],
                directory / 'final.log')
    check_performance_table(plan, ROOT)
    print('Final vault counts:', directory / 'final.json', flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run', 'score', 'final'))
    p.add_argument('--name', required=True, help='fresh letters/digits/underscore prefix; no hyphens')
    p.add_argument('--jobs', type=int, choices=range(1, 9), default=4, help='parallel channel global scorers (fill/solo cap 4)')
    p.add_argument('--resume', action='store_true', help='continue an unchanged interrupted run')
    p.add_argument('--tag', help='legacy final tag only; new plans always compare all four arm ensembles')
    p.add_argument('--dry-run', action='store_true', help='run: print roster/jobs without writes, CUDA or subprocesses; RETRAIN also reads non-vault truth/cache metadata')
    p.add_argument('--arms', nargs='+', default=list(DEFAULT_ARMS), help='space or comma separated: A B C D E O v5b; RETRAIN: BASE ATH')
    p.add_argument('--base-plan', type=Path, default=ROOT / 'scores/chops/chopround3/plan.json')
    p.add_argument('--holdout', type=Path, default=ROOT / 'scratch/sx_diag/w6/HOLDOUT.json')
    p.add_argument('--batch2-vault', type=Path, default=ROOT / 'scratch/atthedrum/batch2_vault.txt')
    p.add_argument('--dedup-groups', type=Path, help='optional JSON {groups: [[video IDs]]}')
    p.add_argument('--weak-lambda', type=float, choices=(.05, .02), default=.05,
                   help='E CTC coefficient; .05 default, .02 preregistered alternative')
    p.add_argument('--folds', type=int, choices=(2, 3), default=2)
    p.add_argument('--steps', type=int, default=1500, help='equal training steps per arm/seed/fold')
    p.add_argument('--gpu', default='0', help='one physical CUDA index or GPU UUID, passed via CUDA_VISIBLE_DEVICES')
    a = p.parse_args()
    a.arms = [arm for token in a.arms for arm in token.split(',')]
    if any(arm not in RECIPES for arm in a.arms):
        p.error('unknown arm; choose A/B/C/D/E/O/v5b or BASE/ATH')
    if 'E' in a.arms and 'A' not in a.arms:
        p.error('E requires A alongside it for the paired stop rule')
    retrain = bool(set(a.arms) & {'BASE', 'ATH'})
    if retrain and (set(a.arms) != {'BASE', 'ATH'} or a.steps != 1500):
        p.error('RETRAIN requires exactly BASE ATH, 1500 steps')
    if a.command == 'final' and retrain:
        p.error('RETRAIN never opens the vault')
    if a.command == 'final' and 'E' in a.arms:
        p.error('WEAK_ORDER.md forbids vault evaluation for E')
    if not re.fullmatch(r'[A-Za-z0-9_]+', a.name):
        p.error('name must contain letters/digits/underscore only, no hyphens')
    if a.command != 'final' and a.tag:
        p.error('--tag is only for final')
    if a.command != 'run' and a.resume:
        p.error('--resume is only for run')
    if a.command != 'run' and a.dry_run:
        p.error('--dry-run is only for run')
    if a.steps <= 0 or len(a.arms) != len(set(a.arms)):
        p.error('positive steps and unique arms required')
    if not re.fullmatch(r'\d+|GPU-[A-Za-z0-9-]+', a.gpu):
        p.error('--gpu must identify exactly one CUDA index or GPU UUID')
    directory = ROOT / 'scores/chops' / a.name
    plan_path = directory / 'plan.json'
    if a.dry_run and retrain:
        if plan_path.exists():
            plan = load_plan(plan_path)
            if 'retrain' not in plan or len(plan['folds']) != a.folds:
                raise ValueError('dry-run options differ from frozen RETRAIN plan')
        else:
            plan = create_retrain_plan(a.name, a.base_plan, a.holdout, a.batch2_vault,
                                       a.dedup_groups, dry_run=True)
            if len(plan['folds']) != a.folds:
                raise ValueError('--folds differs from frozen BASE plan')
        print_retrain_cohorts(plan)
        return
    if a.dry_run:
        plan = (json.loads(plan_path.read_text()) if plan_path.exists() else
                {'tags': experiments(a.name, a.arms), 'folds': [None] * a.folds, 'steps': a.steps})
        plan.setdefault('steps', 1500)
        if plan_path.exists() and (set(plan.get('arms', {})) != set(a.arms) or
                                   len(plan['folds']) != a.folds or plan['steps'] != a.steps):
            raise ValueError('dry-run arms/folds/steps differ from existing plan; use its frozen options or a fresh name')
        plan.setdefault('channels', list(train_roster(vault.Vault().train_channels)))
        jobs = training_jobs(plan, plan_path)
        print(f'DRY RUN: {len(jobs)} training jobs, {plan["steps"]} steps each, GPU {a.gpu}; '
              'no training/inference/data preparation or writes. Real run freezes shared grouped cohorts in plan.json.')
        print_train_policy(plan)
        if not plan_path.exists():
            print('Audit eligibility is unverified in this preview; the real plan checks TRAIN masks before training.')
        if 'E' in a.arms:
            print(f'E weakorder: lambda={a.weak_lambda}, 1 weak sample per 12 strict samples each step, '
                  'whole system +/-0.65s, CPU CTC; fresh A/E plan with no historical reuse.')
        for e, fold, argv in jobs:
            print(f'arm={e.get("arm", e["recipe"])} seed={e["seed"]} fold={fold} steps={plan["steps"]} '
                  f'CUDA_VISIBLE_DEVICES={a.gpu} ' + shlex.join(argv))
        return
    if a.command in ('run', 'final'):
        require_wsl()
        os.environ['CUDA_VISIBLE_DEVICES'] = a.gpu
        gpu = gpu_identity()
    # Freeze harvesting/config pushes for this comparison. No git required on WSL.
    with lock():
        if a.command == 'run':
            if plan_path.exists():
                if not a.resume:
                    raise ValueError('comparison already exists; use a fresh name or --resume with unchanged inputs')
                plan = load_plan(plan_path)
                if (plan.get('gpu') != gpu or plan.get('steps') != a.steps or
                        len(plan['folds']) != a.folds or set(plan.get('arms', {})) != set(a.arms)):
                    raise ValueError('resume GPU/budget/folds/arms differ from frozen plan')
                if 'E' in a.arms and plan['weak_order']['lambda'] != a.weak_lambda:
                    raise ValueError('resume weak lambda differs from frozen plan')
                print_train_policy(plan)
            else:
                tags = [e['tag'] for e in experiments(a.name, a.arms)]
                for tag in tags:
                    if list((CACHE / 'models').glob(f'{tag}-fold*')) or list(CACHE.glob(f'**/student-{tag}-fold*.npy')):
                        raise ValueError(f'tag already has artifacts: {tag}; use a fresh name')
                directory.mkdir(parents=True, exist_ok=True)
                if retrain:
                    plan = create_retrain_plan(a.name, a.base_plan, a.holdout, a.batch2_vault,
                                               a.dedup_groups, gpu=gpu)
                    if len(plan['folds']) != a.folds:
                        raise ValueError('--folds differs from frozen BASE plan')
                    atomic(plan_path, (json.dumps(plan, indent=2) + '\n').encode())
                else:
                    plan = create_plan(a.name, directory, arms=a.arms, folds=a.folds, steps=a.steps, gpu=gpu,
                                       weak_lambda=a.weak_lambda)
            weak_preflight(plan)
            for experiment in plan['tags']:
                tag = experiment['tag']
                for fold in range(len(plan['folds'])):
                    check_plan(plan)
                    base = CACHE / 'models' / f'{tag}-fold{fold}'
                    if a.resume and base.with_suffix('.pt').exists() and base.with_suffix('.json').exists():
                        checkpoint(plan, tag, fold)
                        continue
                    argv = next(argv for e, k, argv in training_jobs(plan, plan_path) if e == experiment and k == fold)
                    command(argv, directory / tag / f'train_fold{fold}.log')
                predict(plan, tag)
                score_tag(plan, tag, directory, a.jobs)
            report(plan, directory)
        else:
            plan = load_plan(plan_path)
            if a.command == 'score':
                for path in (directory / 'comparison.md', directory / 'comparison.json'):
                    path.unlink(missing_ok=True)
                for experiment in plan['tags']:
                    score_tag(plan, experiment['tag'], directory, a.jobs)
                report(plan, directory)
            else:
                if 'retrain' in plan:
                    raise ValueError('RETRAIN never opens the vault')
                if plan.get('gpu') and plan['gpu'] != gpu:
                    raise ValueError('final GPU differs from comparison')
                if 'arms' in plan and a.tag:
                    p.error('four-arm final freezes both seeds; omit --tag')
                if 'arms' not in plan and not a.tag:
                    p.error('legacy final requires --tag')
                final(plan, a.tag or f'{a.name}_final', directory, a.jobs)



def create_retrain_plan(name, base_plan, holdout, batch2_vault, dedup_groups=None, *, dry_run=False, gpu=None):
    """Read existing non-vault truth/caches only. Data preparation is a separate prerequisite."""
    inputs = retrain_inputs(base_plan, holdout, batch2_vault, dedup_groups)
    seal = vault.Vault()
    original_table, digest = load_performance_table(ROOT)
    table = [*original_table, *inputs['dedup_groups']]
    manifest = [r for r in seal.rows.values() if r.get('slug')]
    # Expand all forbidden groups before even inspecting strict truth.
    preliminary = retrain_rosters(manifest + inputs['base']['songs'], inputs, table, seal)
    blocked = set(preliminary['forbidden'])
    channels = [c for f in inputs['base']['folds'] for c in f['channels']]
    references = score_rows(channels, exclude=blocked)
    eligible = [r for r in references if r['strict']]
    config = retrain_rosters(eligible, inputs, table, seal)
    config['inputs'], config['eligible'] = inputs, eligible
    selected = set(config['arms']['ATH']['songs']) | set(config['held_b'])
    # Primary A is other channels. B is the deterministic within-channel reference.
    references = [r for r in references if r['strict'] and (r['slug'] in config['held_b'] or
                  (r['channel'] != 'atthedrum' and r['slug'] not in config['excluded']))]
    selected |= {r['slug'] for r in references if r['strict']}
    songs, missing = [], []
    for row in eligible:
        if row['slug'] not in selected:
            continue
        path = CACHE / row['slug'] / 'truth.json'
        if path.exists():
            meta = json.loads(path.read_text())
            if (meta['slug'] != row['slug'] or meta['channel'] != row['channel'] or
                    meta.get('video_id') != row['video'] or not meta.get('intervals')):
                raise ValueError('strict cache metadata mismatch: ' + str(path))
            songs.append(meta)
        else:
            missing.append(str(path))
            songs.append({**row, 'video_id': row['video']})
    # Reuse the fixed chop dev roster, never discover a different regression cohort.
    solos = inputs['base'].get('solos', [])
    if not solos:
        missing.append('base plan solos (fixed chop dev roster)')
    all_groups = performance_groups([*manifest, *songs, *references, *solos], table)
    forbidden_groups = {all_groups[r['slug']] for r in manifest if r['slug'] in config['excluded']}
    if any(seal.contains_song(r) or all_groups[r['slug']] in forbidden_groups for r in solos):
        raise ValueError('fixed chop dev overlaps a forbidden performance')
    assignments = retrain_partitions(songs, references, inputs, config, table, solos)
    hashes = {}
    for path in [vault.MANIFEST, vault.CONFIG, *[Path(__file__).with_name(n) for n in
                 ('chops_common.py', 'chops_compare.py', 'chops_scoring.py')]]:
        hashes[str(path.relative_to(ROOT))] = sha(path)
    for relative in inputs['base'].get('hashes', {}):
        if relative.startswith(('tools/', 'src/stickmap/')):
            hashes[relative] = sha(ROOT / relative)
    for row in {r['slug']: r for r in references + songs}.values():
        for path in (vault.SONGS / row['slug'] / 'truth').glob('*.json'):
            hashes[str(path.relative_to(ROOT))] = sha(path)
    for row in references:
        for filename in ('truth.json', 'alignment.json', 'strip.json'):
            path = vault.SONGS / row['slug'] / 'truth' / filename
            if not path.exists():
                missing.append(str(path))
    stamps = {}
    folders = {CACHE / r['slug'] for r in references + songs} | {ROOT / r['cache'] for r in solos}
    training = set(config['arms']['ATH']['songs']) | {s for f in assignments for s in f['held']}
    for folder in sorted(folders):
        for filename in ('feat.npy', 'teacher0.npy') if folder.name in training else ('feat.npy',):
            path = folder / filename
            if not path.exists():
                missing.append(str(path))
            else:
                stat = path.stat()
                stamps[str(path.relative_to(ROOT))] = [stat.st_size, stat.st_mtime_ns]
    for row in songs:
        path = CACHE / row['slug'] / 'truth.json'
        if path.exists():
            hashes[str(path.relative_to(ROOT))] = sha(path)
    for row in solos:
        folder = ROOT / 'truth' / row['video']
        pointer = folder / 'song.txt'
        if not pointer.exists():
            missing.append(str(pointer))
        else:
            song = Path(pointer.read_text().strip())
            if not song.is_absolute() or song.resolve() != (ROOT / row['song']).resolve() or seal.contains_song({'slug': song.name}):
                raise ValueError('fixed chop dev pointer differs from base plan: ' + str(pointer))
            hashes[str(pointer.relative_to(ROOT))] = sha(pointer)
        for filename in ('strip.json', 'alignment.json'):
            if not (folder / filename).exists():
                missing.append(str(folder / filename))
        for path in folder.glob('*.json'):
            hashes[str(path.relative_to(ROOT))] = sha(path)
    plan = {'name': name, 'songs': songs, 'references': references, 'solos': solos,
            'channels': channels, 'folds': assignments, 'arms': config.pop('arms'),
            'tags': experiments(name, ('BASE', 'ATH')), 'steps': 1500, 'gpu': gpu,
            'same_performance': [list(g) for g in original_table], 'same_performance_sha256': digest,
            'hashes': hashes, 'feature_stamps': stamps, 'retrain': config,
            'missing_caches': sorted(set(missing))}
    if not dry_run:
        if missing:
            raise FileNotFoundError('RETRAIN prerequisites missing (no automatic preparation):\n' + '\n'.join(missing))
        for arm in plan['arms']:
            if any(not arm_fold(plan, {'arm': arm}, k)['train'] for k in range(len(assignments))):
                raise ValueError('empty RETRAIN training fold: ' + arm)
        check_plan(plan)
    return plan


def print_retrain_cohorts(plan):
    print('DRY RUN RETRAIN: no writes, training, prediction, preparation or vault opening')
    print('Frozen inputs: ' + json.dumps(plan['retrain']['inputs']['files'], sort_keys=True))
    for arm in plan['arms']:
        for k in range(len(plan['folds'])):
            item = arm_fold(plan, {'arm': arm}, k)
            print(f'{arm} fold{k} seeds=0,1: ' + json.dumps(item, sort_keys=True))
    for k, fold in enumerate(plan['folds']):
        print(f'BASE fold{k} removed from frozen train: ' + json.dumps(fold['base_removed'], ensure_ascii=False, sort_keys=True))
    print('Training exclusions: ' + json.dumps(plan['retrain']['excluded']))
    print('Missing caches/prerequisites: ' + json.dumps(plan.get('missing_caches', [])))


def score_retrain_tag(plan, tag, directory):
    check_plan(plan)
    check_predictions(plan, tag)
    out = directory / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / 'done.json').unlink(missing_ok=True)
    for row in plan['references'] + plan['solos']:
        folder = ROOT / row['cache'] if 'cache' in row else CACHE / row['slug']
        (folder / f'snare-student-{tag}.npy').unlink(missing_ok=True)
    command([PYTHON, 'tools/gpu/chops_scoring.py', '--retrain-plan', str(directory / 'plan.json'),
             '--detect', 'ft_variants:tuned', '--json', str(out / 'retrain.json')], out / 'retrain.log', tag)
    check_plan(plan)
    atomic(out / 'done.json', json.dumps({'plan_sha256': sha(directory / 'plan.json'),
           'checkpoints': {str(checkpoint(plan, tag, k).relative_to(ROOT)): sha(checkpoint(plan, tag, k))
                           for k in range(len(plan['folds']))}}).encode())


def report_retrain(plan, directory):
    from chops_scoring import retrain_statistics, retrain_decision
    check_plan(plan)
    raw = {scope: {a: {} for a in ('BASE', 'ATH')} for scope in ('A', 'B', 'dev')}
    for e in plan['tags']:
        out = directory / e['tag']
        done = json.loads((out / 'done.json').read_text())
        if done['plan_sha256'] != sha(directory / 'plan.json'):
            raise ValueError('incomplete RETRAIN scoring: ' + e['tag'])
        for relative, digest in done['checkpoints'].items():
            if sha(ROOT / relative) != digest:
                raise ValueError('RETRAIN checkpoint changed: ' + relative)
        data = json.loads((out / 'retrain.json').read_text())
        for scope in raw:
            expected = ({r['slug'] for r in plan['solos']} if scope == 'dev' else
                        {s for f in plan['folds'] for s in f['predict_' + scope.lower()]})
            if set(data[scope]) != expected:
                raise ValueError('RETRAIN score song roster mismatch: ' + scope)
            raw[scope][e['arm']][e['seed']] = data[scope]
    primary = {s for f in plan['folds'] if f['ath_effect'] for s in f['predict_a']}
    null = {s for f in plan['folds'] if not f['ath_effect'] for s in f['predict_a']}
    all_a = raw.pop('A')
    for scope, songs in [('A', primary), ('A_null', null)]:
        raw[scope] = {a: {seed: {s: rows[s] for s in songs} for seed, rows in seeds.items()}
                      for a, seeds in all_a.items()}
    scopes = {scope: retrain_statistics(arms) for scope, arms in raw.items()}
    decision = retrain_decision(scopes)
    effect_folds = [k for k, f in enumerate(plan['folds']) if f['ath_effect']]
    null_folds = [k for k, f in enumerate(plan['folds']) if not f['ath_effect']]
    interpretation = (f'Primary: held-out A on atthedrum-training folds {effect_folds}; B is report-only on those folds. '
                      f'Null/noise reference: folds {null_folds} have ATH == BASE training; their A results are A_null '
                      'and do not enter the decision rule. HH/cymbal F1 uses complete strict intervals. '
                      'Snare, hihat and cymbal precision are lower bounds because truth is partially notated. '
                      'Complete assumes usable alignment systems with no non-exportable cymbal-family attacks in strip.json; '
                      'tied continuations are not attacks, and overlapping incomplete systems are excluded. '
                      'All-exported recall uses strict truth; no precision is inferred outside complete intervals. '
                      'Reference eligibility is strict exported truth; channel fold assignments are unchanged. '
                      'K/S/T retain legacy strict/all and strict/fast fill counts; chop dev retains legacy fast counters.')
    result = {'scopes': scopes, 'decision': decision, 'per_song': raw, 'resamples': 2000,
              'bootstrap_seed': 20261004, 'bootstrap_unit': 'song', 'units': 'percentage points',
              'interpretation': interpretation}
    atomic(directory / 'comparison.json', (json.dumps(result, indent=2) + '\n').encode())
    lines = ['# RETRAIN comparison', '', decision['verdict'], '', interpretation, '', decision['spread_policy'], '',
             '| Set | Metric | ΔP (pp) | ΔR (pp) | ΔF1 (pp), 95% CI | Seed spread BASE / ATH (pp) |',
             '|---|---|---|---|---|---|']
    def fmt(v):
        return 'unavailable' if v is None else f'{v:.3f}'
    for scope, metrics in scopes.items():
        for metric, entry in metrics.items():
            diffs = [fmt(entry.get(k, {}).get('difference')) for k in ('P', 'R', 'F1')]
            ci = entry.get('F1', {}).get('ci95', [None, None])
            spreads = [fmt(entry['seed_spread'].get(a)) for a in ('BASE', 'ATH')]
            lines.append(f'| {scope} | {metric} | {diffs[0]} | {diffs[1]} | {diffs[2]} '
                         f'[{fmt(ci[0])}, {fmt(ci[1])}] | {spreads[0]} / {spreads[1]} |')
    lines += ['', 'Decision details: `' + json.dumps(decision, sort_keys=True) + '`', '']
    atomic(directory / 'comparison.md', '\n'.join(lines).encode())


if __name__ == '__main__':
    main()
