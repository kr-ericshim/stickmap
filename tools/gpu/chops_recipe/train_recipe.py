"""Durable A/B v2, C hierarchical, D three-state, optional frozen v5b."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'tools/gpu'), str(ROOT / 'tools/finetune'), str(Path(__file__).parent)]
from chops_common import CACHE, arm_fold, check_plan, load_plan, performance_groups, promotion_policy, splits, vault


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', choices=('v2', 'hierarchical', 'three_state', 'weakorder', 'v5b'), required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--fold', type=int, choices=(0, 1, 2), required=True)
    p.add_argument('--seed', type=int, choices=(0, 1), required=True)
    a = p.parse_args()
    plan = load_plan(a.plan)
    policy = promotion_policy(plan)
    if a.recipe == 'three_state' and not policy['D']['promotion_eligible']:
        print(f'WARNING: {policy["D"]["label"]}; excluded from promotion.', flush=True)
    if a.fold >= len(plan['folds']):
        p.error('fold is outside this plan')
    experiment = next((e for e in plan.get('tags', []) if e['tag'] == a.tag), None)
    if plan.get('tags') and experiment is None:
        raise ValueError('tag is not in the frozen plan')
    if experiment and (experiment['recipe'] != a.recipe or experiment['seed'] != a.seed):
        raise ValueError('tag/recipe/seed differs from frozen plan')
    if not re.fullmatch(r'[A-Za-z0-9_]+', a.tag):
        p.error('tag must contain letters/digits/underscore only, no hyphens')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU/MPS fallback')
    previous_folds = splits.FOLDS
    splits.FOLDS = tuple(tuple(item['channels']) for item in plan['folds'])
    if a.recipe != 'v5b':
        import train
    else:
        import train_synth as synth
        train = synth.train
        synth.P_UNISON_KICK, synth.P_UNISON_TOM, synth.P_DOUBLE = 0.08, 0.05, 0.05  # --low-prior
    previous = {key: getattr(train, key) for key in ('FOLDS', 'same_song', 'MODELS', 'load_songs',
                                                    'split_songs', 'batch', 'loss_fn', 'original_model') if hasattr(train, key)}
    train.FOLDS = splits.FOLDS
    train.same_song = splits.same_song
    train.MODELS = CACHE / 'models'
    selected = arm_fold(plan, experiment, a.fold) if experiment else plan['folds'][a.fold]
    allowed = set(selected['train']) | set(selected['held'])
    state_counts = {}

    def safe_load(device, **options):
        check_plan(plan)
        if options.get('dense'):
            raise ValueError('chops comparison requires dense off')
        seal = vault.Vault()
        teacher = train.original_model(device).eval()
        songs = []
        for slug in sorted(allowed):
            fresh = vault.Vault()
            row = next(r for r in plan['songs'] if r['slug'] == slug)
            if fresh.contains_song(row):
                raise ValueError('vault/cohort changed before strict metadata read')
            folder = CACHE / slug
            meta = json.loads((folder / 'truth.json').read_text())
            if seal.contains_song(meta) or meta['channel'] not in plan['channels']:
                raise ValueError('vault/cohort changed before Song loading')
            kwargs = {'withheld_mask': False} if a.recipe != 'v5b' else {}
            song = train.Song(folder, teacher, device, **kwargs)
            if a.recipe == 'three_state':
                from three_state import masks
                relative = plan.get('audits', {}).get(slug)
                audit = json.loads((ROOT / relative).read_text()) if relative else None
                song.target, song.truth_w, song.kd_w = masks(meta, len(song.feat), audit)
                if slug in selected['train']:
                    for k, cls in enumerate(train.CLASSES[:3]):
                        values = state_counts.setdefault(cls, {'positive': 0, 'negative': 0, 'unknown': 0})
                        values['positive'] += int(((song.target[:, k] > 0) & (song.truth_w[:, k] > 0)).sum())
                        values['negative'] += int(((song.target[:, k] == 0) & (song.truth_w[:, k] > 0)).sum())
                        values['unknown'] += int((song.truth_w[:, k] == 0).sum())
            songs.append(song)
        return songs

    sampler = None
    if a.recipe != 'v5b':
        train.load_songs = safe_load
        def split_songs(songs, fold):
            if str(fold) != str(a.fold):
                raise ValueError('trainer fold differs from plan')
            by_slug = {s.slug: s for s in songs}
            return ([by_slug[s] for s in selected['train']], [by_slug[s] for s in selected['held']])
        train.split_songs = split_songs
        if a.recipe in ('hierarchical', 'three_state'):
            from stratified import HierarchicalSampler, make_batch
            original_load = train.load_songs
            def stratified_load(device, **options):
                nonlocal sampler
                songs = original_load(device, **options)
                training, _ = split_songs(songs, a.fold)
                metas = {s['slug']: s for s in plan['songs']}
                sampler = HierarchicalSampler(training, metas, performance_groups(plan['songs']))
                train.batch = make_batch(train, sampler)
                return songs
            train.load_songs = stratified_load
        if a.recipe == 'three_state':
            from three_state import loss_fn
            train.loss_fn = loss_fn
        weak_summary = None
        if a.recipe == 'weakorder':
            from weakorder import install
            weak_summary = install(train, plan, a.plan, a.fold, a.seed, ROOT)
    else:
        synth._load_songs = safe_load
        # synth.load_songs' known-channel filter now reads the expanded train.FOLDS.
        train.load_songs = synth.load_songs
    sys.argv = [str(Path(train.__file__)), '--device', 'cuda', '--threads', '1',
                '--fold', str(a.fold), '--tag', a.tag, '--seed', str(a.seed),
                '--steps', str(plan.get('steps', 1500)), '--doubt', '0.3']
    if a.recipe != 'v5b':
        sys.argv += ['--dense', '0', '--no-withheld-mask', '--only', *plan['channels']]
    try:
        train.main()
    finally:
        # This is a subprocess in production, but offline tests invoke it in
        # process. Keep the shared current trainer clean for subsequent callers.
        for key, value in previous.items():
            setattr(train, key, value)
        splits.FOLDS = previous_folds
    check_plan(plan)
    path = CACHE / 'models' / f'{a.tag}-fold{a.fold}.json'
    metadata = json.loads(path.read_text())
    if set(metadata['held']) != set(selected['held']):
        raise ValueError('checkpoint held cohort differs from the plan')
    if 'train' in metadata and set(metadata['train']) != set(selected['train']):
        raise ValueError('checkpoint training cohort differs from the plan')
    metadata.update(train=selected['train'], recipe=a.recipe,
                    plan=str(a.plan.resolve().relative_to(ROOT)), low_prior=a.recipe == 'v5b')
    if experiment:
        metadata['arm'] = experiment.get('arm')
    if sampler:
        metadata['sampling'] = sampler.summary()
    if a.recipe == 'three_state':
        metadata['label_policy'] = 'audited negatives only; unknown KST has no BCE/KD; all KST KD off +/-2 frames of any written KST'
        metadata['audits'] = plan.get('audits', {})
        metadata['training_label_counts'] = state_counts
        metadata['promotion'] = policy['D']
    if a.recipe == 'weakorder':
        metadata['weak_order'] = weak_summary()
        metadata['weak_order']['sidecar_sha256'] = plan['hashes'][plan['weak_order']['sidecars'][a.fold]]
    path.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
