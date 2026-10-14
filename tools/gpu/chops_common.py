"""Shared, inspectable cohort, performance grouping and fold policy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / '.eval-cache/ft'
sys.path[:0] = [str(ROOT / 'tools/finetune'), str(ROOT / 'tools/harvest')]
import splits
import vault

# Same performance / recording, NOT merely the same composition. Grouping only makes splits stricter,
# so unconfirmed but likely groups are listed too. Evidence: truth/harvest/PILOT_NEW.md "Same-performance checks".
# Checked and NOT the same performance: AlexBlack J-Rod 2AzwR38fHKU vs truth/jrod bd17mxbLaNE; Tim Buell Larnell
# eik7Wk03o-8 vs i1nnfqkv5WY / XUdFq81ZSRQ; El Baquetazo Tony EEJEkbpBnjY vs Qo5aDGoU7sQ.
SAME_PERFORMANCE: tuple[tuple[str, ...], ...] = (
    # confirmed: AlexBlack "Change Your Mind", Ivannikov Larnell solo, Alfio overdub of the same shoot,
    # Tim Buell "Larnell Lewis Change Your Mind"
    ("N0PXFlgigjw", "ZsMtIfjkubo", "K_AtZoNIoKo", "6CLJH8YJtFM"),
    ("9juoKgimeR4", "koBhkNbncAw"),  # unconfirmed, conservative: Eric Moore on dailychops and Nyenty
)


# Arm O: the A cohort plus only the new Ollie Woods Shorts strict truth (isolates that channel's effect).
ARM_EXTRA_CHANNELS = {'O': ('olliewoods_shorts',)}


def arm_includes(arm: str, channel: str) -> bool:
    """A/E train on the original 14 channels; O adds its listed channels; other arms take every strict song."""
    if arm in ('A', 'E'):
        return channel in splits.ORIGINAL_CHANNELS
    if arm in ARM_EXTRA_CHANNELS:
        return channel in splits.ORIGINAL_CHANNELS or channel in ARM_EXTRA_CHANNELS[arm]
    return True


def promotion_policy(plan: dict) -> dict:
    """D is diagnostic unless every training fold has usable audited negatives.

    Evidence is frozen from TRAIN masks, never inferred from an unrelated or
    sealed audit file. Eligibility only clears this gate; it is not promotion.
    """
    supported = set(plan.get('audited_negative_songs', []))
    folds = plan.get('folds', [])
    covered = [bool(supported.intersection(f.get('train', []))) for f in folds if f is not None]
    eligible = bool(covered) and all(covered)
    label = ('D' if eligible else 'D (audit-incomplete stress test)' if any(covered)
             else 'D (positive-only stress test)')
    arms = plan.get('arms', {e.get('arm', e.get('recipe', 'legacy')): {} for e in plan.get('tags', [])})
    return {'eligible_arms': [a for a in arms if a != 'D' or eligible],
            'D': {'label': label, 'promotion_eligible': eligible,
                  'audited_training_folds': covered,
                  'reason': ('audit gate satisfied; performance/regression gates still required' if eligible else
                             'no usable audited K/S/T negatives in one or more training folds')}}


def arm_label(arm: str, plan: dict | None = None) -> str:
    return promotion_policy(plan or {})['D']['label'] if arm == 'D' else arm


def train_roster(train_channels) -> tuple[str, ...]:
    """Channel candidates only: vault.contains_song remains the song-level gate."""
    return tuple(sorted({c for group in splits.FOLDS for c in group} |
                        set(splits.ORIGINAL_CHANNELS) | set(train_channels)))


def load_performance_table(root=None):
    """Load optional owner-authored groups; hash the exact bytes that were parsed."""
    path = (Path(root) / 'truth/harvest' if root is not None else vault.OUT) / 'same_performance.json'
    extra, digest = vault.performance_table_file(path)
    return (*SAME_PERFORMANCE, *extra), digest


def check_performance_table(plan, root=None) -> None:
    table, digest = load_performance_table(ROOT if root is None else root)
    if 'same_performance_sha256' not in plan:
        raise ValueError('plan lacks same_performance.json snapshot; use a fresh comparison name')
    if plan['same_performance_sha256'] != digest:
        raise ValueError('same_performance.json changed since routine plan; use a fresh comparison name')
    if plan.get('same_performance') != [list(g) for g in table]:
        raise ValueError('SAME_PERFORMANCE changed since plan; use a fresh comparison name')


def performance_groups(rows, table=None) -> dict[str, str]:
    """Transitive video-ID components, stable even when a peer is absent."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for group in load_performance_table()[0] if table is None else table:
        if (not isinstance(group, (list, tuple)) or len(group) < 2 or
                any(not isinstance(v, str) or not v.strip() for v in group)):
            raise ValueError('SAME_PERFORMANCE needs groups of at least two nonempty video IDs')
        for video in group:
            a, b = find(group[0]), find(video)
            parent[max(a, b)] = min(a, b)
    result = {}
    for row in rows:
        video = row.get('video') or row.get('video_id') or row['slug']
        result[row['slug']] = find(video)
    return result


def require_sealed_groups(seal, table=None) -> dict[str, list[str]]:
    """Return sealed IDs disqualified by non-sealed peers; never expand the seal.

    Missing manifest peers are non-sealed unless explicitly sealed. The same
    transitive components drive training splits and vault eligibility.
    Invalid/unreadable tables raise before any sealed truth is accessed.
    """
    table = load_performance_table()[0] if table is None else table
    ids = set(seal.rows) | set(seal.explicit) | {v for group in table for v in group}
    rows = [{**seal.rows.get(v, {}), 'slug': v, 'video': v} for v in sorted(ids)]
    groups = performance_groups(rows, table)
    peers = {}
    for r in rows:
        if not seal.contains_song(r):
            peers.setdefault(groups[r['slug']], []).append(r['video'])
    return {r['video']: peers[groups[r['slug']]] for r in rows
            if seal.contains_song(r) and groups[r['slug']] in peers}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fold_policy(train_channels, count=3) -> tuple[tuple[str, ...], ...]:
    """Fixed two-way or historical three-way base; new channels use a name hash."""
    if count not in (2, 3):
        raise ValueError('fold count must be 2 or 3')
    groups = ([list(group) for group in splits.FOLDS] if count == 3 else
              [list(splits.ORIGINAL_CHANNELS[k::2]) for k in range(2)])
    assigned = {c for group in groups for c in group}
    for channel in sorted(set(train_roster(train_channels)) - assigned):
        if not isinstance(channel, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', channel):
            raise ValueError('invalid vault train channel')
        groups[int(hashlib.sha256(channel.encode()).hexdigest(), 16) % count].append(channel)
    return tuple(tuple(group) for group in groups)


def partitions(metadata: list[dict], references: list[dict], folds, table=None) -> list[dict]:
    # Cross-channel performances connect their channels. Move each connected
    # channel component to its lowest original fold; a channel remains held out
    # as a whole and no registered performance can straddle train and held-out.
    rows = {row['slug']: row for row in metadata + references}
    groups = performance_groups(rows.values(), table)
    channel_fold = {c: k for k, channels in enumerate(folds) for c in channels}
    if any(r['channel'] not in channel_fold for r in rows.values()):
        raise ValueError('reference/training channel missing from fold policy')
    components = {}
    for slug, row in rows.items():
        components.setdefault(groups[slug], set()).add(row['channel'])
    changed = True
    while changed:
        changed = False
        for channels in components.values():
            target = min(channel_fold[c] for c in channels)
            for c in channels:
                if channel_fold[c] != target:
                    channel_fold[c] = target
                    changed = True
    assigned = [[c for c, k in channel_fold.items() if k == i] for i in range(len(folds))]
    result = []
    for channels in assigned:
        held = [row['slug'] for row in metadata if row['channel'] in channels]
        ref = [row['slug'] for row in references if row['channel'] in channels]
        train = [row['slug'] for row in metadata if row['channel'] not in channels
                 and not any(splits.same_song(row['slug'], slug) for slug in ref)]
        result.append({'channels': list(channels), 'train': train, 'held': held,
                       'predict': ref})
    return result


def check_plan(plan: dict) -> None:
    if 'retrain' in plan:
        check_retrain_plan(plan)
        return
    seal = vault.Vault()
    for row in plan['songs'] + plan.get('references', []):
        if seal.contains_song(row):
            raise ValueError(f'forbidden training song: {row["slug"]}')
    check_performance_table(plan)
    for relative, expected in plan['hashes'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f'input/code changed since plan: {relative}; use a fresh comparison name')
    for relative, expected in plan.get('feature_stamps', {}).items():
        path = ROOT / relative
        stat = path.stat()
        if [stat.st_size, stat.st_mtime_ns] != expected:
            raise ValueError(f'feature/teacher cache changed since plan: {relative}')
    require_sealed_groups(seal)
    weak_channels = plan.get('weak_channels', [])
    if weak_channels and 'E' not in plan.get('arms', {}):
        raise ValueError('weak channel extension requires arm E')
    expected_weak_channels = sorted({r['channel'] for r in seal.rows.values()
                                    if r.get('channel') and not seal.contains_song(r)})
    if 'E' in plan.get('arms', {}) and weak_channels != expected_weak_channels:
        raise ValueError('weak TRAIN channel roster changed since plan')
    allowed = set(train_roster(set(seal.train_channels) | set(weak_channels)))
    if set(plan['channels']) != allowed:
        raise ValueError('TRAIN roster changed since plan; use a fresh comparison name')
    for row in plan['songs'] + plan.get('references', []):
        if row['channel'] not in allowed or seal.contains_song(row):
            raise ValueError(f'forbidden training song: {row["slug"]}')
    for item in plan['folds']:
        if set(item['train']) & set(item['held']):
            raise ValueError('train/held overlap')
        by_slug = {s['slug']: s for s in plan['songs']}
        if any(by_slug[s]['channel'] in item['channels'] for s in item['train']):
            raise ValueError('held channel leaked into training')
        groups = performance_groups(plan['songs'] + plan.get('references', []))
        held_groups = {groups[s] for s in item['held'] + item.get('predict', [])}
        if any(groups[s] in held_groups for s in item['train']):
            raise ValueError('same performance leaked into training')
        for slug in item['train']:
            if any(splits.same_song(slug, ref) for ref in item.get('predict', [])):
                raise ValueError('same song leaked into training')
    if 'arms' in plan:
        for arm, info in plan['arms'].items():
            expected = {r['slug'] for r in plan['songs'] if arm_includes(arm, r['channel'])}
            if set(info['songs']) != expected:
                raise ValueError(f'arm cohort mismatch: {arm}')
        roster = {(e['arm'], e['seed']) for e in plan['tags']}
        if roster != {(a, s) for a in plan['arms'] for s in (0, 1)} or len(roster) != len(plan['tags']):
            raise ValueError('each arm needs exactly two frozen seed tags')
    for relative in plan.get('absent_audits', []):
        if (ROOT / relative).exists():
            raise ValueError(f'audit added since plan: {relative}; use a fresh comparison name')
    for relative in plan.get('absent_weak_reviews', []):
        if (ROOT / relative).exists():
            raise ValueError(f'weak review added since plan: {relative}; use a fresh comparison name')
    if 'E' in plan.get('arms', {}):
        from weak_set import POLICY, seal_fingerprint, external_exclusions, external_solos, manifest_rows
        solo, _ = external_solos(ROOT)
        rows = [r for r in manifest_rows(seal) if not seal.contains_song(r)]
        expected = external_exclusions(rows, solo, load_performance_table(ROOT)[0])
        if plan.get('external_solo_exclusions') != expected:
            raise ValueError('frozen external-solo exclusions differ from current metadata')
        if any(r['slug'] in expected for r in plan['songs'] + plan.get('references', [])):
            raise ValueError('external-solo peer leaked into strict/scoring cohort')
        config = plan.get('weak_order', {})
        if (config.get('policy_version') != POLICY or config.get('lambda') not in (.05, .02)
                or config.get('batch_size') != 1 or config.get('ctc_backend') != 'cpu'
                or len(config.get('sidecars', [])) != len(plan['folds'])):
            raise ValueError('invalid frozen weak-order recipe')
        for fold, relative in enumerate(config['sidecars']):
            data = json.loads((ROOT / relative).read_text())
            if data.get('quality_fold') != fold or data.get('vault_seal_fingerprint') != seal_fingerprint():
                raise ValueError('weak sidecar fold/seal mismatch')
            for slug, windows in data['songs'].items():
                if windows and seal.contains_song({'slug': slug}):
                    raise ValueError('sealed weak song in sidecar')


def arm_fold(plan, experiment, fold) -> dict:
    item = dict(plan['folds'][fold])
    if 'train_by_arm' in item:
        item['train'] = item['train_by_arm'][experiment['arm']]
    if 'arms' in plan:
        allowed = set(plan['arms'][experiment['arm']]['songs'])
        item['train'] = [slug for slug in item['train'] if slug in allowed]
    # All arms predict the same held-out references, including the new data.
    return item


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    check_plan(plan)
    return plan


RETRAIN_ARMS = ('BASE', 'ATH')


def retrain_inputs(base_plan, holdout, batch2_vault, dedup_groups=None):
    """Required owner files fail closed; hashes cover the exact parsed bytes."""
    paths = {'base_plan': base_plan, 'holdout': holdout, 'batch2_vault': batch2_vault}
    if dedup_groups is not None:
        paths['dedup_groups'] = dedup_groups
    missing = [str(p) for p in paths.values() if p is None or not Path(p).is_file()]
    if missing:
        raise FileNotFoundError('missing RETRAIN inputs: ' + ', '.join(missing))
    raw = {k: Path(p).read_bytes() for k, p in paths.items()}
    base = json.loads(raw['base_plan'])
    held = json.loads(raw['holdout'])['holdout']
    if not isinstance(held, list):
        raise ValueError('HOLDOUT.holdout must be a list')
    def identifiers(values):
        result = set()
        for v in values:
            ids = [v] if isinstance(v, str) else [v.get(k) for k in ('slug', 'video', 'video_id')]
            if not any(isinstance(i, str) and i.strip() for i in ids):
                raise ValueError('exclusion needs a slug or video ID')
            result.update(i for i in ids if isinstance(i, str) and i.strip())
        return sorted(result)
    batch = [line.split('#', 1)[0].strip() for line in raw['batch2_vault'].decode().splitlines()]
    if any(len(v.split()) != 1 for v in batch if v):
        raise ValueError('batch2_vault requires one video ID or slug per line')
    extra = json.loads(raw['dedup_groups'])['groups'] if 'dedup_groups' in raw else []
    performance_groups([], extra)  # Validate even absent peers.
    songs = base['songs']
    if not songs or any(not isinstance(r, dict) or not r.get('slug') for r in songs):
        raise ValueError('base plan needs its original strict songs metadata')
    if len({r['slug'] for r in songs}) != len(songs):
        raise ValueError('duplicate base song')
    return {'base': base, 'holdout': identifiers(held), 'batch2_vault': identifiers([v for v in batch if v]),
            'dedup_groups': extra,
            'files': {k: {'path': str(Path(paths[k]).resolve()), 'sha256': hashlib.sha256(v).hexdigest()}
                      for k, v in raw.items()}}


def retrain_rosters(rows, inputs, table, seal):
    """Eligible strict rows in, frozen song cohorts out; expand before selecting B."""
    base_rows = inputs['base']['songs']
    all_rows = {r['slug']: r for r in [*seal.rows.values(), *base_rows, *rows] if r.get('slug')}
    ids = set(inputs['holdout']) | set(inputs['batch2_vault']) | {'bMUE_E21yI8'} | set(seal.explicit)
    peers = {v for g in table for v in g} | ids
    synthetic = [{'slug': 'id:' + v, 'video': v} for v in peers]
    groups = performance_groups([*all_rows.values(), *synthetic], table)
    def expand(values):
        targets = {groups['id:' + v] for v in values if 'id:' + v in groups}
        targets.update(groups[s] for s in values if s in all_rows)
        return {s for s in all_rows if groups[s] in targets}
    hold = expand(inputs['holdout'])
    reason_sets = {'HOLDOUT': expand(inputs['holdout']),
                   'batch2 vault': expand(inputs['batch2_vault']),
                   'J-Rod': expand({'bMUE_E21yI8'}), 'vault': expand(seal.explicit)}
    forbidden = expand(ids)
    sealed_groups = {groups[s] for s, r in all_rows.items() if seal.contains_song(r)}
    sealed_groups.update(groups[r['slug']] for r in synthetic if seal.contains_song(r))
    reason_sets['vault'].update(s for s in all_rows if groups[s] in sealed_groups)
    forbidden.update(reason_sets['vault'])
    eligible = {r['slug'] for r in rows}
    selected = [r['slug'] for r in rows if r['channel'] == 'atthedrum' and r['slug'] not in forbidden
                and int(hashlib.sha256(('groovemap-retrain-ath:' +
                                        (r.get('video') or r['video_id'])).encode()).hexdigest(), 16) % 5 == 0]
    b_groups = {groups[s] for s in selected}
    held_b = {s for s in eligible if groups[s] in b_groups}
    if held_b & forbidden or held_b & hold:
        raise ValueError('RETRAIN B overlaps HOLDOUT/excluded performance')
    exclusions = forbidden | {s for s in all_rows if groups[s] in b_groups}
    reason_sets['B'] = {s for s in all_rows if groups[s] in b_groups}
    reasons = {s: [reason + ' (including performance groups)' for reason, members in reason_sets.items()
                   if s in members] for s in sorted(exclusions)}
    original = {r['slug'] for r in base_rows}
    missing = original - eligible - exclusions
    if missing:
        raise ValueError('frozen BASE songs lack eligible strict truth: ' + ', '.join(sorted(missing)))
    base = original - exclusions
    ath = (original | {r['slug'] for r in rows if r['channel'] == 'atthedrum'}) - exclusions
    arms = {'BASE': {'songs': sorted(base)}, 'ATH': {'songs': sorted(ath)}}
    for arm in arms.values():
        if set(arm['songs']) & exclusions:
            raise ValueError('excluded RETRAIN training song')
    return {'arms': arms, 'held_b': sorted(held_b), 'excluded': sorted(exclusions),
            'forbidden': sorted(forbidden), 'holdout': sorted(hold), 'base_original': sorted(original),
            'exclusion_reasons': reasons}


def check_retrain_plan(plan):
    config = plan['retrain']
    for info in config['inputs']['files'].values():
        if sha(Path(info['path'])) != info['sha256']:
            raise ValueError('RETRAIN input changed: ' + info['path'])
    for relative, digest in plan['hashes'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError('RETRAIN input/code changed: ' + relative)
    check_performance_table(plan)
    seal = vault.Vault()
    table = plan['same_performance'] + config['inputs']['dedup_groups']
    expected = retrain_rosters(config['eligible'], config['inputs'], table, seal)
    if any(config[k] != expected[k] for k in expected if k != 'arms') or plan['arms'] != expected['arms']:
        raise ValueError('RETRAIN frozen roster mismatch')
    if plan['steps'] != 1500 or set(plan['arms']) != set(RETRAIN_ARMS):
        raise ValueError('RETRAIN requires BASE/ATH and 1500 steps')
    if sorted((e['arm'], e['seed'], e['recipe']) for e in plan['tags']) != [
            (a, s, 'v2') for a in sorted(RETRAIN_ARMS) for s in (0, 1)]:
        raise ValueError('RETRAIN requires v2 and exactly seeds 0/1')
    folds = retrain_partitions(plan['songs'], plan['references'], config['inputs'], expected, table, plan['solos'])
    if plan['folds'] != folds:
        raise ValueError('RETRAIN frozen fold mismatch')
    rows = plan['songs'] + plan['references'] + plan['solos']
    groups = performance_groups(rows, table)
    forbidden = set(config['excluded'])
    if set(config['held_b']) & set(config['holdout']):
        raise ValueError('B overlaps HOLDOUT')
    for row in rows:
        if seal.contains_song(row) or row['slug'] in config['forbidden']:
            raise ValueError('forbidden RETRAIN input: ' + row['slug'])
    for k, fold in enumerate(plan['folds']):
        for arm in RETRAIN_ARMS:
            item = arm_fold(plan, {'arm': arm}, k)
            held_groups = {groups[s] for s in item['held'] + item['predict']} | {
                groups[s['slug']] for s in plan['solos']}
            if any(s in forbidden or groups[s] in held_groups for s in item['train']):
                raise ValueError('excluded/same-performance RETRAIN train overlap')
            if any(next(r for r in plan['songs'] if r['slug'] == s)['channel'] in fold['channels']
                   for s in item['train']):
                raise ValueError('held channel leaked into RETRAIN training')
    for relative, stamp in plan.get('feature_stamps', {}).items():
        stat = (ROOT / relative).stat()
        if [stat.st_size, stat.st_mtime_ns] != stamp:
            raise ValueError('RETRAIN cache changed: ' + relative)


def retrain_partitions(songs, references, inputs, cohorts, table, solos=()):
    """Preserve historical channel folds; new dedup edges remove train peers, never move channels."""
    original = inputs['base']['folds']
    channels = [c for f in original for c in f['channels']]
    if len(set(channels)) != len(channels):
        raise ValueError('base plan repeats a held channel')
    if any(r['channel'] not in channels for r in songs + references):
        raise ValueError('RETRAIN channel absent from frozen BASE folds')
    groups = performance_groups(songs + references + list(solos), table)
    b = set(cohorts['held_b'])
    excluded = set(cohorts['excluded'])
    dev_groups = {groups[r['slug']] for r in solos}
    dev_slugs = [r['slug'] for r in solos]
    folds = []
    by_slug = {r['slug']: r for r in songs}
    base_original = set(cohorts['base_original'])
    extra = set(cohorts['arms']['ATH']['songs']) - base_original
    for frozen in original:
        frozen_train = frozen['train']  # Global songs cannot reconstruct historical per-fold removals.
        if len(set(frozen_train)) != len(frozen_train) or not set(frozen_train) <= base_original:
            raise ValueError('invalid frozen BASE fold train roster')
        active = 'atthedrum' not in frozen['channels']
        held = [r['slug'] for r in songs if r['channel'] in frozen['channels']]
        predict_a = [r['slug'] for r in references if r['channel'] in frozen['channels'] and r['slug'] not in b]
        predict_b = [r['slug'] for r in references if r['slug'] in b] if active else []
        predict = predict_a + predict_b
        held_groups = {groups[s] for s in held + predict}
        def reasons(slug):
            result = list(cohorts['exclusion_reasons'].get(slug, []))
            if slug in excluded:
                return result
            if slug not in by_slug:
                raise ValueError('frozen BASE fold song missing: ' + slug)
            if by_slug[slug]['channel'] in frozen['channels']:
                raise ValueError('frozen BASE train contains a held channel: ' + slug)
            if groups[slug] in held_groups or any(splits.same_song(slug, s) for s in predict):
                result.append('held-out performance group')
            if groups[slug] in dev_groups or any(splits.same_song(slug, s) for s in dev_slugs):
                result.append('chop-dev overlap')
            return result
        removed = {s: reasons(s) for s in frozen_train if reasons(s)}
        base = [s for s in frozen_train if s not in removed]
        added = [r['slug'] for r in songs if active and r['slug'] in extra and not reasons(r['slug'])]
        train = base + added
        assert set(base) <= set(frozen_train)
        folds.append({'channels': list(frozen['channels']), 'train': train, 'held': held,
                      'train_by_arm': {'BASE': base, 'ATH': train}, 'base_removed': removed,
                      'ath_effect': active, 'predict': predict, 'predict_a': predict_a,
                      'predict_b': predict_b})
    return folds
