"""Synthetic offline verification; no weights, audio, training or inference."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE / 'chops_recipe')]
import chops_common as common
import chops_compare as compare
from chops_statistics import paired_bootstrap, sufficiency
from stratified import HierarchicalSampler
from three_state import loss_fn, masks


def row(slug, channel, video=None):
    return {'slug': slug, 'channel': channel, 'video_id': video or slug,
            'intervals': [[0., 10.]], 'events': {'kick': [1., 1.06], 'snare': [2., 2.06], 'tom': [3., 3.06]}}


def test_transitive_performance_group_joins_channels_and_loose_only_refs(monkeypatch):
    monkeypatch.setattr(common, 'SAME_PERFORMANCE', (('daily', 'tim'), ('tim', 'nyenty')))
    rows = [row('a', 'swartz', 'daily'), row('b', 'drumshin', 'tim'), row('c', 'yeol', 'nyenty'),
            row('independent', 'nakamura')]
    loose = row('loose', 'dailychops', 'nyenty')
    folds = common.partitions(rows, rows + [loose], common.fold_policy(['dailychops']))
    groups = common.performance_groups(rows + [loose])
    assert groups['a'] == groups['b'] == groups['c'] == groups['loose']
    assert len([f for f in folds if any(s in f['held'] for s in ('a', 'b', 'c'))]) == 1
    for f in folds:
        assert not ({groups[s] for s in f['train']} & {groups[s] for s in f['predict']})
        assert {r['channel'] for r in rows if r['slug'] in f['train']}.isdisjoint(f['channels'])


def test_arm_a_uses_only_original_data_with_identical_held_refs():
    rows = [row('old', 'swartz'), row('new', 'dailychops'), row('other', 'drumshin')]
    plan = {'songs': rows, 'folds': common.partitions(rows, rows, common.fold_policy(['dailychops'], 2)),
            'arms': {'A': {'songs': ['old', 'other']}, 'B': {'songs': ['old', 'new', 'other']}}}
    for k in range(2):
        a = common.arm_fold(plan, {'arm': 'A'}, k)
        b = common.arm_fold(plan, {'arm': 'B'}, k)
        assert 'new' not in a['train']
        assert a['held'] == b['held'] and a['predict'] == b['predict']


@pytest.mark.parametrize('extra_groups', [False, True])
@pytest.mark.parametrize('audit_mode', ['none', 'empty', 'negative'])
def test_create_plan_freezes_real_arm_cohorts_and_16_jobs_without_preparation(tmp_path, monkeypatch, extra_groups, audit_mode):
    rows = [row('old0', 'swartz'), row('old1', 'allyeodrum'), row('new', 'dailychops'),
            row('moyin', 'moyin'), row('olliewoods', 'olliewoods'), row('golly', 'golly')]
    references = [{**r, 'video': r['video_id'], 'strict': True, 'loose': True} for r in rows]
    references.append({**row('sealed', 'moyin'), 'video': 'sealed', 'strict': True, 'loose': True})
    seal = SimpleNamespace(train_channels={'dailychops'}, explicit=set(),
                           rows={r['video_id']: {**r, 'video': r['video_id']} for r in rows},
                           contains_song=lambda r: r['slug'] == 'sealed')
    cache = tmp_path / 'ft'
    for r in rows:
        folder = cache / r['slug']
        folder.mkdir(parents=True)
        (folder / 'truth.json').write_text(json.dumps(r))
        np.save(folder / 'feat.npy', np.zeros((1100, 2)))
        if audit_mode != 'none':
            truth = tmp_path / 'songs' / r['slug'] / 'truth'
            truth.mkdir(parents=True)
            payload = audit(verified_negative=[{'class': 'kick', 'start': 4., 'end': 4.1}]
                            if audit_mode == 'negative' else [])
            (truth / 'kst_audit.json').write_text(json.dumps(payload))
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    monkeypatch.setattr(common.vault, 'OUT', tmp_path / 'truth/harvest')
    table_file = common.vault.OUT / 'same_performance.json'
    if extra_groups:
        table_file.parent.mkdir(parents=True)
        table_file.write_text(json.dumps({'groups': [['old0', 'bridge']], 'source': 'fixture', 'generated': 'fixture'}))
    monkeypatch.setattr(compare, 'CACHE', cache)
    monkeypatch.setattr(common, 'SAME_PERFORMANCE', ())
    monkeypatch.setattr(compare.vault, 'Vault', lambda: seal)
    monkeypatch.setattr(compare.vault, 'SONGS', tmp_path / 'songs')
    monkeypatch.setattr(compare, 'command', lambda *a, **kw: None)
    def feature(song):
        assert song.name != 'sealed', 'sealed features accessed'
        return cache / song.name
    monkeypatch.setattr(compare, 'feature', feature)
    monkeypatch.setattr(compare, 'score_rows', lambda channels: references)
    monkeypatch.setattr(compare, 'solo_inputs', lambda: [{'slug': 'solo', 'video': 'solo_video', 'cache': 'solos/solo'}])
    monkeypatch.setattr(compare, 'sha', lambda path: 'fixture_digest')
    directory = tmp_path / 'comparison'
    plan = compare.create_plan('fixture', directory, gpu={'name': 'fixture GPU'})
    assert plan['arms']['A']['songs'] == ['old0', 'old1']
    assert all(plan['arms'][a]['songs'] == [r['slug'] for r in rows] for a in 'BCD')
    assert 'sealed' not in {r['slug'] for r in plan['references']}
    assert plan['promotion']['D']['promotion_eligible'] == (audit_mode == 'negative')
    assert ('D' in plan['promotion']['eligible_arms']) == (audit_mode == 'negative')
    for fold in plan['cohort_diagnostics']['B']:
        assert sum(scope['fast_positive_frames']['tom'] for scope in fold.values()) == 12
        assert sum(scope['accepted_seconds']['original'] for scope in fold.values()) == 20
        assert sum(scope['accepted_seconds']['expanded'] for scope in fold.values()) == 40
    assert sum(f['train']['fast_positive_frames']['tom'] for f in plan['cohort_diagnostics']['A']) == 4
    assert len(compare.training_jobs(plan, directory / 'plan.json')) == 16
    assert json.loads((directory / 'plan.json').read_text()) == plan
    assert plan['final_policy']['resamples'] == 10000
    assert plan['final_policy']['bootstrap_seed'] == 20261002
    assert plan['same_performance_sha256'] == (common.sha(table_file) if extra_groups else None)
    if extra_groups:
        assert plan['hashes']['truth/harvest/same_performance.json'] == common.sha(table_file)
        assert plan['same_performance'] == [['old0', 'bridge']]
        assert plan['performance_groups']['old0'] == 'bridge'


def test_mixed_performance_excludes_sealed_member_without_sealing_train(monkeypatch):
    monkeypatch.setattr(common, 'SAME_PERFORMANCE', (('sealed', 'train'),))
    seal = SimpleNamespace(rows={'train': {'video': 'train', 'channel': 'dailychops'}}, explicit={'sealed'},
                           contains_song=lambda r: r.get('video') == 'sealed')
    assert common.require_sealed_groups(seal) == {'sealed': ['train']}
    assert not seal.contains_song({'video': 'train'})


def test_sealed_group_also_protects_peer_absent_from_manifest(monkeypatch):
    monkeypatch.setattr(common, 'SAME_PERFORMANCE', (('sealed', 'external_solo'),))
    seal = SimpleNamespace(rows={}, explicit={'sealed'},
                           contains_song=lambda r: r.get('video') == 'sealed')
    assert common.require_sealed_groups(seal) == {'sealed': ['external_solo']}


def test_json_table_transitively_merges_builtin_and_freezes_exact_sha(tmp_path, monkeypatch):
    out = tmp_path / 'truth/harvest'
    out.mkdir(parents=True)
    monkeypatch.setattr(common.vault, 'OUT', out)
    monkeypatch.setattr(common, 'SAME_PERFORMANCE', (('a', 'b'),))
    path = out / 'same_performance.json'
    path.write_text(json.dumps({'groups': [['b', 'c'], ['c', 'd']],
                                'source': 'offline matching', 'generated': 'fixture'}))
    table, digest = common.load_performance_table(tmp_path)
    assert table == (('a', 'b'), ('b', 'c'), ('c', 'd'))
    assert digest == common.sha(path)
    groups = common.performance_groups([row('one', 'swartz', 'a'), row('two', 'drumshin', 'd')])
    assert groups == {'one': 'a', 'two': 'a'}
    plan = {'same_performance': [list(g) for g in table], 'same_performance_sha256': digest}
    common.check_performance_table(plan, tmp_path)
    path.write_text(path.read_text() + '\n')  # even unchanged semantics invalidate the frozen bytes
    with pytest.raises(ValueError, match='changed since routine'):
        common.check_performance_table(plan, tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match='changed since routine'):
        common.check_performance_table(plan, tmp_path)
    plan['same_performance_sha256'] = None
    plan['same_performance'] = [['a', 'b']]
    common.check_performance_table(plan, tmp_path)
    path.write_text('{"groups": []}')
    with pytest.raises(ValueError, match='changed since routine'):
        common.check_performance_table(plan, tmp_path)


@pytest.mark.parametrize('payload', ['broken', '{}', '{"groups": ["abc"]}',
                                    '{"groups": [["a"]]}', '{"groups": [["a", ""]]}'])
def test_invalid_json_table_refuses_final_before_opening(tmp_path, monkeypatch, payload):
    out = tmp_path / 'truth/harvest'
    out.mkdir(parents=True)
    (out / 'same_performance.json').write_text(payload)
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    monkeypatch.setattr(compare.vault, 'open_final', lambda *a: pytest.fail('invalid table opened vault'))
    with pytest.raises(ValueError):
        compare.final({}, 'fixture', tmp_path, 1)
    assert not (tmp_path / 'final_once.json').exists()


def test_final_rejects_table_change_before_opening(tmp_path, monkeypatch):
    out = tmp_path / 'truth/harvest'
    out.mkdir(parents=True)
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    table, digest = common.load_performance_table(tmp_path)
    plan = {'same_performance': [list(g) for g in table], 'same_performance_sha256': digest}
    (out / 'same_performance.json').write_text('{"groups": [["one", "two"]]}')
    monkeypatch.setattr(compare.vault, 'open_final', lambda *a: pytest.fail('changed table opened vault'))
    with pytest.raises(ValueError, match='changed since routine'):
        compare.final(plan, 'fixture', tmp_path, 1)
    assert not (tmp_path / 'final_once.json').exists()


def sampler_fixture():
    metas = {s: row(s, c) for s, c in [('short', 'swartz'), ('long', 'dailychops'), ('peer', 'dailychops')]}
    for meta in metas.values():
        meta['events']['kick'] += [6.]
        meta['events']['snare'] += [7.]
        meta['events']['tom'] += [8.]
    # Long recording and another clip share one performance; short != long.
    songs = [SimpleNamespace(slug=s, channel=m['channel'], feat=np.zeros((n, 2)))
             for (s, m), n in zip(metas.items(), [1000, 10000, 1000])]
    sampler = HierarchicalSampler(songs, metas, {'short': 'one', 'long': 'two', 'peer': 'two'})
    return sampler


def test_sampler_buckets_classes_and_performances_ignore_duration():
    sampler = sampler_fixture()
    rng = np.random.default_rng(7)
    fast_groups, fast_classes = {}, {}
    for _ in range(12000):
        before = sampler.draws['bucket.fast']
        previous = dict(sampler.draws)
        i, window = sampler.draw(800, rng, lambda *a: slice(0, 800))
        assert 0 <= window.start < window.stop <= len(sampler.songs[i].feat)
        if sampler.draws['bucket.fast'] > before:
            group = sampler.group_for[sampler.songs[i].slug]
            fast_groups[group] = fast_groups.get(group, 0) + 1
            for cls in ('kick', 'snare', 'tom'):
                if sampler.draws[f'target.{cls}'] > previous.get(f'target.{cls}', 0):
                    fast_classes[cls] = fast_classes.get(cls, 0) + 1
    summary = sampler.summary()
    for bucket, expected in [('fast', .5), ('ordinary', .3), ('slow', .2)]:
        assert abs(summary['draws'][f'bucket.{bucket}'] / 12000 - expected) < .025
    assert abs(fast_groups['one'] / sum(fast_groups.values()) - .5) < .03
    for count in fast_classes.values():
        assert abs(count / sum(fast_classes.values()) - 1 / 3) < .03
    assert all(summary['unique_fast_seen'][c] == 6 for c in ('kick', 'snare', 'tom'))


def test_sampler_deterministic_unison_and_missing_bucket_fallback():
    meta = row('solo', 'dailychops')
    meta['events'] = {'kick': [1., 1.05], 'snare': [], 'tom': [1., 1.05]}
    song = SimpleNamespace(slug='solo', channel='dailychops', feat=np.zeros((1000, 2)))
    a = HierarchicalSampler([song], {'solo': meta}, {'solo': 'performance'})
    b = HierarchicalSampler([song], {'solo': meta}, {'solo': 'performance'})
    assert a.summary()['bucket_probabilities'] == {'fast': 1.}
    assert 'kick+tom' in a.centres[0]['fast']
    x, y = np.random.default_rng(2), np.random.default_rng(2)
    assert [a.draw(800, x, None) for _ in range(30)] == [b.draw(800, y, None) for _ in range(30)]


def audit(**kwargs):
    return {'schema': 1, 'auditor': 'fixture', 'evidence': 'independent waveform and video audit', **kwargs}


def test_three_state_unknown_has_no_gradient_and_conflicting_kd_is_masked():
    import torch
    meta = row('song', 'dailychops')
    meta['events'] = {'kick': [], 'snare': [1.], 'tom': []}
    y, m, kd = masks(meta, 1100)
    assert y[100, 1] == m[100, 1] == 1
    assert not m[:, [0, 2]].any() and not kd[:, :3].any()
    assert np.all(kd[98:103, :3] == 0) and np.all(kd[:, 3:] == 1)
    out = torch.full((1, 1100, 5), .7, requires_grad=True)
    tensors = [torch.from_numpy(a)[None] for a in (y, m, kd)]
    loss_fn(out, *tensors, torch.full_like(out, .95), .3).backward()
    assert out.grad[0, 100, 1] < 0
    assert torch.count_nonzero(out.grad[:, :, 0]) == 0
    assert torch.count_nonzero(out.grad[:, :, 2]) == 0
    assert out.grad[0, 1050, :3].abs().sum() == 0  # rejected/outside interval


def test_three_state_audit_negatives_withheld_and_multilabel():
    meta = row('song', 'dailychops')
    meta['events'] = {'kick': [1.], 'snare': [], 'tom': [1.]}
    meta['withheld_heads'] = [{'class': 'snare', 'time': 1.5}]
    a = audit(complete_intervals=[{'start': .5, 'end': 2., 'classes': ['kick', 'snare', 'tom']}])
    y, m, kd = masks(meta, 1000, a)
    assert y[100, 0] == y[100, 2] == 1
    assert m[120, :3].tolist() == [1, 1, 1]
    assert not m[147:154, 1].any() and not kd[147:154, 1].any()
    assert not kd[98:103, :3].any()
    assert not m[300, :3].any() and not kd[300, :3].any()


def test_three_state_rejects_unverified_or_conflicting_negative():
    meta = row('song', 'dailychops')
    with pytest.raises(ValueError, match='auditor and evidence'):
        masks(meta, 1000, {'schema': 1})
    with pytest.raises(ValueError, match='conflicts with written positive'):
        masks(meta, 1000, audit(verified_negative=[{'class': 'kick', 'start': 1., 'end': 1.}]))
    with pytest.raises(ValueError, match='strict accepted'):
        masks(meta, 1200, audit(verified_negative=[{'class': 'kick', 'start': 10.5, 'end': 11.}]))


def test_three_state_positive_negative_normalize_separately():
    import torch
    out = torch.tensor([[[.5, .5, .5]]])
    target = torch.tensor([[[1., 0., 0.]]])
    mask = torch.ones_like(target)
    loss = loss_fn(out, target, mask, torch.zeros_like(mask), out, 0.)
    assert float(loss) == pytest.approx(2 * np.log(2))


def bootstrap_fixture():
    counts = {g: np.tile(np.array([tp, est, ref]), (2, 3, 1)).astype(np.int64)
              for g, tp, est, ref in [('one', 40, 50, 60), ('two', 2, 10, 8), ('three', 9, 12, 20)]}
    arms = {a: {g: c.copy() for g, c in counts.items()} for a in 'ABCD'}
    arms['B']['one'][..., 0] += 3
    arms['C']['one'][..., 0] += 3
    arms['D']['one'][..., 0] += 4
    return arms


def test_bootstrap_deterministic_paired_and_repooled():
    arms = bootstrap_fixture()
    result = paired_bootstrap(arms)
    assert result == paired_bootstrap(arms)
    assert result['resamples'] == 10000
    same = result['differences']['C-B']['fast']['kick']['F']
    assert same['difference'] == 0 and same['ci95'] == [0., 0.]
    d = result['differences']['B-A']['all']['kick']['F']
    assert d['difference'] == pytest.approx(6 / (50 + 60 + 10 + 8 + 12 + 20))
    assert d['valid_resamples'] == 10000
    assert paired_bootstrap(arms, seed=42) != result


def test_bootstrap_rejects_unpaired_groups_or_references():
    arms = bootstrap_fixture()
    del arms['B']['two']
    with pytest.raises(ValueError, match='identical'):
        paired_bootstrap(arms)
    arms = bootstrap_fixture()
    arms['D']['one'][..., 2] += 1
    with pytest.raises(ValueError, match='reference counts differ'):
        paired_bootstrap(arms)


def test_bootstrap_handles_undefined_metrics_and_thresholds():
    arms = {a: {'one': np.zeros((2, 3, 3), dtype=np.int64)} for a in 'ABCD'}
    result = paired_bootstrap(arms)
    assert result['differences']['D-A']['fast']['macro_F']['ci95'] == [None, None]
    json.dumps(result, allow_nan=False)
    check = sufficiency({'kick': 499, 'snare': 1000, 'tom': 2000}, 6)
    assert not check['classes']['kick']['first_look_500']
    assert check['classes']['snare']['final_minimum_1000']
    assert not check['classes']['snare']['final_target_2000']
    assert check['classes']['tom']['final_target_2000'] and not check['group_target_met']


def test_dry_run_prints_16_jobs_without_any_work(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    monkeypatch.setattr(compare, 'create_plan', lambda *a, **kw: pytest.fail('dry-run prepared data'))
    monkeypatch.setattr(compare, 'command', lambda *a, **kw: pytest.fail('dry-run ran a subprocess'))
    monkeypatch.setattr(compare, 'require_wsl', lambda: pytest.fail('dry-run required WSL/CUDA'))
    monkeypatch.setattr(sys, 'argv', ['chops_compare', 'run', '--name', 'offline', '--dry-run'])
    compare.main()
    lines = capsys.readouterr().out.splitlines()
    jobs = [line for line in lines if line.startswith('arm=')]
    assert len(jobs) == 16
    assert not list(tmp_path.iterdir())
    assert all('steps=1500' in line and 'CUDA_VISIBLE_DEVICES=0' in line for line in jobs)
    assert {line.split()[0] for line in jobs} == {'arm=A', 'arm=B', 'arm=C', 'arm=D'}
    assert not any('v5b' in line for line in jobs)


def test_dry_run_selectable_v5b_and_three_folds(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['chops_compare', 'run', '--name', 'offline', '--dry-run',
                                    '--arms', 'v5b', '--folds', '3', '--steps', '6000', '--gpu', '1'])
    compare.main()
    jobs = [line for line in capsys.readouterr().out.splitlines() if line.startswith('arm=')]
    assert len(jobs) == 6 and all('steps=6000' in line and 'CUDA_VISIBLE_DEVICES=1' in line for line in jobs)


def test_vault_primary_unavailable_without_audit_and_repost_counts_conservative():
    from chops_vault_score import summarize
    rows = [row('one', 'dailychops', 'vid1'), row('two', 'dailychops', 'vid2')]
    plan = {'name': 'offline', 'same_performance': [['vid1', 'vid2']],
            'final_policy': {'ensemble': 'both seeds/folds'}}
    counts = {f'{s}.{c}.{key}': value for s in ('all', 'fast') for c in ('kick', 'snare', 'tom')
              for key, value in zip(('tp', 'est', 'ref'), (10, 20, 30))}
    results = {a: {r['slug']: {'complete': {}, 'partial': counts} for r in rows} for a in 'ABCD'}
    result = summarize(plan, rows, results)
    assert not result['scopes']['complete']['available']
    partial = result['scopes']['partial']
    assert partial['sufficiency']['independent_groups'] == 1
    assert partial['raw_fast_reference_moments']['kick'] == 60
    assert partial['sufficiency']['classes']['kick']['fast_moments'] == 30


@pytest.mark.parametrize('arm', ['A', 'B', 'C', 'D'])
def test_durable_wrapper_respects_frozen_training_and_installs_recipe(arm, tmp_path, monkeypatch):
    import train
    import train_recipe as runner
    import torch
    rows = [row('old', 'swartz'), row('new', 'dailychops'), row('held', 'drumshin')]
    e = compare.experiments('fixture', [arm])[0]
    plan = {'channels': ['swartz', 'dailychops', 'drumshin'], 'songs': rows, 'tags': [e], 'steps': 1500,
            'arms': {arm: {'songs': ['old', 'held'] if arm == 'A' else ['old', 'new', 'held']}},
            'folds': [{'train': ['old', 'new'], 'held': ['held'], 'channels': ['drumshin'], 'predict': ['held']}],
            'audits': {}}
    cache = tmp_path / 'ft'
    for meta in rows:
        folder = cache / meta['slug']
        folder.mkdir(parents=True)
        (folder / 'truth.json').write_text(json.dumps(meta))
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'CACHE', cache)
    monkeypatch.setattr(runner, 'load_plan', lambda path: plan)
    monkeypatch.setattr(runner, 'check_plan', lambda p: None)
    # The real vault API keeps the train-channel member of a mixed performance
    # trainable; a sealed sibling never enters the frozen training cohort.
    out = tmp_path / 'truth/harvest'
    out.mkdir(parents=True)
    monkeypatch.setattr(runner.vault, 'OUT', out)
    monkeypatch.setattr(runner.vault, 'CONFIG', out / 'vault.json')
    monkeypatch.setattr(runner.vault, 'MANIFEST', out / 'manifest.jsonl')
    (out / 'vault.json').write_text(json.dumps({'cutoff': 0., 'video_ids': ['sealed'],
                                              'train_channels': ['dailychops']}))
    (out / 'manifest.jsonl').write_text(''.join(json.dumps({
        'video': r['video_id'], 'slug': r['slug'], 'channel': r['channel'], 'at': 1. if r['slug'] == 'new' else -1.
    }) + '\n' for r in rows))
    (out / 'same_performance.json').write_text('{"groups": [["new", "sealed"]]}')
    assert runner.vault.Vault().contains_song({'video': 'sealed'})
    assert not runner.vault.Vault().contains_song(rows[1])
    assert runner.vault.Vault().final_exclusions()['sealed']['train_peers'] == ['new']
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(train, 'original_model', lambda device: SimpleNamespace(eval=lambda: None))
    original_split, original_batch, original_loss = train.split_songs, train.batch, train.loss_fn
    original_folds = runner.splits.FOLDS
    def song(folder, *a, **kw):
        meta = json.loads((folder / 'truth.json').read_text())
        return SimpleNamespace(**meta, feat=np.zeros((1000, 2), np.float32),
                               target=np.zeros((1000, 5), np.float32), truth_w=np.ones((1000, 5), np.float32),
                               kd_w=np.ones((1000, 5), np.float32))
    monkeypatch.setattr(train, 'Song', song)
    def main():
        songs = train.load_songs('cuda')
        training, held = train.split_songs(songs, '0')
        assert {s.slug for s in training} == ({'old'} if arm == 'A' else {'old', 'new'})
        assert [s.slug for s in held] == ['held']
        if arm in ('C', 'D'):
            assert train.batch is not original_batch
        if arm == 'D':
            assert train.loss_fn is loss_fn
            assert not training[0].kd_w[:, :3].any()
            assert not training[0].truth_w[500, :3].any()
        train.MODELS.mkdir()
        (train.MODELS / f'{e["tag"]}-fold0.json').write_text(json.dumps({
            'args': {}, 'train': [s.slug for s in training], 'held': ['held']}))
    monkeypatch.setattr(train, 'main', main)
    monkeypatch.setattr(sys, 'argv', ['train_recipe', '--recipe', e['recipe'], '--plan', str(plan_path),
                                    '--tag', e['tag'], '--seed', '0', '--fold', '0'])
    runner.main()
    metadata = json.loads((cache / 'models' / f'{e["tag"]}-fold0.json').read_text())
    assert metadata['arm'] == arm and metadata['recipe'] == e['recipe']
    assert train.split_songs is original_split and train.batch is original_batch and train.loss_fn is original_loss
    assert runner.splits.FOLDS == original_folds
    if arm in ('C', 'D'):
        assert metadata['sampling']['groups_per_bucket']['fast'] == 2
    if arm == 'D':
        assert metadata['training_label_counts']['kick']['negative'] == 0


def test_four_arm_final_one_opening_recorded_before_truth_and_no_training(tmp_path, monkeypatch):
    import torch
    import train
    out = tmp_path / 'truth/harvest'
    out.mkdir(parents=True)
    monkeypatch.setattr(common.vault, 'OUT', out)
    monkeypatch.setattr(common.vault, 'SONGS', out / 'songs')
    monkeypatch.setattr(common.vault, 'CONFIG', out / 'vault.json')
    monkeypatch.setattr(common.vault, 'LOG', out / 'vault_log.jsonl')
    (out / 'vault.json').write_text(json.dumps({'cutoff': 0., 'video_ids': ['sealed']}))
    tags = compare.experiments('fixture')
    plan = {'name': 'fixture', 'tags': tags, 'arms': {a: {} for a in 'ABCD'},
            'folds': [{}, {}], 'final_policy': {'ensemble': 'both seeds and folds'}}
    table, digest = common.load_performance_table(tmp_path)
    plan.update(same_performance=[list(g) for g in table], same_performance_sha256=digest)
    monkeypatch.setattr(common.vault, 'MANIFEST', out / 'manifest.jsonl')
    directory = tmp_path / 'comparison'
    directory.mkdir()
    (directory / 'plan.json').write_text(json.dumps(plan))
    weights = tmp_path / 'weights.pt'
    weights.write_bytes(b'fixture placeholder, never a real model')
    for e in tags:
        (directory / e['tag']).mkdir()
        (directory / e['tag'] / 'done.json').write_text(json.dumps({
            'plan_sha256': common.sha(directory / 'plan.json'), 'checkpoints': {'weights.pt': common.sha(weights)}}))
    folder = tmp_path / 'features'
    folder.mkdir()
    np.save(folder / 'feat.npy', np.zeros((5, 2), dtype=np.float32))
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    def read(**kw):
        records = [json.loads(line) for line in common.vault.LOG.read_text().splitlines()]
        assert len(records) == 1
        assert json.loads((directory / 'final_once.json').read_text())['opening_id'] == records[0]['opening_id']
        return [{'slug': 'sealed', 'video': 'sealed', 'channel': 'dailychops'}]
    monkeypatch.setattr(compare, 'score_rows', read)
    monkeypatch.setattr(compare, 'feature', lambda song: folder)
    monkeypatch.setattr(compare, 'checkpoint', lambda *a: weights)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(torch, 'set_num_threads', lambda n: None)
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: {})
    monkeypatch.setattr(train, 'original_model', lambda device: SimpleNamespace(load_state_dict=lambda d: None, eval=lambda: None))
    # Fixed array fixture exercises orchestration without a model forward pass.
    counter = iter(range(16))
    monkeypatch.setattr(train, 'run', lambda *a: np.full((5, 5), next(counter), dtype=np.float32))
    commands = []
    def command(argv, log, tag=None):
        assert argv[1] == 'tools/gpu/chops_vault_score.py'
        assert '--opening-id' in argv and '--model-tag' in argv
        commands.append(argv)
    monkeypatch.setattr(compare, 'command', command)
    compare.final(plan, 'fixture_final', directory, 4)
    assert len(commands) == 1
    assert len(common.vault.LOG.read_text().splitlines()) == 1
    assert len(list(folder.glob('student-*_final-fold*.npy'))) == 8
    assert np.load(folder / 'student-fixture_A_final-fold0.npy')[0, 0] == .5
    assert np.load(folder / 'student-fixture_A_final-fold1.npy')[0, 0] == 2.5
    with pytest.raises(FileExistsError):
        compare.final(plan, 'fixture_final', directory, 4)


def test_wsl_guard_refuses_mac_before_run(monkeypatch, tmp_path):
    monkeypatch.setattr(compare, 'Path', lambda path: tmp_path / 'missing-proc')
    with pytest.raises(RuntimeError, match='require WSL'):
        compare.require_wsl()


def test_final_report_formats_missing_primary_and_keeps_integer_artifacts(tmp_path):
    from chops_vault_score import summarize, write_report
    rows = [row('one', 'dailychops')]
    plan = {'name': 'offline', 'same_performance': [], 'final_policy': {'ensemble': 'both seeds/folds'}}
    counts = {f'{s}.{c}.{key}': value for s in ('all', 'fast') for c in ('kick', 'snare', 'tom')
              for key, value in zip(('tp', 'est', 'ref'), (10, 20, 30))}
    results = {a: {'one': {'complete': {}, 'partial': counts}} for a in 'ABCD'}
    output = tmp_path / 'final.json'
    write_report(summarize(plan, rows, results), output)
    data = json.loads(output.read_text())
    assert data['scopes']['partial']['arms']['A']['fast']['kick']['counts'] == [10, 20, 30]
    md = output.with_suffix('.md').read_text()
    assert 'complete: unavailable' in md and 'partial | A | fast' in md
    assert data['scopes']['partial']['bootstrap']['resamples'] == 10000


def test_excluded_results_never_enter_bootstrap_sufficiency_or_reports(tmp_path):
    from chops_vault_score import summarize, write_report
    rows = [row('kept', 'timbuell', 'clean'), row('excluded', 'timbuell', 'sealed')]
    plan = {'name': 'fixture', 'same_performance': [['sealed', 'train']],
            'final_policy': {'ensemble': 'both seeds/folds'}}
    def counts(n):
        return {f'{s}.{c}.{key}': n for s in ('all', 'fast') for c in ('kick', 'snare', 'tom')
                for key in ('tp', 'est', 'ref')}
    results = {a: {'kept': {'complete': counts(7), 'partial': {}},
                   'excluded': {'complete': counts(10000), 'partial': {}}} for a in 'ABCD'}
    excluded = {'sealed': {'video': 'sealed', 'slug': 'excluded', 'train_peers': ['train'],
                           'reason': 'excluded: train peer train'}}
    result = summarize(plan, rows, results, excluded=excluded)
    scope = result['scopes']['complete']
    assert scope['sufficiency']['independent_groups'] == 1
    assert scope['sufficiency']['classes']['kick']['fast_moments'] == 7
    assert scope['raw_fast_reference_moments']['kick'] == 7
    assert scope['arms']['A']['fast']['kick']['counts'] == [7, 7, 7]
    assert result['performance_groups'] == {'kept': 'clean'}
    assert all(set(values) == {'kept'} for values in result['per_song'].values())
    output = tmp_path / 'final.json'
    write_report(result, output)
    assert json.loads(output.read_text())['excluded'] == excluded
    assert 'sealed: excluded: train peer train' in output.with_suffix('.md').read_text()


def test_all_mixed_vault_final_writes_exclusions_without_cuda_or_truth(tmp_path, monkeypatch, capsys):
    out = tmp_path / 'truth/harvest'
    out.mkdir(parents=True)
    for name, path in [('OUT', out), ('CONFIG', out / 'vault.json'), ('MANIFEST', out / 'manifest.jsonl'),
                       ('LOG', out / 'vault_log.jsonl'), ('SONGS', out / 'songs')]:
        monkeypatch.setattr(common.vault, name, path)
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    (out / 'vault.json').write_text(json.dumps({'cutoff': 0, 'video_ids': ['sealed'],
                                              'train_channels': ['dailychops']}))
    (out / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in [
        {'video': 'train', 'slug': 'train_song', 'channel': 'dailychops', 'at': 1., 'status': 'read'},
        {'video': 'sealed', 'slug': 'sealed_song', 'channel': 'timbuell', 'at': 1., 'status': 'read'}]))
    (out / 'same_performance.json').write_text('{"groups": [["sealed", "train"]]}')
    table, digest = common.load_performance_table(tmp_path)
    plan = {'name': 'fixture', 'tags': [], 'arms': {a: {} for a in 'ABCD'},
            'same_performance': [list(g) for g in table], 'same_performance_sha256': digest,
            'final_policy': {'ensemble': 'both seeds/folds'}}
    directory = tmp_path / 'comparison'
    directory.mkdir()
    (directory / 'plan.json').write_text(json.dumps(plan))
    # Neither notation nor audio exists. A single excluded row is enough for a
    # useful empty final report, but must never trigger feature/model work.
    monkeypatch.setattr(compare, 'feature', lambda *a: pytest.fail('excluded vault prepared features'))
    monkeypatch.setattr(compare, 'command', lambda *a: pytest.fail('empty vault launched a scorer'))
    compare.final(plan, 'fixture_final', directory, 1)
    data = json.loads((directory / 'final.json').read_text())
    assert data['excluded']['sealed']['train_peers'] == ['train']
    assert all(not scope['available'] and scope['sufficiency']['independent_groups'] == 0
               for scope in data['scopes'].values())
    line = 'sealed: excluded: train peer train'
    assert line in capsys.readouterr().out
    assert line in (directory / 'final.md').read_text()
    assert line in (directory / 'final.log').read_text()
    assert len((out / 'vault_log.jsonl').read_text().splitlines()) == 1


def test_child_scorer_excludes_mixed_group_and_logs_before_scoring(tmp_path, monkeypatch, capsys):
    import chops_vault_score as scorer
    out = tmp_path / 'truth/harvest'
    out.mkdir(parents=True)
    for name, path in [('OUT', out), ('CONFIG', out / 'vault.json'), ('MANIFEST', out / 'manifest.jsonl'),
                       ('LOG', out / 'vault_log.jsonl'), ('SONGS', out / 'songs')]:
        monkeypatch.setattr(common.vault, name, path)
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    monkeypatch.setattr(scorer, 'ROOT', tmp_path)
    (out / 'vault.json').write_text(json.dumps({'cutoff': 0, 'video_ids': ['sealed', 'clean'],
                                              'train_channels': ['dailychops']}))
    rows = [{'video': v, 'slug': v + '_song', 'channel': c, 'at': 1., 'status': 'read'}
            for v, c in [('train', 'dailychops'), ('sealed', 'timbuell'), ('clean', 'timbuell')]]
    (out / 'manifest.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (out / 'same_performance.json').write_text('{"groups": [["sealed", "train"]]}')
    table, digest = common.load_performance_table(tmp_path)
    plan = {'name': 'fixture', 'channels': list(common.train_roster(['dailychops'])),
            'songs': [], 'folds': [], 'hashes': {}, 'same_performance': [list(g) for g in table],
            'same_performance_sha256': digest, 'final_policy': {'ensemble': 'both seeds/folds'}}
    directory = tmp_path / 'comparison'
    directory.mkdir()
    (directory / 'plan.json').write_text(json.dumps(plan))
    (directory / 'final_inputs.json').write_text(json.dumps({
        'slugs': ['clean_song'], 'hashes': {}, 'checkpoints': {}}))
    opening = common.vault.open_final('fixture_final')
    monkeypatch.setattr(common.vault.os, 'getppid', common.vault.os.getpid)
    class Pool:
        def __init__(self, jobs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def map(self, func, tasks):
            return map(func, tasks)
    monkeypatch.setattr(scorer, 'ProcessPoolExecutor', Pool)
    called = []
    def score_song(r, tag):
        assert r['video'] == 'clean'
        assert common.vault.LOG.exists()
        called.append(tag)
        counts = {f'{s}.{c}.{key}': 7 for s in ('all', 'fast') for c in ('kick', 'snare', 'tom')
                  for key in ('tp', 'est', 'ref')}
        return {'complete': counts, 'partial': {}}
    monkeypatch.setattr(scorer, 'score_song', score_song)
    output = directory / 'final.json'
    monkeypatch.setattr(sys, 'argv', ['chops_vault_score.py', '--plan', str(directory / 'plan.json'),
                                    '--opening-id', opening, '--model-tag', 'fixture_final',
                                    '--json', str(output), '--jobs', '1'])
    scorer.main()
    assert called == [f'fixture_{a}_final' for a in 'ABCD']
    data = json.loads(output.read_text())
    assert data['scopes']['complete']['sufficiency']['classes']['kick']['fast_moments'] == 7
    assert data['excluded']['sealed']['train_peers'] == ['train']
    assert 'sealed: excluded: train peer train' in capsys.readouterr().out
    assert 'sealed: excluded: train peer train' in output.with_suffix('.md').read_text()
    assert len(common.vault.LOG.read_text().splitlines()) == 1


def test_comparison_name_still_refuses_hyphens(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['chops_compare.py', 'run', '--name', 'bad-name', '--dry-run'])
    with pytest.raises(SystemExit):
        compare.main()
