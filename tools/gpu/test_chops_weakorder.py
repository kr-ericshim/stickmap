"""A/E plan, stop-rule and real CPU auxiliary-forward smoke tests."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).parent
sys.path[:0] = [str(HERE), str(HERE / 'chops_recipe'), str(HERE.parent / 'finetune')]
import chops_compare as compare
import chops_common as common
import weakorder
import train
from weak_order import WeakOrderWindow


def test_dry_run_comma_arms_a_e_eight_jobs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    monkeypatch.setattr(compare, 'create_plan', lambda *a, **k: pytest.fail('data preparation'))
    monkeypatch.setattr(compare, 'command', lambda *a, **k: pytest.fail('training'))
    monkeypatch.setattr(sys, 'argv', ['compare', 'run', '--name', 'ae', '--arms', 'A,E', '--dry-run'])
    compare.main()
    output = capsys.readouterr().out
    assert len([s for s in output.splitlines() if s.startswith('arm=')]) == 8
    assert 'Promotion-eligible arms: A, E' in output
    assert 'lambda=0.05' in output and 'BOTH seeds' in output
    assert not list(tmp_path.iterdir())


def test_e_has_identical_a_strict_cohort_in_every_fold():
    plan = {'arms': {'A': {'songs': ['old', 'held']}, 'E': {'songs': ['old', 'held']}},
            'folds': [{'train': ['old', 'expanded'], 'held': ['held'], 'predict': ['held'], 'channels': ['held']}]}
    assert common.arm_fold(plan, {'arm': 'A'}, 0) == common.arm_fold(plan, {'arm': 'E'}, 0)
    assert 'E' in common.promotion_policy(plan)['eligible_arms']


def test_a_e_real_planner_freezes_sidecars_and_reruns_a(tmp_path, monkeypatch):
    import weak_set
    rows = [{'slug': s, 'video': s, 'video_id': s, 'channel': c,
             'intervals': [[0., 2.]], 'events': {'kick': [1., 1.05], 'snare': [], 'tom': []}}
            for s, c in [('old0', 'swartz'), ('old1', 'allyeodrum'), ('new', 'dailychops')]]
    cache = tmp_path / 'ft'
    for r in rows:
        folder = cache / r['slug']
        folder.mkdir(parents=True)
        (folder / 'truth.json').write_text(json.dumps(r))
        np.save(folder / 'feat.npy', np.zeros((300, 2)))
    weak_row = {'slug': 'weak-only', 'video': 'weak', 'channel': 'future_weak', 'seconds': 10.}
    pointer = tmp_path / 'truth/external/song.txt'
    pointer.parent.mkdir(parents=True)
    pointer.write_text('/other/host/solo-AAAAAAAAAAA')
    seal = SimpleNamespace(train_channels={'dailychops'}, explicit=set(),
                           rows={r['video']: r for r in rows + [weak_row]}, contains_song=lambda _: False)
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    monkeypatch.setattr(compare, 'CACHE', cache)
    monkeypatch.setattr(compare.vault, 'SONGS', tmp_path / 'truth/harvest/songs')
    monkeypatch.setattr(compare.vault, 'OUT', tmp_path / 'truth/harvest')
    monkeypatch.setattr(compare.vault, 'Vault', lambda: seal)
    commands = []
    monkeypatch.setattr(compare, 'command', lambda argv, *a: commands.append(argv))
    monkeypatch.setattr(common, 'SAME_PERFORMANCE', (('new', 'AAAAAAAAAAA'),))
    monkeypatch.setattr(compare, 'sha', lambda _: 'frozen-fixture')
    monkeypatch.setattr(compare, 'feature', lambda song: cache / song.name)
    monkeypatch.setattr(compare, 'score_rows', lambda _, **k: [{**r, 'strict': True, 'loose': True} for r in rows])
    monkeypatch.setattr(compare, 'solo_inputs', lambda: [{'slug': 'solo', 'video': 'solo', 'cache': 'solo'}])
    calls = []
    def build(plan, fold, **kwargs):
        calls.append(fold)
        return {'quality_fold': fold, 'songs': {'weak-only': []}, 'counts': {}, 'source_hashes': {},
                'skipped': {'missing-rhythm-review': 1}, 'exclusions': {}}
    monkeypatch.setattr(weak_set, 'build_fold', build)
    monkeypatch.setattr(weak_set, 'print_counts', lambda _: None)
    directory = tmp_path / 'scores/chops/ae'
    plan = compare.create_plan('ae', directory, arms=['A', 'E'])
    assert calls == [0, 1]
    assert plan['arms']['E']['songs'] == plan['arms']['A']['songs'] == ['old0', 'old1']
    assert plan['baseline_policy']['reuse'] is False
    assert plan['external_solo_exclusions']['new']['solo_sources'] == ['external']
    assert 'new' not in {r['slug'] for r in plan['references'] + plan['songs']}
    assert commands[0][1] == 'tools/finetune/weak_set.py' and '--exclude' in commands[0]
    assert 'new' in commands[0]
    assert plan['weak_order']['lambda'] == .05 and plan['weak_order']['batch_size'] == 1
    assert 'future_weak' in plan['channels']
    assert len(compare.training_jobs(plan, directory / 'plan.json')) == 8
    assert len(plan['weak_order']['sidecars']) == 2
    for relative in plan['weak_order']['sidecars']:
        assert relative in plan['hashes'] and (tmp_path / relative).exists()
    for fold in range(2):
        assert common.arm_fold(plan, {'arm': 'A'}, fold) == common.arm_fold(plan, {'arm': 'E'}, fold)


def test_empty_weak_preflight_stops_before_training(monkeypatch):
    import weak_set
    monkeypatch.setattr(weak_set, 'load_fold', lambda *a, **k: {'songs': {'song': []}, 'skipped': {'missing-rhythm-review': 3}})
    with pytest.raises(ValueError, match='zero accepted weak windows'):
        compare.weak_preflight({'arms': {'A': {}, 'E': {}}, 'weak_order': {'sidecars': ['synthetic']}})


def test_channel_score_jobs_receive_frozen_external_peer_exclusions(tmp_path, monkeypatch):
    plan = {'channels': ['moyin'], 'solos': [{'video': 'solo'}],
            'external_solo_exclusions': {'excluded-peer': {}}}
    monkeypatch.setattr(compare, 'CACHE', tmp_path)
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    monkeypatch.setattr(compare, 'check_plan', lambda _: None)
    monkeypatch.setattr(compare, 'check_predictions', lambda *a: None)
    monkeypatch.setattr(compare, 'sha', lambda _: 'fixture')
    commands = []
    def command(argv, *a):
        commands.append(argv)
        path = Path(argv[argv.index('--json') + 1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    monkeypatch.setattr(compare, 'command', command)
    compare.score_tag(plan, 'fixture', tmp_path, 1)
    assert len(commands) == 4
    assert all('--exclude' in c and c[c.index('--exclude') + 1] == 'excluded-peer' for c in commands)


def test_global_wrapper_excludes_peer_before_tally(tmp_path, monkeypatch):
    import global_score
    namespace = {'FAMILIES': ('kick', 'snare', 'tom', 'hihat', 'cymbal'),
                 'songs': lambda *a, **k: [{'slug': 'excluded-peer'}, {'slug': 'kept'}],
                 'tally': lambda *a: {c: [1, 1, 1] for c in ('kick', 'snare', 'tom', 'hihat', 'cymbal')}}
    exec('def main(**kwargs):\n    for row in songs(None):\n        assert row["slug"] == "kept"\n        tally(row)\n', namespace)
    monkeypatch.setattr(global_score.runpy, 'run_path', lambda _: namespace)
    monkeypatch.setitem(sys.modules, 'weak_fixture_detector', SimpleNamespace(detect=lambda _: None))
    monkeypatch.setattr(global_score.evaluate, 'detect', global_score.evaluate.detect)
    output = tmp_path / 'global.json'
    monkeypatch.setattr(sys, 'argv', ['global_score', '--detect', 'weak_fixture_detector:detect',
                                    '--exclude', 'excluded-peer', '--json', str(output)])
    global_score.main()
    counts = json.loads(output.read_text())
    assert counts['songs'] == 1 and counts['kick'] == [1, 1, 1]


@pytest.mark.parametrize('scope', ['strict', 'loose'])
def test_routine_scorer_excludes_peer_before_worker_reads(scope, tmp_path, monkeypatch):
    from collections import Counter
    import chops_scoring
    import score
    monkeypatch.setattr(score, 'songs', lambda *a, **k: [{'slug': 'excluded-peer'}, {'slug': 'kept'}])
    class Pool:
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def map(self, function, tasks):
            tasks = list(tasks)
            assert [row['slug'] for _, row, _ in tasks] == ['kept']
            return [Counter()]
    monkeypatch.setattr(chops_scoring, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(sys, 'argv', ['scoring', '--scope', scope, '--exclude', 'excluded-peer',
                                    '--json', str(tmp_path / 'counts.json')])
    chops_scoring.main()


def test_strict_preparation_excludes_peer_before_data_worker(tmp_path, monkeypatch):
    import weak_set
    import data
    import concurrent.futures
    monkeypatch.setattr(weak_set.vault, 'Vault', lambda: SimpleNamespace(contains_song=lambda _: False))
    monkeypatch.setattr(data, 'OUT', tmp_path / 'ft')
    monkeypatch.setattr(data, 'songs', lambda *a: [{'slug': 'excluded-peer'}, {'slug': 'kept'}])
    class Pool:
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def map(self, function, rows):
            assert [r['slug'] for r in rows] == ['kept']
            return ['kept']
    monkeypatch.setattr(concurrent.futures, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(sys, 'argv', ['weak_set', '--prepare-strict', '--exclude', 'excluded-peer'])
    weak_set.main()


def test_e_final_rejected_before_any_vault_action(monkeypatch, tmp_path):
    monkeypatch.setattr(compare.vault, 'open_final', lambda *a: pytest.fail('vault opened'))
    with pytest.raises(ValueError, match='forbids vault'):
        compare.final({'arms': {'A': {}, 'E': {}}}, 'tag', tmp_path, 1)


def test_cpu_weak_kd_and_order_gradient_masks():
    torch.set_num_threads(1)
    logits = torch.zeros((1, 30, 5), requires_grad=True)
    outputs = logits.sigmoid()
    w = WeakOrderWindow(0, 5, 25, (1, 2, 4, 4))
    loss, order, replay = weakorder.weak_objective(outputs, torch.full_like(outputs, .2), [w], train.loss_fn)
    assert torch.isfinite(loss) and order > 0 and replay > 0
    replay_grad, = torch.autograd.grad(replay, logits, retain_graph=True)
    assert replay_grad[0, 5:25, :3].count_nonzero() == 0
    assert replay_grad[0, 5:25, 3:].abs().sum() > 0
    order_grad, = torch.autograd.grad(order, logits, retain_graph=True)
    assert order_grad[0, :5].count_nonzero() == 0 and order_grad[0, 25:].count_nonzero() == 0
    assert order_grad[..., 3:].count_nonzero() == 0
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0


def test_tiny_cpu_recipe_forward_backward_preserves_strict_rng(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    folder = tmp_path / 'weak'
    folder.mkdir()
    np.save(folder / 'feat.npy', np.ones((40, 2), np.float32) * .1)
    np.save(folder / 'teacher0.npy', np.ones((40, 5), np.float32) * .2)
    monkeypatch.setattr(common.vault, 'Vault', lambda: SimpleNamespace(contains_song=lambda _: False))
    e = {'slug': 'weak', 'feature_frames': 40, 'window_start_frame': 5, 'window_stop_frame': 25,
         'tokens': [1, 2, 4, 4], 'confidence': 1., 'weight': 1.}
    stub = SimpleNamespace(CACHE=tmp_path, augment=train.augment)
    aux = weakorder.Auxiliary(stub, {'songs': {'weak': [e]}}, 0, .05, 'cpu', None)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 5)
            self.dropout = torch.nn.Dropout(.2)
        def forward(self, x):
            return self.linear(self.dropout(x[..., 0])).sigmoid()
    model = Model()
    aux.model, aux.base_loss = model, train.loss_fn
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = torch.ones((1, 40, 2, 1))
    args = [torch.zeros((1, 40, 5)), torch.ones((1, 40, 5)), torch.ones((1, 40, 5)),
            torch.full((1, 40, 5), .2), .3, (), 0., 'peak', 1.]
    strict_rng = np.random.default_rng(0)
    for _ in range(2):
        before_numpy = copy.deepcopy(strict_rng.bit_generator.state)
        strict_out = model(x)
        before_torch = torch.random.get_rng_state().clone()
        loss = aux.loss(strict_out, *args)
        assert torch.equal(torch.random.get_rng_state(), before_torch)
        assert strict_rng.bit_generator.state == before_numpy
        optimizer.zero_grad()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert aux.summary()['steps'] == 2 and aux.summary()['counts']['windows'] == 2


def test_e_wrapper_installs_auxiliary_and_preserves_v2_initialization(tmp_path, monkeypatch):
    import train_recipe as runner
    import weak_set
    torch.set_num_threads(1)
    rows = [{'slug': s, 'video_id': s, 'channel': c} for s, c in
            [('old', 'swartz'), ('new', 'dailychops'), ('held', 'drumshin')]]
    cache = tmp_path / 'ft'
    for r in rows:
        folder = cache / r['slug']
        folder.mkdir(parents=True)
        (folder / 'truth.json').write_text(json.dumps(r))
    folder = cache / 'weak'
    folder.mkdir()
    np.save(folder / 'feat.npy', np.ones((40, 2), np.float32) * .1)
    np.save(folder / 'teacher0.npy', np.ones((40, 5), np.float32) * .2)
    experiment = compare.experiments('fixture', ['E'])[0]
    plan = {'channels': ['swartz', 'dailychops', 'drumshin'], 'songs': rows, 'tags': [experiment], 'steps': 1,
            'arms': {'A': {'songs': ['old', 'held']}, 'E': {'songs': ['old', 'held']}},
            'folds': [{'channels': ['drumshin'], 'train': ['old', 'new'], 'held': ['held'], 'predict': ['held']}],
            'weak_order': {'lambda': .05, 'sidecars': ['weak.json']}, 'hashes': {'weak.json': 'synthetic'}}
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'CACHE', cache)
    monkeypatch.setattr(train, 'CACHE', cache)
    monkeypatch.setattr(runner, 'load_plan', lambda _: plan)
    monkeypatch.setattr(runner, 'check_plan', lambda _: None)
    monkeypatch.setattr(runner.vault, 'Vault', lambda: SimpleNamespace(contains_song=lambda _: False))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    entry = {'slug': 'weak', 'feature_frames': 40, 'window_start_frame': 5, 'window_stop_frame': 25,
             'tokens': [1, 2, 4, 4], 'confidence': 1., 'weight': 1.}
    monkeypatch.setattr(weakorder, 'load_fold', lambda *a, **k: {'songs': {'weak': [entry]}})
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 5)
        def forward(self, x):
            return self.linear(x[..., 0]).sigmoid()
    monkeypatch.setattr(train, 'original_model', lambda device: Model())
    monkeypatch.setattr(train, 'Song', lambda folder, *a, **k: SimpleNamespace(
        **json.loads((folder / 'truth.json').read_text())))
    original_batch, original_loss, original_model = train.batch, train.loss_fn, train.original_model
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        Model()  # v2's safe_load teacher
        expected = Model().state_dict()
    def main():
        torch.manual_seed(0)
        songs = train.load_songs('cpu', dense=False)
        training, held = train.split_songs(songs, '0')
        assert [s.slug for s in training] == ['old'] and [s.slug for s in held] == ['held']
        assert train.batch is original_batch
        model = train.original_model('cpu')
        for key, value in expected.items():
            torch.testing.assert_close(model.state_dict()[key], value)
        out = model(torch.ones((1, 40, 2, 1)))
        loss = train.loss_fn(out, torch.zeros_like(out), torch.ones_like(out), torch.ones_like(out),
                             torch.full_like(out, .2), .3, (), 0., 'peak', 1.)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        train.MODELS.mkdir()
        (train.MODELS / f'{experiment["tag"]}-fold0.json').write_text(json.dumps({'args': {}, 'held': ['held'], 'train': ['old']}))
    monkeypatch.setattr(train, 'main', main)
    monkeypatch.setattr(sys, 'argv', ['recipe', '--recipe', 'weakorder', '--plan', str(tmp_path / 'plan.json'),
                                    '--tag', experiment['tag'], '--fold', '0', '--seed', '0'])
    runner.main()
    metadata = json.loads((cache / 'models' / f'{experiment["tag"]}-fold0.json').read_text())
    assert metadata['weak_order']['steps'] == 1 and metadata['weak_order']['lambda'] == .05
    assert metadata['weak_order']['sidecar_sha256'] == 'synthetic'
    assert train.original_model is original_model and train.loss_fn is original_loss and train.batch is original_batch


def metric_rows(gains=(.03, .03), guard=0.):
    rows = []
    for seed in (0, 1):
        for arm in ('A', 'E'):
            for metric, subsets in (('solos', ('fast',)), ('strict', ('fast', 'slow', 'all')), ('global', ('all',))):
                for subset in subsets:
                    f = .7 + (gains[seed] if metric == 'solos' else guard) * (arm == 'E')
                    classes = ('kick', 'snare', 'tom', 'hihat', 'cymbal') if metric == 'global' else compare.METRICS
                    rows.append({'arm': arm, 'seed': seed, 'metric': metric, 'subset': subset,
                                 'prf': {c: [f, f, f] for c in classes},
                                 'counts': {c: [70, 100, 100] for c in classes}, 'all_song_prf': [f, f, f]})
    return rows


@pytest.mark.parametrize('gains,guard,expected', [((.02, .02), -.005, True), ((.03, .019), 0., False),
                                               ((.03, .03), -.006, False)])
def test_two_seed_macro_and_regression_stop_rule(gains, guard, expected):
    assert compare.weak_stop_rule(metric_rows(gains, guard))['promising'] is expected


def test_missing_metrics_are_inconclusive_and_reference_changes_raise():
    rows = metric_rows()
    rows[0]['prf']['tom'][2] = None
    assert not compare.weak_stop_rule(rows)['promising']
    rows = metric_rows()
    rows[0]['counts']['tom'][2] = 101
    with pytest.raises(ValueError, match='cohorts differ'):
        compare.weak_stop_rule(rows)


def test_report_contains_e_primary_and_both_seed_verdict(tmp_path):
    tags = compare.experiments('fixture', ['A', 'E'])
    plan = {'name': 'fixture', 'arms': {'A': {}, 'E': {}}, 'tags': tags,
            'songs': [], 'references': [], 'solos': [], 'channels': []}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    for e in tags:
        directory = tmp_path / e['tag']
        directory.mkdir()
        (directory / 'done.json').write_text(json.dumps({'plan_sha256': compare.sha(tmp_path / 'plan.json')}))
        for metric in ('loose', 'strict', 'solos'):
            counts = {}
            tp = 73 if metric == 'solos' and e['arm'] == 'E' else 70
            for cls in ('kick', 'snare', 'tom'):
                for speed in ('fast', 'slow', 'all'):
                    for key, value in [('tp', tp), ('est', 100), ('ref', 100)]:
                        counts[f'{speed}.{cls}.{key}'] = value * (2 if speed == 'all' else 1)
            (directory / f'{metric}.json').write_text(json.dumps(counts))
        (directory / 'global.json').write_text(json.dumps({c: [70, 100, 100] for c in
                                                         ('kick', 'snare', 'tom', 'hihat', 'cymbal')}))
    compare.report(plan, tmp_path)
    output = json.loads((tmp_path / 'comparison.json').read_text())
    assert output['weak_order_result']['promising']
    assert len(output['weak_order_result']['seeds']) == 2
    assert 'Global HH/C' in (tmp_path / 'comparison.md').read_text()
