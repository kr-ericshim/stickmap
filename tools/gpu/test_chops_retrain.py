"""RETRAIN fixtures: offline, no vault opening, audio, models or preparation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import chops_common as common
import chops_compare as compare
import chops_scoring as scoring


def row(slug, channel='atthedrum'):
    return {'slug': slug, 'video': slug, 'video_id': slug, 'channel': channel,
            'strict': True, 'loose': True, 'intervals': [[0, 10]]}


def video(held):
    return next(f'vid{i}' for i in range(100) if
                (int(hashlib.sha256(f'groovemap-retrain-ath:vid{i}'.encode()).hexdigest(), 16) % 5 == 0) == held)


def fixture(tmp_path):
    rows = [row('original', 'swartz'), row(video(False)), row(video(True)), row('hold'), row('vault'),
            row('batch'), row('bMUE_E21yI8'), row('hold_peer'), row('bridge_peer'), row('batch_peer')]
    base = {'songs': rows[:2], 'folds': [{'channels': ['swartz'], 'train': [video(False)]},
                                           {'channels': ['atthedrum'], 'train': ['original']}],
            'solos': [dict(row('solo', 'dev'), cache='ft/solos/solo')]}
    paths = [tmp_path / n for n in ('base.json', 'hold.json', 'batch.txt', 'dedup.json')]
    paths[0].write_text(json.dumps(base))
    paths[1].write_text(json.dumps({'holdout': ['hold']}))
    paths[2].write_text('batch\n')
    paths[3].write_text(json.dumps({'groups': [['hold', 'absent'], ['absent', 'hold_peer'],
                                                   ['batch', 'batch_peer'], ['vault', 'bridge_peer']]}))
    inputs = common.retrain_inputs(*paths)
    seal = SimpleNamespace(rows={r['video']: r for r in rows}, explicit={'vault'},
                           contains_song=lambda r: r['slug'] == 'vault')
    return rows, inputs, seal, paths


def test_frozen_roster_exclusions_absent_peers_and_hashes(tmp_path):
    rows, inputs, seal, paths = fixture(tmp_path)
    result = common.retrain_rosters(rows, inputs, inputs['dedup_groups'], seal)
    assert result['arms']['BASE']['songs'] == sorted(r['slug'] for r in rows[:2])
    assert result['held_b'] == [video(True)]
    assert set(result['excluded']) >= {'hold', 'hold_peer', 'vault', 'bridge_peer', 'batch', 'batch_peer',
                                        'bMUE_E21yI8', video(True)}
    assert set(result['held_b']).isdisjoint(result['holdout'])
    assert inputs['files']['base_plan']['sha256'] == common.sha(paths[0])
    added = row('neweligible')
    while int(hashlib.sha256(('groovemap-retrain-ath:' + added['video']).encode()).hexdigest(), 16) % 5 == 0:
        added = row(added['slug'] + 'x')
    newer = common.retrain_rosters(rows + [added], inputs, inputs['dedup_groups'], seal)
    assert newer['arms']['BASE'] == result['arms']['BASE']
    assert added['slug'] in newer['arms']['ATH']['songs']
    for arm in result['arms'].values():
        assert set(arm['songs']).isdisjoint(result['excluded'])


def test_holdout_group_removed_before_b_hash_selection(tmp_path):
    rows, inputs, seal, _ = fixture(tmp_path)
    table = inputs['dedup_groups'] + [['hold', video(True)]]
    result = common.retrain_rosters(rows, inputs, table, seal)
    assert not result['held_b']
    assert video(True) in result['holdout']


def test_b_expands_whole_performance_and_base_exclusions_win(tmp_path):
    rows, inputs, seal, _ = fixture(tmp_path)
    table = inputs['dedup_groups'] + [[video(True), video(False)]]
    result = common.retrain_rosters(rows, inputs, table, seal)
    assert set(result['held_b']) == {video(True), video(False)}
    assert result['arms']['BASE']['songs'] == ['original']


def test_missing_base_song_is_not_silently_dropped(tmp_path):
    rows, inputs, seal, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='lack eligible'):
        common.retrain_rosters(rows[1:], inputs, [], seal)


def test_inputs_required_and_malformed_dedup_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match='missing RETRAIN'):
        common.retrain_inputs(tmp_path / 'missing', None, None)
    _, _, _, paths = fixture(tmp_path)
    paths[3].write_text('{"groups": [["only"]]}')
    with pytest.raises(ValueError):
        common.retrain_inputs(*paths)


def test_folds_preserve_channels_and_predict_b_identically(tmp_path):
    rows, inputs, seal, _ = fixture(tmp_path)
    table = inputs['dedup_groups']
    config = common.retrain_rosters(rows, inputs, table, seal)
    kept = [r for r in rows if r['slug'] not in config['forbidden']]
    refs = [r for r in kept if r['channel'] != 'atthedrum' or r['slug'] in config['held_b']]
    folds = common.retrain_partitions(kept, refs, inputs, config, table)
    assert [f['channels'] for f in folds] == [f['channels'] for f in inputs['base']['folds']]
    plan = {'folds': folds, 'arms': config['arms']}
    for i, fold in enumerate(folds):
        base, ath = [common.arm_fold(plan, {'arm': a}, i) for a in ('BASE', 'ATH')]
        assert base['predict'] == ath['predict']
        assert set(base['train'] + ath['train']).isdisjoint(config['excluded'])
    assert folds[0]['predict_b'] == [video(True)]
    assert folds[1]['predict_b'] == []
    assert common.arm_fold(plan, {'arm': 'BASE'}, 1)['train'] == common.arm_fold(plan, {'arm': 'ATH'}, 1)['train']
    for k in range(len(folds)):
        assert set(common.arm_fold(plan, {'arm': 'BASE'}, k)['train']) <= set(inputs['base']['folds'][k]['train'])


@pytest.mark.parametrize('arm', ['A', 'B', 'C', 'D', 'E', 'O'])
def test_legacy_arm_policy_and_fold_unchanged(arm):
    rows = [row('old', 'swartz'), row('ath'), row('new', 'dailychops'), row('ollie', 'olliewoods_shorts')]
    expected = [r['slug'] for r in rows if (r['channel'] in common.splits.ORIGINAL_CHANNELS
                or (arm == 'O' and r['channel'] == 'olliewoods_shorts') or arm in 'BCD')]
    actual = [r['slug'] for r in rows if common.arm_includes(arm, r['channel'])]
    assert actual == expected
    fold = {'train': [r['slug'] for r in rows], 'held': ['held'], 'predict': ['ref'], 'channels': ['other']}
    plan = {'folds': [fold], 'arms': {arm: {'songs': actual}}}
    assert common.arm_fold(plan, {'arm': arm}, 0) == {**fold, 'train': actual}
    assert compare.RECIPES[arm] == {'A': 'v2', 'B': 'v2', 'C': 'hierarchical', 'D': 'three_state',
                                    'E': 'weakorder', 'O': 'v2'}[arm]


def test_complete_intervals_counts_open_hh_and_withheld_ties():
    alignment = {'intervals': [[0, 1], [2, 3]], 'systems': [
        {'system': 0, 'usable': True, 'times': [0, 1]},
        {'system': 1, 'usable': True, 'times': [2, 3]}]}
    strip = [{'system': 0, 'notes': [{'inst': 'hihat', 'exportable': True}]},
             {'system': 1, 'notes': [{'inst': 'ride', 'exportable': False}]}]
    truth = [{'time': .2, 'inst': 'hihat_open'}, {'time': .5, 'inst': 'crash'},
             {'time': .5, 'inst': 'ride'}, {'time': 2.5, 'inst': 'hihat'}, {'time': 2.6, 'inst': 'ride_bell'}]
    hits = [SimpleNamespace(time=t, inst=c) for t, c in [(.2, 'hihat'), (.5, 'cymbal'), (.8, 'cymbal'),
                                                        (2.5, 'hihat'), (2.6, 'cymbal'), (1.5, 'cymbal')]]
    assert scoring.complete_intervals(alignment, strip) == [[0, 1]]
    counts = scoring.cymbal_counts(truth, hits, alignment, strip)
    assert counts['complete.hihat'] == [1, 1, 1]
    assert counts['exported.hihat'] == [2, 2, 2]
    assert counts['complete.cymbal'] == [1, 2, 1]
    assert counts['exported.cymbal'] == [2, 3, 2]
    # The frozen hh_diag reference counts a withheld tied cymbal head as withheld too (scorer acceptance test).
    tied = [dict(strip[0], notes=strip[0]['notes'] + [{'inst': 'ride', 'exportable': False, 'tied_continuation': True}]),
            strip[1]]
    assert scoring.complete_intervals(alignment, tied) == []
    assert scoring.complete_intervals(alignment, []) == []
    alignment['systems'][1]['times'] = [.9, 3]
    assert scoring.complete_intervals(alignment, strip) == []


def test_no_complete_denominator_is_unavailable():
    raw = {a: {s: {'song': {'complete.hihat': [0, 0, 0]}} for s in (0, 1)} for a in ('BASE', 'ATH')}
    stats = scoring.retrain_statistics(raw)
    assert stats['complete.hihat']['F1']['difference'] is None
    assert stats['complete.hihat']['F1']['ci95'] == [None, None]


def stats_fixture():
    entry = {'F1': {'difference': 0., 'ci95': [-.1, .1]}, 'seed_spread': {'BASE': .1, 'ATH': .1}}
    metrics = ['all.' + c for c in ('kick', 'snare', 'tom')] + ['complete.hihat', 'complete.cymbal']
    metrics += ['fast.' + c for c in ('kick', 'snare', 'tom')]
    return {scope: {m: copy.deepcopy(entry) for m in metrics} for scope in ('A', 'B', 'dev')}


@pytest.mark.parametrize('gain, expected', [(2.999, False), (3., True), (3.001, True)])
def test_exact_three_point_boundary(gain, expected):
    scopes = stats_fixture()
    scopes['A']['complete.hihat']['F1']['difference'] = gain
    assert scoring.retrain_decision(scopes)['candidate'] is expected


@pytest.mark.parametrize('drop, ci, spread, expected', [
    (-1., [-1.5, -.5], .1, True), (-1.001, [-1.5, 0], .1, False),
    (-.2, [-.7, -.501], .1, False), (-2., [-3., 0], 1., True), (-2.001, [-3., 0], 1., False)])
def test_drop_boundaries_and_both_arm_seed_spreads(drop, ci, spread, expected):
    scopes = stats_fixture()
    scopes['A']['all.tom']['F1']['difference'] = 3.
    scopes['dev']['fast.kick'] = {'F1': {'difference': drop, 'ci95': ci},
                                 'seed_spread': {'BASE': .1, 'ATH': spread}}
    assert scoring.retrain_decision(scopes)['candidate'] is expected


def test_b_only_is_channel_fit_and_missing_guard_cannot_pass():
    scopes = stats_fixture()
    scopes['B']['complete.cymbal']['F1']['difference'] = 12.
    assert scoring.retrain_decision(scopes)['verdict'] == 'channel fit'
    scopes['A']['all.tom']['F1']['difference'] = 3.
    del scopes['A']['complete.hihat']
    result = scoring.retrain_decision(scopes)
    assert not result['candidate'] and result['verdict'] == 'inconclusive'


def test_song_bootstrap_pools_counts_then_averages_seeds():
    raw = {a: {s: {'one': {'all.kick': [80 + s, 100, 100], 'exported.hihat': [80, 100, 100]},
                       'two': {'all.kick': [8, 10, 10], 'exported.hihat': [8, 10, 10]}}
               for s in (0, 1)} for a in ('BASE', 'ATH')}
    for s in (0, 1):
        raw['ATH'][s]['one']['all.kick'][0] += 4
    result = scoring.retrain_statistics(raw)
    assert result == scoring.retrain_statistics(raw)
    metric = result['all.kick']
    assert metric['F1']['difference'] == pytest.approx(100 * 4 / 110)
    assert metric['seed_spread']['BASE'] == pytest.approx(100 / 110)
    assert metric['F1']['valid_resamples'] == 2000
    assert 'P' not in result['exported.hihat'] and 'F1' not in result['exported.hihat']
    raw['ATH'][0]['one']['all.kick'][2] += 1
    with pytest.raises(ValueError, match='reference counts differ'):
        scoring.retrain_statistics(raw)


def test_dry_run_reports_missing_caches_without_preparation(tmp_path, monkeypatch, capsys):
    rows, inputs, seal, paths = fixture(tmp_path)
    seal.train_channels = {'atthedrum'}
    monkeypatch.setattr(compare.vault, 'Vault', lambda: seal)
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    monkeypatch.setattr(compare, 'CACHE', tmp_path / 'ft')
    monkeypatch.setattr(compare, 'sha', lambda p: 'fixture')
    monkeypatch.setattr(compare, 'load_performance_table', lambda root: ([], None))
    monkeypatch.setattr(compare.vault, 'MANIFEST', tmp_path / 'manifest.jsonl')
    monkeypatch.setattr(compare.vault, 'CONFIG', tmp_path / 'vault.json')
    monkeypatch.setattr(compare.vault, 'SONGS', tmp_path / 'songs')
    monkeypatch.setattr(compare, 'score_rows', lambda channels, exclude: [r for r in rows if r['slug'] not in exclude])
    monkeypatch.setattr(compare, 'command', lambda *a, **kw: pytest.fail('launched a subprocess'))
    monkeypatch.setattr(compare, 'feature', lambda *a: pytest.fail('prepared features'))
    # Source hashes are rooted in the installed tool tree, not this synthetic root.
    monkeypatch.setattr(compare, '__file__', str(tmp_path / 'tools/gpu/chops_compare.py'))
    plan = compare.create_retrain_plan('fixture', *paths, dry_run=True)
    compare.print_retrain_cohorts(plan)
    output = capsys.readouterr().out
    assert 'BASE fold0' in output and 'ATH fold1' in output and 'Missing caches' in output
    assert plan['missing_caches']
    assert len(compare.training_jobs(plan, tmp_path / 'plan.json')) == 8
    with pytest.raises(FileNotFoundError, match='prerequisites missing'):
        compare.create_retrain_plan('fixture', *paths)


def test_legacy_kst_scorer_output_unchanged_when_adding_cymbals():
    ref = [(0., frozenset({'kick'})), (.05, frozenset({'snare'})), (.1, frozenset({'tom'}))]
    est = [(0., frozenset({'kick'})), (.05, frozenset({'snare'})), (.1, frozenset({'tom'}))]
    before = scoring.tally(ref, est, .05)
    scoring.cymbal_counts([], [], {'systems': [], 'intervals': []}, [])
    assert scoring.tally(ref, est, .05) == before


def test_retrain_final_refused_before_wsl_or_vault(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['chops_compare.py', 'final', '--name', 'fixture', '--arms', 'BASE', 'ATH'])
    monkeypatch.setattr(compare, 'require_wsl', lambda: pytest.fail('reached WSL'))
    with pytest.raises(SystemExit):
        compare.main()


def test_plan_roundtrip_checks_inputs_rosters_folds_and_cache(tmp_path, monkeypatch):
    rows, inputs, seal, paths = fixture(tmp_path)
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    monkeypatch.setattr(compare, 'CACHE', tmp_path / 'ft')
    monkeypatch.setattr(compare.vault, 'Vault', lambda: seal)
    monkeypatch.setattr(compare.vault, 'SONGS', tmp_path / 'songs')
    monkeypatch.setattr(compare.vault, 'OUT', tmp_path / 'truth/harvest')
    for attr, name in [('MANIFEST', 'manifest.jsonl'), ('CONFIG', 'vault.json')]:
        path = tmp_path / name
        path.write_text('{}')
        monkeypatch.setattr(compare.vault, attr, path)
    code = tmp_path / 'tools/gpu'
    code.mkdir(parents=True)
    for name in ('chops_common.py', 'chops_compare.py', 'chops_scoring.py'):
        (code / name).write_text('# fixture source')
    monkeypatch.setattr(compare, '__file__', str(code / 'chops_compare.py'))
    monkeypatch.setattr(compare, 'score_rows', lambda channels, exclude: [r for r in rows if r['slug'] not in exclude])
    base = inputs['base']
    base['solos'][0]['song'] = 'songs/solo'
    paths[0].write_text(json.dumps(base))
    for r in rows:
        if seal.contains_song(r):
            continue
        cache = tmp_path / 'ft' / r['slug']
        cache.mkdir(parents=True)
        (cache / 'truth.json').write_text(json.dumps(r))
        for name in ('feat.npy', 'teacher0.npy'):
            (cache / name).write_bytes(b'synthetic cache, never loaded')
        folder = tmp_path / 'songs' / r['slug'] / 'truth'
        folder.mkdir(parents=True)
        for name in ('truth.json', 'alignment.json', 'strip.json'):
            (folder / name).write_text('[]')
    cache = tmp_path / 'ft/solos/solo'
    cache.mkdir(parents=True)
    (cache / 'feat.npy').write_bytes(b'synthetic dev cache')
    folder = tmp_path / 'truth/solo'
    folder.mkdir(parents=True)
    (folder / 'song.txt').write_text(str(tmp_path / 'songs/solo'))
    for name in ('strip.json', 'alignment.json'):
        (folder / name).write_text('[]')
    plan = compare.create_retrain_plan('fixture', *paths)
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    assert common.load_plan(plan_path) == plan
    changed = copy.deepcopy(plan)
    changed['arms']['BASE']['songs'].append('hold_peer')
    with pytest.raises(ValueError, match='roster mismatch'):
        common.check_plan(changed)
    changed = copy.deepcopy(plan)
    changed['folds'][0]['train'].append(video(True))
    with pytest.raises(ValueError, match='fold mismatch'):
        common.check_plan(changed)
    paths[1].write_text('{"holdout": []}')
    with pytest.raises(ValueError, match='input changed'):
        common.check_plan(plan)


def test_held_group_with_vault_member_missing_slug_is_excluded(tmp_path):
    rows, inputs, seal, _ = fixture(tmp_path)
    seal.rows['hidden'] = {'video': 'hidden', 'channel': 'other'}
    seal.contains_song = lambda r: (r.get('video') or r.get('video_id')) in {'hidden', 'vault'}
    result = common.retrain_rosters(rows, inputs, [['hidden', video(True)]], seal)
    assert video(True) in result['forbidden'] and not result['held_b']


@pytest.mark.parametrize('primary_gain, null_gain, candidate', [(3, -40, True), (0, 40, False)])
def test_report_includes_precision_recall_ci_seed_spread_and_no_exported_precision(
        tmp_path, monkeypatch, primary_gain, null_gain, candidate):
    scopes = stats_fixture()
    keys = list(scopes['A']) + ['exported.hihat', 'exported.cymbal']
    tags = compare.experiments('fixture', ('BASE', 'ATH'))
    plan = {'tags': tags, 'folds': [{'predict_a': ['other'], 'predict_b': ['ath'], 'ath_effect': True},
                                      {'predict_a': ['null'], 'predict_b': [], 'ath_effect': False}],
            'solos': [{'slug': 'dev'}], 'retrain': {}}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    monkeypatch.setattr(compare, 'check_plan', lambda p: None)
    for e in tags:
        folder = tmp_path / e['tag']
        folder.mkdir()
        (folder / 'done.json').write_text(json.dumps({'plan_sha256': common.sha(tmp_path / 'plan.json'),
                                                     'checkpoints': {}}))
        data = {scope: {song: {m: [80 + (3 if e['arm'] == 'ATH' else 0), 100, 100] for m in keys}}
                for scope, song in [('A', 'other'), ('B', 'ath'), ('dev', 'dev')]}
        data['A']['other'] = {m: [50 + (primary_gain if e['arm'] == 'ATH' else 0), 100, 100] for m in keys}
        data['A']['null'] = {m: [50 + (null_gain if e['arm'] == 'ATH' else 0), 100, 100] for m in keys}
        (folder / 'retrain.json').write_text(json.dumps(data))
    compare.report_retrain(plan, tmp_path)
    output = json.loads((tmp_path / 'comparison.json').read_text())
    assert output['decision']['candidate'] is candidate
    assert set(output['per_song']['A']['BASE']['0']) == {'other'}
    assert set(output['per_song']['A_null']['BASE']['0']) == {'null'}
    assert 'ATH == BASE' in output['interpretation']
    assert output['bootstrap_seed'] == 20261004 and output['resamples'] == 2000
    assert output['scopes']['A']['all.kick']['P']['difference'] == pytest.approx(primary_gain)
    assert output['scopes']['A']['all.kick']['R']['difference'] == pytest.approx(primary_gain)
    assert 'P' not in output['scopes']['A']['exported.hihat']
    assert 'lower bounds' in (tmp_path / 'comparison.md').read_text()


def test_song_scoring_keeps_kst_counts_and_uses_frozen_reference_lists(tmp_path, monkeypatch):
    folder = tmp_path / 'song/truth'
    folder.mkdir(parents=True)
    truth = [{'time': .5, 'inst': 'hihat'}]
    alignment = {'intervals': [[0, 1]], 'systems': [{'system': 0, 'usable': True, 'times': [0, 1]}]}
    strip = [{'system': 0, 'notes': [{'inst': 'hihat', 'exportable': True}]}]
    for name, data in [('truth', truth), ('alignment', alignment), ('strip', strip)]:
        (folder / (name + '.json')).write_text(json.dumps(data))
    monkeypatch.setattr(scoring.vault, 'SONGS', tmp_path)
    monkeypatch.setattr(scoring.vault, 'Vault', lambda: SimpleNamespace(contains_song=lambda r: False))
    c = {f'{s}.{cls}.{key}': n for s in ('all', 'fast') for cls in ('kick', 'snare', 'tom')
         for key, n in zip(('tp', 'est', 'ref'), (2, 3, 4))}
    monkeypatch.setattr(scoring, 'score_one', lambda task: c)
    monkeypatch.setattr(scoring.importlib, 'import_module', lambda name: SimpleNamespace(
        detect=lambda song: [SimpleNamespace(time=.5, inst='hihat')]))
    result = scoring.retrain_score_song(row('song', 'swartz'), 'fixture:detect')
    assert all(result[f'{s}.{cls}'] == [2, 3, 4] for s in ('all', 'fast') for cls in ('kick', 'snare', 'tom'))
    assert result['complete.hihat'] == [1, 1, 1]


def test_dev_same_song_peer_never_trains(tmp_path, monkeypatch):
    rows, inputs, seal, _ = fixture(tmp_path)
    config = common.retrain_rosters(rows, inputs, inputs['dedup_groups'], seal)
    kept = [r for r in rows if r['slug'] not in config['forbidden']]
    refs = [r for r in kept if r['channel'] != 'atthedrum' or r['slug'] in config['held_b']]
    monkeypatch.setattr(common.splits, 'same_song', lambda a, b: a == b or (a == 'original' and b == 'solo'))
    folds = common.retrain_partitions(kept, refs, inputs, config, inputs['dedup_groups'], inputs['base']['solos'])
    assert all('original' not in f['train'] for f in folds)


def test_historical_fold_removals_never_reappear_and_reasons_are_printed(tmp_path, capsys):
    rows, inputs, seal, _ = fixture(tmp_path)
    # These global BASE songs were absent from the historical fold train list.
    rows += [row('bRC0tZ1T1-Y', 'swartz'), row('old_ath_omission')]
    inputs['base']['songs'] += rows[-2:]
    # Documented exclusions are the only allowed differences from frozen fold train.
    inputs['base']['songs'] += [row('hold', 'swartz'), row('dev_peer', 'swartz')]
    rows += [row('dev_peer', 'swartz')]
    inputs['base']['folds'][1]['train'] += ['hold', 'dev_peer']
    table = inputs['dedup_groups'] + [['dev_peer', 'solo']]
    config = common.retrain_rosters(rows, inputs, table, seal)
    kept = [r for r in rows if r['slug'] not in config['forbidden']]
    refs = [r for r in kept if r['channel'] != 'atthedrum' or r['slug'] in config['held_b']]
    folds = common.retrain_partitions(kept, refs, inputs, config, table, inputs['base']['solos'])
    plan = {'folds': folds, 'arms': config['arms'], 'retrain': {**config, 'inputs': inputs}}
    for k, frozen in enumerate(inputs['base']['folds']):
        base = common.arm_fold(plan, {'arm': 'BASE'}, k)['train']
        ath = common.arm_fold(plan, {'arm': 'ATH'}, k)['train']
        assert set(base) == set(frozen['train']) - set(folds[k]['base_removed'])
        assert 'bRC0tZ1T1-Y' not in base + ath
        assert 'old_ath_omission' not in base + ath
    assert folds[1]['base_removed']['hold'] == ['HOLDOUT (including performance groups)']
    assert 'chop-dev overlap' in folds[1]['base_removed']['dev_peer']
    compare.print_retrain_cohorts(plan)
    output = capsys.readouterr().out
    assert 'BASE fold1 removed from frozen train:' in output
    assert 'hold' in output and 'HOLDOUT' in output and 'chop-dev overlap' in output
