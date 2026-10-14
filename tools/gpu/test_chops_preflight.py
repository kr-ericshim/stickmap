"""P0/P1 regressions: synthetic inputs only, no real training or vault opening."""
from collections import Counter
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
import chops_scoring as scoring
from stratified import centres, HierarchicalSampler


def moment(time, *labels):
    return time, frozenset(labels)


def assert_partition(counts):
    for cls in scoring.CLASSES:
        for subset in ('all', 'fast', 'slow'):
            tp, est, ref = [counts.get(f'{subset}.{cls}.{k}', 0) for k in ('tp', 'est', 'ref')]
            assert 0 <= tp <= est and tp <= ref
        for key in ('tp', 'est', 'ref'):
            assert counts.get(f'all.{cls}.{key}', 0) == sum(
                counts.get(f'{s}.{cls}.{key}', 0) for s in ('fast', 'slow'))


def test_fast_tom_tp_one_est_zero_regression():
    ref = [moment(0, 'kick'), moment(.05, 'tom'), moment(.15, 'snare')]
    est = [moment(.105, 'tom')]
    assert scoring.fill_seq.align(ref, est, .06) == [(1, 0)]
    # Legacy implementation reproduces the report's impossible denominator.
    legacy = scoring.fill_seq.tally(ref, est, .06)
    assert [legacy[f'fast.tom.{k}'] for k in ('tp', 'est', 'ref')] == [1, 0, 1]
    fixed = scoring.tally(ref, est, .06)
    assert [fixed[f'fast.tom.{k}'] for k in ('tp', 'est', 'ref')] == [1, 1, 1]
    assert_partition(fixed)
    for cls in scoring.CLASSES:
        for key in ('tp', 'est', 'ref'):
            assert fixed[f'all.{cls}.{key}'] == legacy[f'all.{cls}.{key}']


def test_matched_wrong_and_extra_labels_inherit_reference_subset():
    ref = [moment(0, 'kick'), moment(.05, 'tom'), moment(.15, 'snare')]
    counts = scoring.tally(ref, [moment(.105, 'tom', 'kick')], .06)
    assert counts['fast.kick.est'] == counts['fast.tom.est'] == 1
    assert counts['slow.kick.est'] == 0
    assert_partition(counts)


def test_unmatched_estimates_use_fixed_time_regions_and_earlier_tie():
    ref = [moment(0, 'kick'), moment(.0625, 'tom'), moment(.1875, 'snare')]
    est = [moment(.125, 'kick', 'tom'), moment(.14, 'snare')]
    counts = scoring.tally(ref, est, .001)
    assert counts['fast.kick.est'] == counts['fast.tom.est'] == 1
    assert counts['slow.snare.est'] == 1
    assert counts['all.stroke.tp'] == 0
    assert_partition(counts)
    empty = scoring.tally([], est, .06)
    assert empty['fast.stroke.est'] == 0 and empty['slow.stroke.est'] == 2
    assert_partition(empty)


def test_partition_invariants_for_random_multilabel_alignments():
    rng = np.random.default_rng(10)
    classes = np.array(['kick', 'snare', 'tom'])
    for _ in range(150):
        def sequence():
            times = np.sort(rng.choice(np.arange(50) / 100, size=rng.integers(0, 12), replace=False))
            return [moment(t, *rng.choice(classes, size=rng.integers(1, 4), replace=False)) for t in times]
        assert_partition(scoring.tally(sequence(), sequence(), .06))


@pytest.mark.parametrize('bad', [
    {'all.tom.tp': 1, 'all.tom.est': 0, 'all.tom.ref': 1},
    {'all.tom.tp': 2, 'all.tom.est': 2, 'all.tom.ref': 1},
    {'all.tom.est': 2, 'fast.tom.est': 1},
])
def test_invalid_counts_fail_closed(bad):
    with pytest.raises(ValueError):
        scoring.validate_counts(bad)


@pytest.mark.parametrize('scope', ['loose', 'solos', 'final_counter'])
def test_each_loose_scoring_path_uses_fixed_counter(scope, tmp_path, monkeypatch):
    ref = [moment(0, 'kick'), moment(.05, 'tom'), moment(.15, 'snare')]
    hits = [SimpleNamespace(time=.105, inst='tom')]
    monkeypatch.setattr(scoring.vault, 'SONGS', tmp_path)
    monkeypatch.setattr(scoring.vault, 'Vault', lambda: SimpleNamespace(contains_song=lambda _: False))
    monkeypatch.setattr(scoring.fill_seq, 'TRUTH', tmp_path)
    monkeypatch.setattr(scoring.fill_seq, 'notated', lambda *a: [(0, .15, ref)])
    monkeypatch.setattr(scoring.fill_seq, 'tally', lambda *a: pytest.fail('legacy broken counter called'))
    monkeypatch.setitem(sys.modules, 'fixture_detector', SimpleNamespace(detect=lambda _: hits))
    row = {'slug': 'fixture', 'video': 'fixture', 'channel': 'moyin'}
    folder = tmp_path / 'fixture'
    folder.mkdir()
    (folder / 'song.txt').write_text(str(folder))
    if scope == 'final_counter':
        # Pure scoring helper with synthetic truth; never invoke final/open_final.
        import chops_vault_score
        monkeypatch.setitem(sys.modules, 'ft_variants', SimpleNamespace(tuned=lambda _: hits))
        counts = chops_vault_score.score_song(row, 'fixture')['partial']
    else:
        counts = scoring.score_one((scope, row, 'fixture_detector:detect'))
    assert [counts[f'fast.tom.{k}'] for k in ('tp', 'est', 'ref')] == [1, 1, 1]
    assert_partition(counts)


def test_strict_path_has_slow_alias_and_partition_checks(tmp_path, monkeypatch):
    import fill_score
    monkeypatch.setattr(scoring.vault, 'Vault', lambda: SimpleNamespace(contains_song=lambda _: False))
    counts = Counter({'all.tom.tp': 1, 'all.tom.est': 3, 'all.tom.ref': 2,
                      'fill.tom.tp': 1, 'fill.tom.est': 3, 'fill.tom.ref': 2})
    monkeypatch.setattr(fill_score, 'one', lambda _: ('moyin', counts))
    result = scoring.score_one(('strict', {'slug': 'fixture'}, 'fixture_detector:detect'))
    assert result['slow.tom.est'] == 3
    assert_partition(result)


def test_score_runner_dispatches_all_fill_scopes_to_corrected_cli(tmp_path, monkeypatch):
    plan = {'channels': ['moyin'], 'solos': [{'video': 'solo'}]}
    monkeypatch.setattr(compare, 'CACHE', tmp_path)
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    monkeypatch.setattr(compare, 'check_plan', lambda _: None)
    monkeypatch.setattr(compare, 'check_predictions', lambda *a: None)
    monkeypatch.setattr(compare, 'sha', lambda _: 'fixture')
    commands = []
    def command(argv, *args):
        commands.append(argv)
        path = Path(argv[argv.index('--json') + 1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    monkeypatch.setattr(compare, 'command', command)
    compare.score_tag(plan, 'fixture', tmp_path, 1)
    fill_commands = [c for c in commands if '--scope' in c]
    assert {c[c.index('--scope') + 1] for c in fill_commands} == {'loose', 'strict', 'solos'}
    assert all(c[1] == 'tools/gpu/chops_scoring.py' for c in fill_commands)
    assert all(c[1] != 'tools/truth_video/fill_seq.py' for c in commands)


@pytest.mark.parametrize('count', [2, 3])
def test_train_roster_keeps_every_existing_fold_channel(count, monkeypatch):
    monkeypatch.setattr(common.splits, 'FOLDS', (*common.splits.FOLDS[:-1], common.splits.FOLDS[-1] + ('future_train',)))
    roster = set(common.train_roster(['dailychops']))
    folds = common.fold_policy(['dailychops'], count)
    flat = [c for f in folds for c in f]
    assert {'moyin', 'olliewoods', 'golly', 'ivannikov', 'future_train', 'dailychops'} <= roster
    assert set(flat) == roster and len(flat) == len(roster)
    if count == 3:
        assert all(set(old) <= set(new) for old, new in zip(common.splits.FOLDS, folds))


def test_nonsealed_roster_does_not_unseal_explicit_or_postcutoff_songs(tmp_path, monkeypatch):
    out = tmp_path / 'harvest'
    out.mkdir()
    for key, value in [('OUT', out), ('CONFIG', out / 'vault.json'), ('MANIFEST', out / 'manifest.jsonl'),
                       ('SONGS', out / 'songs')]:
        monkeypatch.setattr(common.vault, key, value)
    common.vault.CONFIG.write_text(json.dumps({'cutoff': 10, 'video_ids': ['explicit'], 'train_channels': ['dailychops']}))
    rows = [{'video': v, 'slug': v, 'channel': c, 'at': at, 'status': 'read'} for v, c, at in [
        ('old_moyin', 'moyin', 1), ('explicit', 'moyin', 1), ('new_moyin', 'moyin', 11),
        ('new_daily', 'dailychops', 11)]]
    common.vault.MANIFEST.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    import score
    assert {r['slug'] for r in score.songs(list(common.train_roster(['dailychops'])))} == {'old_moyin', 'new_daily'}
    # Guard precedes truth/detector work, even when a sealed row is passed directly.
    monkeypatch.setattr(scoring.fill_seq, 'notated', lambda *a: pytest.fail('sealed truth read'))
    for r in rows[1:3]:
        assert scoring.score_one(('loose', r, 'nonexistent:detector')) == Counter()


def test_d_gate_uses_training_negatives_in_every_fold_not_global_audit_presence():
    plan = {'arms': {a: {} for a in 'ABCD'}, 'audits': {'vault_or_unrelated': 'exists'},
            'folds': [{'train': ['a'], 'held': ['b']}, {'train': ['b'], 'held': ['a']}]}
    policy = common.promotion_policy(plan)
    assert policy['eligible_arms'] == ['A', 'B', 'C']
    assert policy['D']['label'] == 'D (positive-only stress test)'
    plan['audited_negative_songs'] = ['a']
    assert not common.promotion_policy(plan)['D']['promotion_eligible']
    plan['audited_negative_songs'].append('b')
    assert common.promotion_policy(plan)['D']['promotion_eligible']


def test_dry_run_prints_expanded_roster_and_d_warning_without_truth_reads(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(compare, 'ROOT', tmp_path)
    monkeypatch.setattr(compare.vault, 'Vault', lambda: SimpleNamespace(train_channels={'dailychops'}))
    monkeypatch.setattr(compare, 'create_plan', lambda *a, **k: pytest.fail('prepared data'))
    monkeypatch.setattr(compare, 'command', lambda *a, **k: pytest.fail('launched training'))
    monkeypatch.setattr(sys, 'argv', ['chops_compare', 'run', '--name', 'p0', '--dry-run'])
    compare.main()
    output = capsys.readouterr().out
    for name in ('moyin', 'olliewoods', 'golly', 'ivannikov', 'dailychops'):
        assert name in output
    assert 'positive-only stress test' in output and 'excluded from promotion' in output
    assert 'Promotion-eligible arms: A, B, C' in output
    assert len([line for line in output.splitlines() if line.startswith('arm=')]) == 16
    assert not list(tmp_path.iterdir())


def meta(events, intervals):
    return {'slug': 'fixture', 'channel': 'dailychops', 'events': events, 'intervals': intervals}


def test_fast_centres_never_bridge_rejected_intervals():
    data = meta({'kick': [1.], 'snare': [], 'tom': [1.05]}, [[0, 1.01], [1.04, 2]])
    points = centres(data, 300)
    assert points['fast'] == {}
    assert points['slow']['kick'].tolist() == [100]
    assert points['slow']['tom'].tolist() == [105]


def test_overlapping_repeated_intervals_deduplicate_fast_centres():
    data = meta({'kick': [1., 1., 1.05], 'snare': [], 'tom': [1.]}, [[0, 2], [0, 2], [.9, 1.01]])
    points = centres(data, 300)
    assert points['fast']['kick'].tolist() == [100, 105]
    assert points['fast']['tom'].tolist() == [100]
    assert not points['slow']


def test_unison_has_no_extra_target_draw_and_exposure_is_separate():
    data = meta({'kick': [1., 1.05], 'snare': [2., 2.05], 'tom': [1., 1.05]}, [[0, 3]])
    song = SimpleNamespace(slug='fixture', channel='dailychops', feat=np.zeros((300, 2)))
    sampler = HierarchicalSampler([song], {'fixture': data}, {'fixture': 'performance'})
    rng = np.random.default_rng(42)
    for _ in range(3000):
        sampler.draw(20, rng, None)
    summary = sampler.summary()
    assert summary['draws'].get('target.kick+tom', 0) == 0
    assert abs(summary['draws']['centre.kick+tom'] / 3000 - 2 / 3) < .03
    assert summary['unison_exposures']['fast.kick+tom'] == 2 * summary['draws']['centre.kick+tom']
    assert summary['draws']['cohort.expanded'] == 3000


def test_final_summary_counts_fast_groups_per_class_without_opening_vault():
    from chops_vault_score import summarize
    rows = [{'slug': 'kick', 'video': 'k'}, {'slug': 'snare', 'video': 's'}, {'slug': 'slow', 'video': 't'}]
    counts = [scoring.tally([moment(0, cls), moment(.05 if cls != 'tom' else 1, cls)], [], .06)
              for cls in ('kick', 'snare', 'tom')]
    results = {a: {r['slug']: {'complete': dict(c), 'partial': {}} for r, c in zip(rows, counts)} for a in 'ABCD'}
    result = summarize({'same_performance': [], 'arms': {a: {} for a in 'ABCD'}}, rows, results)
    sufficient = result['scopes']['complete']['sufficiency']
    assert sufficient['independent_groups'] == 3
    assert [sufficient['classes'][c]['independent_fast_groups'] for c in ('kick', 'snare', 'tom')] == [1, 1, 0]
    assert 'D' not in result['promotion']['eligible_arms']


def test_cohort_summary_deduplicates_intervals_and_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, 'CACHE', tmp_path)
    rows = []
    for slug in ('one', 'two'):
        data = meta({'kick': [1., 1.05], 'snare': [], 'tom': []}, [[0, 2], [0, 2], [.5, 1.5]])
        data['slug'] = slug
        rows.append(data)
        folder = tmp_path / slug
        folder.mkdir()
        np.save(folder / 'feat.npy', np.zeros((300, 2)))
    result = compare.cohort_summary(rows, {'one': 'same', 'two': 'same'})
    assert result['accepted_seconds'] == {'original': 0., 'expanded': 4.}
    assert result['fast_positive_frames'] == {'kick': 4, 'snare': 0, 'tom': 0}
    assert result['independent_fast_performances'] == {'kick': 1, 'snare': 0, 'tom': 0}
