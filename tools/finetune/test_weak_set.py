"""Synthetic sidecar isolation/admission tests; no real songs or vault opening."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weak_set as weak


@pytest.fixture
def harvest(tmp_path, monkeypatch):
    out = tmp_path / 'truth/harvest'
    out.mkdir(parents=True)
    for key, value in [('OUT', out), ('CONFIG', out / 'vault.json'), ('MANIFEST', out / 'manifest.jsonl'),
                       ('SONGS', out / 'songs')]:
        monkeypatch.setattr(weak.vault, key, value)
    weak.vault.CONFIG.write_text(json.dumps({'cutoff': 10., 'video_ids': [], 'train_channels': ['train']}))
    (out / 'same_performance.json').write_text('{"groups": []}')
    pointer = tmp_path / 'truth/external/song.txt'
    pointer.parent.mkdir()
    pointer.write_text('/some/other/host/solo-AAAAAAAAAAA')
    rows = [{'slug': 'song-BBBBBBBBBBB', 'video': 'BBBBBBBBBBB', 'channel': 'train', 'at': 1., 'seconds': 10.},
            {'slug': 'held-CCCCCCCCCCC', 'video': 'CCCCCCCCCCC', 'channel': 'held', 'at': 1., 'seconds': 10.}]
    weak.vault.MANIFEST.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    plan = {'channels': ['train', 'held'], 'songs': [], 'references': [],
            'folds': [{'channels': ['held'], 'train': [rows[0]['slug']],
                       'held': [rows[1]['slug']], 'predict': [rows[1]['slug']]}]}
    system = {'system': 0, 'interval': [3., 5.], 'read_x': [0, 150],
              'notated_beats': 8, 'cursor_x': [i * 10. for i in range(12)],
              'cursor_t': [3 + i / 8 for i in range(12)], 'cursor_coverage': 1.,
              'staff': {'reasons': [], 'clef_confirmed': True, 'spacing': 10.},
              'notes': [{'x': i * 10., 'inst': ('kick', 'snare', 'tom_high')[i % 3],
                         'pos': i % 3, 'verified': True, 'exportable': True} for i in range(12)]}
    alignment = {'system': 0, 'interval': [3., 5.], 'usable': False, 'strict_usable': False,
                 'scoring_usable': True, 'tier': 'scoring', 'scoring_reasons': [],
                 'reasons': ['instrument-band agreement low'], 'mode': 'cursor',
                 'after': {'within_40ms': .95}, 'held_out': {'within_40ms': .9},
                 'held_out_gain_40ms': .2, 'strictly_monotone': True,
                 'times': [3 + i / 8 for i in range(12)], 'moment_groups': [[i] for i in range(12)]}
    review = dict.fromkeys(('staff_clean', 'clef_verified', 'legend_verified', 'visited_span_verified',
                           'reader_complete', 'heads_complete', 'duplicates_removed', 'rhythm_verified',
                           'ties_verified', 'rests_verified', 'grace_roll_verified', 'system_location_verified',
                           'complete_window_sequence', 'padding_sequence_verified'), True)
    review.update(quality_fold=0, reference_scope='training-only', unresolved_rhythm=0, unresolved_identity=0,
                  reviewer='synthetic reviewer', evidence='synthetic native notation/span audit',
                  source_revision='fixture1', source_span_seconds=[3., 5.], audio_offset_seconds=0.,
                  feature_origin_seconds=0., diagnostic_tempo_bpm=240,
                  moments=[{'onset': f'{i}/8', 'duration': '1/8', 'chord_id': f'c{i}',
                            'tuplet_ratio': '1/1', 'tie_chain': [], 'grace': False, 'roll': False,
                            'repeat_count': 1, 'head_indices': [i]} for i in range(12)])
    truth = weak.vault.SONGS / rows[0]['slug'] / 'truth'
    truth.mkdir(parents=True)
    np.save(truth.parent / 'attacks.htdemucs.npy', np.array([3 + i / 8 for i in range(12)]))
    folder = tmp_path / '.eval-cache/ft' / rows[0]['slug']
    folder.mkdir(parents=True)
    np.save(folder / 'feat.npy', np.zeros((1000, 2), np.float16))
    strict = {'systems': [copy.deepcopy(alignment)], 'intervals': []}

    def write():
        for name, data in [('strip.json', [system]), ('alignment.json', strict),
                           ('alignment.scoring.json', {'systems': [alignment]})]:
            (truth / name).write_text(json.dumps(data))
        (truth / weak.REVIEW).write_text(json.dumps({
            'schema_version': 1, 'source_hashes': {n: weak.digest(truth / n) for n in
                                                  ('strip.json', 'alignment.json', 'alignment.scoring.json')},
            'systems': {'0': review}}))
    write()
    return tmp_path, plan, rows, system, alignment, strict, review, truth, write


def build(harvest, *, manual_review=True):
    root, plan, *_ = harvest
    return weak.build_fold(plan, 0, root=root, manual_review=manual_review)


def test_reviewed_sequence_full_window_and_no_frame_labels(harvest):
    data = build(harvest)
    slug = harvest[2][0]['slug']
    e, = data['songs'][slug]
    assert e['tokens'] == [1, 2, 4] * 4
    assert e['window_start_frame'] == 235 and e['window_stop_frame'] == 565
    assert e['band_check'] == 'off'
    assert data['counts']['train']['fast_kst'] == dict.fromkeys(weak.FAMILIES, 4)
    assert data['songs'][harvest[2][1]['slug']] == []
    assert 'events' not in e and 'times' not in e and 'intervals' not in e
    assert all('time' not in m for m in e['moments'])
    path = harvest[0] / 'sidecar.json'
    path.write_text(json.dumps(data))
    assert weak.load_fold(path, harvest[1], 0, root=harvest[0]) == data


@pytest.mark.parametrize('kind', ['vault', 'held_channel', 'same_song', 'same_performance', 'external'])
def test_exclusion_happens_before_any_notation_or_feature_read(harvest, kind):
    root, plan, rows, *_, truth, write = harvest
    if kind == 'vault':
        weak.vault.CONFIG.write_text(json.dumps({'cutoff': 10., 'video_ids': [rows[0]['video']]}))
    elif kind == 'held_channel':
        plan['folds'][0]['channels'].append('train')
    elif kind == 'same_song':
        rows[0]['slug'] = 'last-christmas-BBBBBBBBBBB'
        rows[1]['slug'] = 'last-christmas-CCCCCCCCCCC'
        plan['folds'][0]['held'] = [rows[1]['slug']]
        plan['folds'][0]['predict'] = [rows[1]['slug']]
        weak.vault.MANIFEST.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    else:
        # External exclusion must follow transitive groups, including absent bridge IDs.
        groups = [[rows[0]['video'], 'DDDDDDDDDDD'], ['DDDDDDDDDDD',
                  'AAAAAAAAAAA' if kind == 'external' else rows[1]['video']]]
        (weak.vault.OUT / 'same_performance.json').write_text(json.dumps({'groups': groups}))
    (truth / 'strip.json').write_text('invalid JSON: must never be read')
    data = build(harvest, manual_review=False)
    assert data['songs'][rows[0]['slug']] == []
    assert rows[0]['slug'] in data['exclusions']
    if kind == 'external':
        assert data['exclusions'][rows[0]['slug']]['solo_sources'] == ['external']


def test_seal_failure_precedes_sidecar_and_notation_reads(harvest):
    weak.vault.CONFIG.write_text('invalid')
    with pytest.raises(ValueError):
        build(harvest)
    with pytest.raises(ValueError):
        weak.load_fold(Path('/nonexistent/sidecar'), harvest[1], 0, root=harvest[0])


def test_seal_rechecked_before_load(harvest):
    data = build(harvest)
    path = harvest[0] / 'sidecar.json'
    path.write_text(json.dumps(data))
    weak.vault.CONFIG.write_text(json.dumps({'cutoff': 10., 'video_ids': ['BBBBBBBBBBB']}))
    (harvest[7] / 'strip.json').write_text('invalid: now sealed')
    with pytest.raises(ValueError, match='seal changed'):
        weak.load_fold(path, harvest[1], 0, root=harvest[0])


@pytest.mark.parametrize('fault', ['empty', 'missing_review', 'rhythm', 'scan_span', 'unknown',
                                  'withheld', 'timing', 'short', 'partial_window', 'wrong_fold',
                                  'missing_head', 'reused_head', 'grace', 'strict_overlap', 'feature_boundary'])
def test_incomplete_or_unverified_systems_are_not_targets(harvest, fault):
    root, plan, rows, system, alignment, strict, review, truth, write = harvest
    if fault == 'empty':
        for head in system['notes']:
            head['inst'] = 'hihat'
    elif fault == 'missing_review':
        (truth / weak.REVIEW).unlink()
    elif fault == 'rhythm':
        review['rhythm_verified'] = False
    elif fault == 'scan_span':
        alignment['scoring_reasons'] = ['unverified scan span']
    elif fault == 'unknown':
        system['notes'][0]['inst'] = 'unknown'
    elif fault == 'withheld':
        system['notes'][0]['exportable'] = False
    elif fault == 'timing':
        alignment['held_out']['within_40ms'] = .7
    elif fault == 'short':
        system['notes'] = system['notes'][:6]
        alignment['moment_groups'] = alignment['moment_groups'][:6]
        review['moments'] = review['moments'][:6]
    elif fault == 'partial_window':
        review['complete_window_sequence'] = False
    elif fault == 'wrong_fold':
        review['quality_fold'] = 1
    elif fault == 'missing_head':
        review['moments'].pop()
    elif fault == 'reused_head':
        review['moments'][-1]['head_indices'] = [0]
    elif fault == 'grace':
        review['moments'][0]['grace'] = True
    elif fault == 'strict_overlap':
        # Guard catches the rounded boundary just outside the padded window.
        strict['intervals'] = [[5.65, 6.]]
    elif fault == 'feature_boundary':
        np.save(root / '.eval-cache/ft' / rows[0]['slug'] / 'feat.npy', np.zeros((500, 2)))
    if fault != 'missing_review':
        write()
    data = build(harvest)
    assert data['songs'][rows[0]['slug']] == []
    assert data['counts']['train']['systems'] == 0
    assert data['skipped']


def test_stale_review_hash_and_sidecar_tampering_fail_closed(harvest):
    data = build(harvest)
    path = harvest[0] / 'sidecar.json'
    data['songs'][harvest[2][0]['slug']][0]['tokens'][0] = 4
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='differs'):
        weak.load_fold(path, harvest[1], 0, root=harvest[0])
    (harvest[7] / 'strip.json').write_text('[]')
    with pytest.raises(ValueError, match='stale'):
        build(harvest)


def test_automatic_is_default_and_ignores_manual_rhythm_review(harvest):
    root, plan, rows, system, alignment, strict, review, truth, write = harvest
    alignment['scoring_reasons'] = ['unresolved rhythm durations', 'tuplet ratio uncertain', 'unknown duration']
    write()
    (truth / weak.REVIEW).write_text('invalid old manual review is unused')
    data = weak.build_fold(plan, 0, root=root)
    entry, = data['songs'][rows[0]['slug']]
    assert entry['tokens'] == [1, 2, 4] * 4
    assert data['acceptance_mode'] == 'automatic'
    assert not any(p.endswith(weak.REVIEW) for p in data['source_hashes'])
    assert entry['acceptance_evidence']['acoustic']['core_assignment_fraction'] == 1.
    assert entry['acceptance_evidence']['span']['tempo_bpm'] == 240
    assert 'onset' not in entry['moments'][0] and 'duration' not in entry['moments'][0]
    path = root / 'auto.json'
    path.write_text(json.dumps(data))
    assert weak.load_fold(path, plan, 0, root=root) == data
    # Acoustic source changes invalidate even a numerically equivalent sidecar.
    np.save(truth.parent / 'attacks.htdemucs.npy', np.r_[3 + np.arange(12) / 8, 9.])
    with pytest.raises(ValueError, match='differs'):
        weak.load_fold(path, plan, 0, root=root)


def test_ties_removed_from_training_tokens_even_with_stale_alignment_groups(harvest):
    root, plan, rows, system, alignment, strict, review, truth, write = harvest
    system['notes'][2].update(tied_continuation=True, exportable=False, verified=False, inst='unknown')
    write()
    data = build(harvest, manual_review=False)
    entry, = data['songs'][rows[0]['slug']]
    assert entry['tokens'] == [1, 2] + [1, 2, 4] * 3
    assert all(2 not in m['head_indices'] for m in entry['moments'])
    assert entry['acceptance_evidence']['tied_continuations_removed'] == 1


@pytest.mark.parametrize('n,accepted', [(7, False), (8, True)])
def test_automatic_minimum_eight_moments(harvest, n, accepted):
    root, plan, rows, system, alignment, strict, review, truth, write = harvest
    system['notes'] = system['notes'][:n]
    np.save(truth.parent / 'attacks.htdemucs.npy', np.array(system['cursor_t'][:n]))
    write()
    assert bool(build(harvest, manual_review=False)['songs'][rows[0]['slug']]) is accepted


@pytest.mark.parametrize('fault,reason', [
    ('missing_attacks', 'missing-drum-stem-attacks'), ('deficit', 'acoustic-onset-count'),
    ('excess', 'acoustic-onset-count'), ('bad_order', 'acoustic-order-assignment'),
    ('tempo_low', 'implausible-span-tempo'), ('tempo_high', 'implausible-span-tempo'),
    ('no_beats', 'missing-notated-beats/barlines'), ('cursor', 'missing/invalid-cursor-span'),
    ('strict_source', 'strict-overlap'), ('strict_fitted', 'strict-overlap'),
    ('strict_rounded', 'strict-overlap'), ('features', 'missing-features'),
    ('feature_boundary', 'feature-boundary'), ('unknown', 'unknown/unverified-head'),
    ('withheld', 'withheld-head'), ('duplicates', 'reader/head/span-reason'),
    ('roll', 'unexpanded-head-multiplicity'), ('span_mismatch', 'alignment/strip-span-mismatch'),
])
def test_automatic_rejections_have_per_system_evidence(harvest, fault, reason):
    root, plan, rows, system, alignment, strict, review, truth, write = harvest
    attacks = truth.parent / 'attacks.htdemucs.npy'
    feature = root / '.eval-cache/ft' / rows[0]['slug'] / 'feat.npy'
    if fault == 'missing_attacks': attacks.unlink()
    elif fault == 'deficit': np.save(attacks, np.array(system['cursor_t'][:10]))
    elif fault == 'excess': np.save(attacks, np.linspace(2.5, 5.5, 30))
    elif fault == 'bad_order': np.save(attacks, np.linspace(5.3, 5.6, 12))
    elif fault == 'tempo_low': system['notated_beats'] = 1
    elif fault == 'tempo_high': system['notated_beats'] = 11
    elif fault == 'no_beats': del system['notated_beats']
    elif fault == 'cursor': system['cursor_x'] = [0, 0, 10, 20]
    elif fault == 'strict_source': strict['intervals'] = [[3.1, 3.2]]
    elif fault == 'strict_fitted':
        alignment['times'] = [7., 8.]
        strict['systems'].append({'system': 99, 'usable': True, 'interval': [9., 9.5], 'times': [7., 8.]})
    elif fault == 'strict_rounded': strict['intervals'] = [[5.65, 6.]]
    elif fault == 'features': feature.unlink()
    elif fault == 'feature_boundary': np.save(feature, np.zeros((500, 2)))
    elif fault == 'unknown': system['notes'][0]['inst'] = 'unknown'
    elif fault == 'withheld': system['notes'][0]['exportable'] = False
    elif fault == 'duplicates': alignment['scoring_reasons'] = ['tuplet duplicated head']
    elif fault == 'roll': system['notes'][0]['roll'] = True
    elif fault == 'span_mismatch': alignment['interval'] = [2., 5.]
    write()
    data = build(harvest, manual_review=False)
    assert data['songs'][rows[0]['slug']] == []
    result = data['system_audits'][rows[0]['slug']]['systems']['0']
    assert result['status'] == 'rejected' and reason in result['reasons']
    assert data['counts']['train']['rejected'][reason] == 1


def test_audio_counts_allow_only_notated_hh_cymbal_budget():
    predicted = np.arange(10) * .2 + 1
    core = [True] * 8 + [False] * 2
    evidence, reasons = weak.acoustic_check(predicted, core, np.r_[predicted, 3.2, 3.4], [50, 400])
    assert reasons == []
    assert evidence['maximum_onsets'] == 12
    assert evidence['matched_hh_cymbal_only_moments'] == 2
    _, reasons = weak.acoustic_check(predicted, core, np.r_[predicted, 3.2, 3.4, 3.6], [50, 400])
    assert 'acoustic-onset-count' in reasons


def test_simultaneous_heads_form_one_core_moment_with_no_extra_hh_budget(harvest):
    root, plan, rows, system, alignment, strict, review, truth, write = harvest
    system['notes'].extend([
        {'x': 0., 'inst': 'snare', 'verified': True},
        {'x': 0., 'inst': 'hihat', 'verified': True},
    ])
    write()
    entry, = build(harvest, manual_review=False)['songs'][rows[0]['slug']]
    assert len(entry['tokens']) == 12 and entry['tokens'][0] == 3
    assert entry['acceptance_evidence']['acoustic']['hh_cymbal_only_moments'] == 0


def test_onset_window_is_stop_exclusive_and_duplicate_peaks_are_one_onset():
    predicted = np.arange(8) * .2 + 1
    attacks = np.r_[predicted, predicted, .49, 4.]
    evidence, reasons = weak.acoustic_check(predicted, [True] * 8, attacks, [50, 400])
    assert reasons == [] and evidence['onsets'] == 8


@pytest.mark.parametrize('attacks', [[float('nan')], [-.1], [[1., 2.]]])
def test_invalid_acoustic_cache_is_rejected(attacks):
    with pytest.raises(ValueError, match='invalid-drum-stem-attacks'):
        weak.acoustic_check(np.arange(8) * .2 + 1, [True] * 8, attacks, [50, 400])


@pytest.mark.parametrize('matched,accepted', [(16, False), (17, True)])
def test_monotone_one_to_one_threshold_and_no_attack_reuse(matched, accepted):
    predicted = np.arange(20) * .2 + 1
    # Count passes with distant excess attacks; only the first N admit pairing.
    attacks = np.r_[predicted[:matched], np.linspace(8, 8.5, 20 - matched)]
    evidence, reasons = weak.acoustic_check(predicted, [True] * 20, attacks, [50, 900])
    assert evidence['matched_core_moments'] == matched
    assert ('acoustic-order-assignment' not in reasons) is accepted


def test_notated_bar_meter_evidence_and_tempo_boundaries():
    system = {'notes': [{'x': 10}], 'staff': {'bar_x': [0, 50, 100]}, 'time_signature': '3/4'}
    assert weak.notated_beats(system) == (6., 'measure-count/meter')
    del system['time_signature']
    assert weak.notated_beats(system) == (8., 'bar-count; assumed 4/4')
    for beats in (2, 15):
        system.update(notated_beats=beats, interval=[3., 6.])
        _, _, evidence = weak.automatic_window(system, {}, {}, {'seconds': 10})
        assert evidence['tempo_bpm'] in (40, 300)


def test_dense_diagnostics_clearly_report_missing_and_system_reasons(harvest, capsys):
    data = build(harvest, manual_review=False)
    video = weak.DENSE_SOLOS[0]
    audit = data['system_audits'][harvest[2][0]['slug']]
    audit['video_id'] = video
    audit['systems']['3'] = {'status': 'rejected', 'reasons': ['acoustic-onset-count']}
    weak.print_dense_diagnostics(data)
    output = capsys.readouterr().out
    assert f'{weak.DENSE_SOLOS[1]}: missing' in output
    assert 'system 3: rejected; acoustic-onset-count' in output
    assert '"acoustic-onset-count": 1' in output
