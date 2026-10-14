"""Temporary fixtures only. Never open/read the real vault, models or media.

    .venv/bin/python -B -m pytest -q -p no:cacheprovider tools/gpu/test_vault_absolute.py
    .venv/bin/python -B -m pytest -s -q -p no:cacheprovider tools/gpu/test_vault_absolute.py -k dry_run_output
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'vision_read')]
import chops_common as common
import vault_absolute as absolute
import vault_absolute_score as scorer
import vault_read as reader
from vault_absolute_io import guard, hashes, load, sha, write

CODE_ROOT = Path(__file__).resolve().parents[2]
MANUAL = ['xH-pMgRZWCA', 'YytKWBpYIyM', 'cDqw_3-F0Ag']


def events():
    moments = [(1., 'kick'), (1.04, 'snare'), (1.08, 'tom'), (1.12, 'kick'), (1.16, 'snare'),
               (1.2, 'tom'), (2., 'kick'), (2.1, 'snare'), (2.2, 'tom'), (2.3, 'kick')]
    return [dict(time=t, inst=c, pos=6 if c == 'tom' else 0, ghost=False) for t, c in moments]


def hit_fixture():
    return [SimpleNamespace(time=e['time'], inst=e['inst'], ghost=False) for e in events()]


def truth_fixture(folder, empty=False):
    write(folder / 'truth.json', [] if empty else events())
    write(folder / 'alignment.json', {'intervals': [[.9, 2.4]], 'systems': []})
    write(folder / 'strip.json', [])


@pytest.fixture
def env(tmp_path, monkeypatch):
    vault = absolute.vault
    out = tmp_path / 'truth/harvest'
    out.mkdir(parents=True)
    for key, path in {'ROOT': tmp_path, 'OUT': out, 'CONFIG': out / 'vault.json',
                      'MANIFEST': out / 'manifest.jsonl', 'LOG': out / 'vault_log.jsonl',
                      'SONGS': out / 'songs'}.items():
        monkeypatch.setattr(vault, key, path)
    monkeypatch.setattr(absolute, 'ROOT', tmp_path)
    monkeypatch.setattr(common, 'ROOT', tmp_path)
    monkeypatch.setattr(absolute.compare, 'ROOT', tmp_path)
    monkeypatch.setattr(absolute.compare, 'CACHE', tmp_path / '.eval-cache/ft')
    monkeypatch.setattr(common, 'CACHE', tmp_path / '.eval-cache/ft')
    monkeypatch.setattr(common, 'SAME_PERFORMANCE', ())
    monkeypatch.setattr(absolute, 'SPLIT', tmp_path / '.eval-cache/chops_split/split.json')
    monkeypatch.setattr(absolute, 'EXCLUSIONS', out / 'vault_final_exclusions.json')
    monkeypatch.setattr(absolute, 'VAULT_TRUTH', tmp_path / 'vault_truth/olliewoods_shorts')
    release = tmp_path / 'osrelease'
    release.write_text('6.6.0-microsoft-standard-WSL2')
    monkeypatch.setattr(absolute, 'RELEASE', release)
    ids = [f'V{i:010d}' for i in range(30)]
    write(vault.CONFIG, {'cutoff': 10, 'video_ids': ids + MANUAL + ['man00000001', 'mix00000001', 'nil00000001']})
    table = [['mix00000001', 'train000001']]
    write(out / 'same_performance.json', {'groups': table})
    rows = [dict(video='man00000001', slug='fixture-main', channel='moyin', at=1, status='read'),
            dict(video='nil00000001', slug='fixture-empty', channel='moyin', at=1, status='fetched'),
            dict(video='mix00000001', slug='fixture-mixed', channel='dailychops', at=1, status='read'),
            dict(video='train000001', slug='train-peer', channel='dailychops', at=1, status='read'),
            dict(video=MANUAL[0], slug='manual-manifest', channel='olliewoods_shorts', at=1, status='read')]
    vault.MANIFEST.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    write(absolute.SPLIT, {'vault': ids + MANUAL, 'train': []})
    write(absolute.EXCLUSIONS, {'exclusions': [{'video': v, 'reason': 'fixture Eric train-peer risk'} for v in MANUAL]})
    for r in rows:
        song = vault.SONGS / r['slug']
        song.mkdir(parents=True)
        np.save(song / 'activations.mix.npy', np.zeros((300, 5)))
        np.save(song / 'attacks.htdemucs.npy', np.zeros(3))
        (song / 'drums.htdemucs.wav').write_bytes(b'fixture stem; no decoder')
        if r['video'] != 'nil00000001':
            truth_fixture(song / 'truth')
    for video in ids:
        song = tmp_path / 'sources/olliewoods_shorts' / video
        song.mkdir(parents=True)
        write(song / 'meta.json', {'video_id': video, 'channel': 'olliewoods_shorts'})
        for name in ('video.webm', 'drums.htdemucs.wav', 'attacks.htdemucs.npy'):
            (song / name).write_bytes(b'fixture: never decoded')
    for name in ('align.py', 'bands.py', 'tiers.py'):
        path = tmp_path / 'tools/truth_video' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# fixture gate provenance only\n')
    directory = tmp_path / 'scores/chops/chopround3'
    tags = [dict(tag=f'chopround3_A_s{s}', arm='A', seed=s, recipe='v2') for s in (0, 1)]
    plan = dict(name='chopround3', tags=tags, arms={'A': {'songs': []}}, steps=1500,
                songs=[], references=[], channels=list(common.train_roster([])),
                folds=[dict(train=[], held=[], predict=[], channels=[]) for _ in (0, 1)], hashes={},
                same_performance=table, same_performance_sha256=sha(out / 'same_performance.json'))
    path = directory / 'plan.json'
    write(path, plan)
    for tag in tags:
        frozen = {}
        for fold in (0, 1):
            checkpoint = absolute.compare.CACHE / 'models' / f'{tag["tag"]}-fold{fold}.pt'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f'fixture weights {tag["seed"]} {fold}'.encode())
            write(checkpoint.with_suffix('.json'), dict(held=[], train=[], recipe='v2', arm='A',
                  args=dict(steps=1500, device='cuda', seed=tag['seed'], doubt=.3,
                            batch=12, seconds=8., lr=1e-4, distill=.3)))
            frozen[checkpoint.relative_to(tmp_path).as_posix()] = sha(checkpoint)
        write(directory / tag['tag'] / 'done.json', {'tag': tag['tag'], 'plan_sha256': sha(path), 'checkpoints': frozen})
    return SimpleNamespace(root=tmp_path, plan=path, ids=ids, directory=directory / 'vault_absolute',
                           release=release, monkeypatch=monkeypatch)


def freeze_request(env):
    job = absolute.preflight(env.plan)
    env.directory.mkdir()
    opening = absolute.vault.open_final(job['model_tag'])  # fake ROOT/log only
    job['opening_id'] = opening
    write(env.directory / 'request.json', job)
    write(env.directory / 'once.json', {'opening_id': opening, 'model_tag': job['model_tag'],
                                      'request_sha256': sha(env.directory / 'request.json')})
    return job


def test_dry_run_output(env, capsys):
    before = hashes([p for p in env.root.rglob('*') if p.is_file()], env.root)
    with guard(reads_forbidden=[absolute.vault.SONGS, env.root / 'sources', env.root / 'vault_truth']):
        result = absolute.run(env.plan, dry_run=True)
    assert result['manifest_vault_count'] == 2 and result['ollie_vault_count'] == 30
    assert len(result['checkpoints']) == 4
    assert not result['vault_opened'] and not result['sealed_payloads_read']
    assert not result['development_payload_hashes_checked']
    assert set(result['excluded']['manual']) == set(MANUAL)
    assert 'mix00000001' in result['excluded']['mixed_performance']
    assert hashes([p for p in env.root.rglob('*') if p.is_file()], env.root) == before
    # -s makes this fixture-only demonstration reviewable on a mirror.
    output = capsys.readouterr().out
    print(output)


@pytest.mark.parametrize('mirror', [False, True])
def test_real_run_refuses_without_wsl_authority_before_attempt(env, mirror):
    if mirror:
        (absolute.vault.OUT / '.wsl-mirror').touch()
    else:
        env.release.write_text('Darwin')
    with pytest.raises(RuntimeError, match='WSL'):
        absolute.run(env.plan)
    assert not absolute.vault.LOG.exists()
    assert not (env.directory / 'once.json').exists()
    absolute.run(env.plan, dry_run=True)  # mirror does not block metadata preview


@pytest.mark.parametrize('mutation', ['weights', 'done', 'seed', 'count', 'symlink', 'sealed_hash'])
def test_invalid_preflight_never_opens_or_creates_once(env, mutation):
    if mutation == 'weights':
        checkpoint = absolute.compare.CACHE / 'models/chopround3_A_s0-fold0.pt'
        checkpoint.write_bytes(b'changed weights')
    elif mutation == 'done':
        (env.plan.parent / 'chopround3_A_s0/done.json').unlink()
    elif mutation == 'seed':
        plan = load(env.plan); plan['tags'][1]['seed'] = 0; write(env.plan, plan)
    elif mutation == 'count':
        data = load(absolute.SPLIT); data['vault'].pop(0); write(absolute.SPLIT, data)
    elif mutation == 'symlink':
        source = env.root / 'sources/olliewoods_shorts' / env.ids[0]
        (source / 'video.webm').unlink(); (source / 'video.webm').symlink_to(source / 'drums.htdemucs.wav')
    else:
        plan = load(env.plan)
        path = absolute.vault.SONGS / 'fixture-main/truth/truth.json'
        plan['hashes'][path.relative_to(env.root).as_posix()] = 'not-read'
        write(env.plan, plan)
    with pytest.raises((ValueError, OSError)):
        absolute.run(env.plan)
    assert not absolute.vault.LOG.exists()
    assert not (env.directory / 'once.json').exists()


def test_once_retains_failed_open_attempt_and_refuses_retry(env):
    env.monkeypatch.setattr(absolute, 'runtime_preflight', lambda: None)
    env.monkeypatch.setattr(absolute.vault, 'open_final', lambda *a: (_ for _ in ()).throw(ValueError('log broken')))
    with pytest.raises(ValueError, match='log broken'):
        absolute.run(env.plan)
    assert load(env.directory / 'once.json')['state'] == 'attempted'
    with pytest.raises(FileExistsError, match='once.json'):
        absolute.run(env.plan)


def test_vault_reader_refuses_missing_opening_before_even_request(env):
    absolute.vault.LOG.write_text('')
    with pytest.raises(ValueError, match='matching durable'):
        reader.execute(env.root / 'missing.json', 'missing', 'chopround3_A_absolute')
    assert not absolute.VAULT_TRUTH.exists()


def test_actual_direct_child_resume_checks_pid_tag_and_opening(env):
    opening = absolute.vault.open_final('chopround3_A_absolute')
    code = '''import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import vault
root = Path(sys.argv[2]); vault.OUT = root / 'truth/harvest'
vault.CONFIG = vault.OUT / 'vault.json'; vault.MANIFEST = vault.OUT / 'manifest.jsonl'; vault.LOG = vault.OUT / 'vault_log.jsonl'
vault.resume_final(sys.argv[3], sys.argv[4])
'''
    args = [sys.executable, '-B', '-c', code, str(CODE_ROOT / 'tools/harvest'), str(env.root), opening]
    assert subprocess.run([*args, 'chopround3_A_absolute'], capture_output=True).returncode == 0
    assert subprocess.run([*args, 'wrong-tag'], capture_output=True).returncode != 0
    row = json.loads(absolute.vault.LOG.read_text())
    row['pid'] += 1
    absolute.vault.LOG.write_text(json.dumps(row) + '\n')
    assert subprocess.run([*args, 'chopround3_A_absolute'], capture_output=True).returncode != 0


def test_channel_read_still_refuses_vault_without_touching_source(env):
    import channel_read
    from tools.harvest import vault as development_vault
    env.monkeypatch.setattr(development_vault, 'config', absolute.vault.config)
    env.monkeypatch.setattr(development_vault, 'Vault', absolute.vault.Vault)
    with pytest.raises(ValueError, match='vault input forbidden'):
        channel_read.require_allowed(env.ids[0], frozenset([env.ids[0]]))


def test_canonical_binding_is_narrow_and_restores_development(env):
    import common as vision_common
    import hybrid
    folder = absolute.VAULT_TRUTH / env.ids[0]
    original = vision_common.confined
    with reader.canonical_paths(folder, lambda v: None):
        vision_common.dump(folder / 'allowed.json', {'purpose': 'canonical_truth'})
        with pytest.raises(PermissionError):
            vision_common.dump(env.root / '.eval-cache/vision_pilot/forbidden.json', {})
        with pytest.raises(PermissionError):
            hybrid.confined(env.root / 'truth/harvest/manifest.jsonl')
    assert vision_common.confined is original
    with pytest.raises(ValueError):
        vision_common.dump(folder / 'outside-development.json', {})


def test_runtime_dependency_failure_does_not_consume_once(env):
    def missing():
        raise RuntimeError('CUDA unavailable')
    env.monkeypatch.setattr(absolute, 'runtime_preflight', missing)
    with pytest.raises(RuntimeError, match='CUDA'):
        absolute.run(env.plan)
    assert not absolute.vault.LOG.exists() and not (env.directory / 'once.json').exists()


def test_request_mutation_and_wrong_opening_refused_before_source(env):
    job = freeze_request(env)
    env.monkeypatch.setattr(os, 'getppid', os.getpid)
    request = env.directory / 'request.json'
    with pytest.raises(ValueError, match='matching durable'):
        reader.execute(request, 'wrong-opening', job['model_tag'])
    changed = load(request); changed['songs'][0]['video'] = 'changed0001'; write(request, changed)
    with pytest.raises(ValueError, match='request changed'):
        reader.execute(request, job['opening_id'], job['model_tag'])
    assert not absolute.VAULT_TRUTH.exists()


def test_strict_equals_chops_scoring_on_fast_and_slow_fixture(env):
    import chops_scoring
    import fill_score
    song = absolute.vault.SONGS / 'fixture-main'
    env.monkeypatch.setattr(fill_score.harvest, 'SONGS', absolute.vault.SONGS)
    env.monkeypatch.setattr(chops_scoring.vault, 'Vault', lambda: SimpleNamespace(contains_song=lambda _: False))
    env.monkeypatch.setitem(sys.modules, 'absolute_fixture_detector', SimpleNamespace(detect=lambda _: hit_fixture()))
    expected = chops_scoring.score_one(('strict', {'slug': song.name, 'channel': 'moyin'}, 'absolute_fixture_detector:detect'))
    actual, has_truth = scorer.strict_counts(song / 'truth', hit_fixture())
    assert has_truth and actual == expected
    assert actual['fast.tom.ref'] == 2 and actual['slow.tom.ref'] == 1
    assert actual['all.tom.tp'] == 3


def test_loose_equals_existing_vault_score_song(env):
    import fill_seq
    import ft_variants
    import chops_vault_score
    refs = [(e['time'], frozenset([e['inst']])) for e in events()]
    env.monkeypatch.setattr(fill_seq, 'notated', lambda *a: [(1., 2.3, refs)])
    env.monkeypatch.setattr(ft_variants, 'tuned', lambda _: hit_fixture())
    expected = chops_vault_score.score_song({'slug': 'fixture-main'}, 'fixture')
    actual = scorer.loose_counts(absolute.vault.SONGS / 'fixture-main', hit_fixture())
    assert actual == expected


def install_fake_cv(env):
    import channel_read_core as core
    calls = []
    def check():
        rows = [json.loads(line) for line in absolute.vault.LOG.read_text().splitlines()]
        assert len(rows) == 1 and rows[0]['pid'] == os.getpid()
    def notation(song, out, legend):
        check(); calls.append('notation'); write(out / 'scan.json', {})
    def read_cv(song, out, legend, scan):
        calls.append('read_cv'); truth_fixture(out)
    def extract(song, channel, native, read, scan, system_id):
        calls.append('extract'); native.mkdir(parents=True)
        return dict(video=song.name, systems=[], native=native.relative_to(env.root).as_posix())
    def prepare(inventory, work):
        calls.append('prepare'); write(work / 'hybrid/prepared.json', {})
    def stage(inventory, work, out):
        calls.append('stage'); truth_fixture(out)
    def gate(song, out):
        calls.append('gate'); return {'strict_notes': len(events()), 'scoring_notes': len(events())}
    for name, fn in dict(notation=notation, read_cv=read_cv, counts=lambda _: {}, extract=extract,
                         prepare=prepare, stage=stage, gate=gate).items():
        env.monkeypatch.setattr(core, name, fn)
    return calls


def test_fake_end_to_end_once_log_order_exclusions_and_write_locations(env):
    env.monkeypatch.setattr(absolute, 'runtime_preflight', lambda: None)
    calls = install_fake_cv(env)
    env.monkeypatch.setattr(os, 'getppid', os.getpid)  # emulate direct children in this synthetic run
    def command(script, job, opening, log):
        request = env.directory / 'request.json'
        assert load(env.directory / 'once.json')['opening_id'] == opening
        if script.endswith('vault_read.py'):
            reader.execute(request, opening, job['model_tag'])
        else:
            scorer.execute(request, opening, job['model_tag'])
    env.monkeypatch.setattr(absolute, 'command', command)
    def audio(job):
        assert (env.directory / 'inputs.json').is_file()  # truth frozen BEFORE any model inference
        for item in job['songs']:
            song, cache = env.root / item['song'], env.root / item['cache']
            cache.mkdir(parents=True, exist_ok=True)
            np.save(cache / 'feat.npy', np.zeros((300, 2)))
            if item['cohort'] == 'ollie':
                np.save(song / 'activations.mix.npy', np.zeros((300, 5)))
                np.save(song / 'attacks.htdemucs.npy', np.zeros(3))
                (song / 'drums.htdemucs.wav').write_bytes(b'fixture')
    env.monkeypatch.setattr(absolute, 'prepare_audio', audio)
    def predictions(job):
        for item in job['songs']:
            for fold in (0, 1):
                np.save(env.root / item['cache'] / f'student-{job["model_tag"]}-fold{fold}.npy', np.zeros((300, 5)))
        return {'device': 'cpu', 'reason': 'fixture'}
    env.monkeypatch.setattr(absolute, 'inference', predictions)
    def hits(song, cache, tag):
        np.save(cache / f'snare-student-{tag}.npy', np.zeros(300))
        return hit_fixture()
    env.monkeypatch.setattr(scorer, 'full_hits', hits)
    import fill_seq
    env.monkeypatch.setattr(fill_seq, 'notated', lambda folder, *_: [] if 'empty' in str(folder) else
        [(1., 2.3, [(e['time'], frozenset([e['inst']])) for e in events()])])
    before = hashes([p for p in absolute.vault.SONGS.rglob('*') if p.is_file()], env.root)
    manifest_hash = sha(absolute.vault.MANIFEST)
    result = absolute.run(env.plan)
    assert result['coverage']['manifest'] == dict(considered=2, with_strict_truth=1, with_notation=1, with_strict_fills=1, unusable_inputs=0)
    assert result['coverage']['ollie']['considered'] == 30
    assert result['coverage']['combined']['with_strict_truth'] == 31
    assert result['strict']['combined']['fast']['tom']['counts'] == [62, 62, 62]
    assert calls == ['notation', 'read_cv', 'extract', 'prepare', 'stage', 'gate'] * 30
    assert result['device']['device'] == 'cpu' and result['device']['cpu_check']['checked'] is False
    assert load(env.directory / 'device.json')['device'] == 'cpu'
    assert len(absolute.vault.LOG.read_text().splitlines()) == 1
    assert sha(absolute.vault.MANIFEST) == manifest_hash
    assert hashes([p for p in absolute.vault.SONGS.rglob('*') if p.is_file()], env.root) == before
    for name in ('.eval-cache/ft/solos', '.eval-cache/vision_pilot'):
        assert not (env.root / name).exists()
    assert (env.directory / 'final.json').is_file() and (env.directory / 'REPORT.md').is_file()
    assert load(env.directory / 'once.json')['state'] == 'complete'
    with pytest.raises(FileExistsError, match='once.json'):
        absolute.run(env.plan)
    assert len(absolute.vault.LOG.read_text().splitlines()) == 1


@pytest.mark.parametrize('reason', ['no sustained bottom white Shorts page', 'broken decoder'])
def test_scan_rejection_exports_no_truth_and_unexpected_errors_abort(env, reason):
    job = freeze_request(env)
    env.monkeypatch.setattr(os, 'getppid', os.getpid)
    job['songs'] = [s for s in job['songs'] if s['cohort'] == 'ollie'][:1]
    write(env.directory / 'request.json', job)
    once = load(env.directory / 'once.json'); once['request_sha256'] = sha(env.directory / 'request.json')
    write(env.directory / 'once.json', once)
    import channel_read_core as core
    def no_scan(*a):
        raise ValueError(reason)
    env.monkeypatch.setattr(core, 'notation', no_scan)
    env.monkeypatch.setattr(core, 'read_cv', lambda *a: pytest.fail('no candidate scan reached downstream CV'))
    if reason == 'broken decoder':
        with pytest.raises(ValueError, match=reason):
            reader.execute(env.directory / 'request.json', job['opening_id'], job['model_tag'])
    else:
        reader.execute(env.directory / 'request.json', job['opening_id'], job['model_tag'])
        song = env.root / job['songs'][0]['song']
        assert load(song / 'truth/reader_rejection.json')['status'] == 'reader_rejected'
        assert not (song / 'truth/truth.json').exists()
        counts, has_truth = scorer.strict_counts(song / 'truth', [])
        assert not counts and not has_truth and not scorer.notation_groups(song / 'truth')


def test_snapshot_detects_changed_truth_added_truth_and_changed_checkpoint(env):
    job = freeze_request(env)
    snapshot = absolute.truth_snapshot(job)
    absolute.verify_inputs(job, snapshot)
    path = absolute.vault.SONGS / 'fixture-main/truth/truth.json'
    original = path.read_bytes(); path.write_text('[]')
    with pytest.raises(ValueError, match='frozen input changed'):
        absolute.verify_inputs(job, snapshot)
    path.write_bytes(original)
    write(path.parent / 'added.json', {})
    with pytest.raises(ValueError, match='inventory changed'):
        absolute.verify_inputs(job, snapshot)
    (path.parent / 'added.json').unlink()
    checkpoint = absolute.compare.CACHE / 'models/chopround3_A_s0-fold0.pt'
    checkpoint.write_bytes(b'changed frozen weight')
    with pytest.raises(ValueError, match='frozen input changed'):
        absolute.verify_inputs(job, snapshot)


def test_full_chain_averages_folds_and_snare_stays_final(env):
    import ft_variants
    from groovemap import models, dynamics, fills, toms
    song = absolute.vault.SONGS / 'fixture-main'
    cache = env.directory / 'cache/manifest/fixture-main'
    cache.mkdir(parents=True)
    tag = 'chopround3_A_absolute'
    np.save(cache / f'student-{tag}-fold0.npy', np.full((300, 5), .2))
    np.save(cache / f'student-{tag}-fold1.npy', np.full((300, 5), .6))
    calls = []
    env.monkeypatch.setattr(dynamics, 'snare_response', lambda stem, act: calls.append(('snare', float(act[0, 0]))) or np.zeros(300))
    env.monkeypatch.setattr(models, 'pick_hits', lambda act, *a, **kw: calls.append(('pick', float(act[0, 0]))) or hit_fixture())
    env.monkeypatch.setattr(fills, 'tom_runs', lambda hits, *a: calls.append(('tom_runs',)) or hits)
    env.monkeypatch.setattr(fills, 'fill_snares', lambda hits, *a: calls.append(('fill_snares',)) or hits)
    env.monkeypatch.setattr(toms, 'tom_sizes', lambda stem, hits: calls.append(('tom_sizes',)) or hits)
    with guard(writes=[cache]):
        assert scorer.full_hits(song, cache, tag) == hit_fixture()
    assert calls[0][1] == pytest.approx(.4) and calls[1][1] == pytest.approx(.4)
    assert [c[0] for c in calls] == ['snare', 'pick', 'tom_runs', 'fill_snares', 'tom_sizes']
    assert (cache / f'snare-student-{tag}.npy').exists()
    assert not (absolute.compare.CACHE / 'solos').exists()


def test_prediction_uses_both_seeds_in_each_separate_fold_file(env):
    import torch
    import train
    job = absolute.preflight(env.plan)
    job['songs'] = job['songs'][:1]
    cache = env.root / job['songs'][0]['cache']
    cache.mkdir(parents=True)
    np.save(cache / 'feat.npy', np.zeros((300, 2)))
    class Model:
        def load_state_dict(self, value):
            self.value = value
        def eval(self):
            return self
    def weights(path, **kwargs):
        return .2 + (.4 if '_s1-' in path.name else 0) + (.2 if '-fold1.' in path.name else 0)
    env.monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    env.monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    env.monkeypatch.setattr(torch, 'load', weights)
    env.monkeypatch.setattr(train, 'original_model', lambda _: Model())
    env.monkeypatch.setattr(train, 'run', lambda model, feat, device: np.full((len(feat), 5), model.value))
    with guard(writes=[cache]):
        absolute.predict(job, 'cuda')
    assert np.load(cache / 'student-chopround3_A_absolute-fold0.npy')[0, 0] == pytest.approx(.4)
    assert np.load(cache / 'student-chopround3_A_absolute-fold1.npy')[0, 0] == pytest.approx(.6)


def test_audio_preparation_uses_m4a_or_video_and_only_explicit_caches(env):
    import adtof_pytorch.audio as audio
    from groovemap import models
    job = absolute.preflight(env.plan)
    job['songs'] = [job['songs'][0], *[s for s in job['songs'] if s['cohort'] == 'ollie'][:2]]
    ollie = job['songs'][1:]
    (env.root / ollie[0]['source'] / 'audio.m4a').write_bytes(b'mix fixture')
    for item in ollie:
        (env.root / item['song']).mkdir(parents=True)
    chosen = []
    def ffmpeg(argv, **kwargs):
        assert argv[:2] == ['ffmpeg', '-nostdin'] and '-n' in argv
        chosen.append(Path(argv[argv.index('-i') + 1]).name)
        Path(argv[-1]).write_bytes(b'decoded fixture')
    def feature(song):
        folder = absolute.compare.CACHE / song.name
        folder.mkdir(parents=True)
        np.save(folder / 'feat.npy', np.zeros((300, 2)))
        return folder
    env.monkeypatch.setattr(absolute.subprocess, 'run', ffmpeg)
    env.monkeypatch.setattr(absolute.compare, 'feature', feature)
    env.monkeypatch.setattr(audio, 'process_audio_file', lambda _: (np.zeros((300, 2, 1)), None))
    env.monkeypatch.setattr(models, 'drum_activations', lambda *a, **kw: np.full((300, 5), .25))
    with guard(writes=[env.directory, absolute.VAULT_TRUTH,
                       absolute.compare.CACHE / job['songs'][0]['slug'] / 'feat.npy']):
        absolute.prepare_audio(job)
        assert not (env.root / ollie[0]['song'] / 'activations.mix.npy').exists()  # model work is a separate phase
        absolute.mix_activations(job, 'cpu')
    assert chosen == ['audio.m4a', 'video.webm']
    assert np.load(env.root / ollie[0]['song'] / 'activations.mix.npy').shape == (300, 5)
    for item in job['songs']:
        assert (env.root / item['cache'] / 'feat.npy').is_file()
    assert not (absolute.compare.CACHE / 'solos').exists()


def test_cuda_oom_reruns_whole_inference_phase_on_cpu(env, monkeypatch):
    import torch
    devices = []
    monkeypatch.setattr(absolute, 'pick_device', lambda: ('cuda', 'fixture'))
    def mix(job, device):
        devices.append(('mix', device))
    def predict(job, device):
        devices.append(('predict', device))
        if device == 'cuda':
            raise torch.cuda.OutOfMemoryError('CUDA out of memory. Tried to allocate 3 GiB')
    monkeypatch.setattr(absolute, 'mix_activations', mix)
    monkeypatch.setattr(absolute, 'predict', predict)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    record = absolute.inference({})
    assert record['device'] == 'cpu' and 'OOM' in record['reason']
    assert devices == [('mix', 'cuda'), ('predict', 'cuda'), ('mix', 'cpu'), ('predict', 'cpu')]
    monkeypatch.setattr(absolute, 'predict', lambda job, device: (_ for _ in ()).throw(RuntimeError('other failure')))
    with pytest.raises(RuntimeError, match='other failure'):
        absolute.inference({})


def test_pick_device_needs_free_vram_and_caps_memory(monkeypatch):
    import torch
    caps = []
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_per_process_memory_fraction', caps.append)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (2 * 2**30, 16 * 2**30))
    assert absolute.pick_device()[0] == 'cpu' and caps == []
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (10 * 2**30, 16 * 2**30))
    assert absolute.pick_device()[0] == 'cuda' and caps == [pytest.approx(2 / 16)]


def test_device_check_compares_cuda_arrays_with_cpu_rerun(env, monkeypatch):
    import torch
    import train
    from groovemap import models
    job = absolute.preflight(env.plan)
    songs = [*[s for s in job['songs'] if s['cohort'] == 'manifest'][:1], *[s for s in job['songs'] if s['cohort'] == 'ollie'][:1]]
    job['songs'] = songs
    class Model:
        def load_state_dict(self, value):
            self.value = value
        def eval(self):
            return self
    monkeypatch.setattr(torch, 'load', lambda path, **kw: .2 + (.4 if '_s1-' in path.name else 0))
    monkeypatch.setattr(train, 'original_model', lambda _: Model())
    monkeypatch.setattr(train, 'run', lambda model, feat, device: np.full((len(feat), 5), model.value))
    monkeypatch.setattr(models, 'drum_activations', lambda *a, **kw: np.full((300, 5), .25))
    for item in songs:
        cache, song = env.root / item['cache'], env.root / item['song']
        cache.mkdir(parents=True, exist_ok=True); song.mkdir(parents=True, exist_ok=True)
        np.save(cache / 'feat.npy', np.zeros((300, 2)))
        for fold in (0, 1):
            np.save(cache / f'student-{job["model_tag"]}-fold{fold}.npy', np.full((300, 5), .4 + 1e-6))
        np.save(song / 'activations.mix.npy', np.full((300, 5), .25))
    check = absolute.device_check(job, {'device': 'cuda'})
    assert check['checked'] and check['max_abs'] == pytest.approx(1e-6, abs=1e-7)
    assert set(check['songs'][songs[1]['slug']]) == {'student_fold0', 'student_fold1', 'mix'}
    assert absolute.device_check(job, {'device': 'cpu'})['checked'] is False


def test_student_forward_retries_without_cudnn_on_long_input(env, monkeypatch):
    import torch
    import train
    job = absolute.preflight(env.plan)
    job['songs'] = job['songs'][:1]
    cache = env.root / job['songs'][0]['cache']
    cache.mkdir(parents=True)
    np.save(cache / 'feat.npy', np.zeros((300, 2)))
    class Model:
        def load_state_dict(self, value):
            pass
        def eval(self):
            return self
    calls = []
    def run(model, feat, device):
        calls.append(torch.backends.cudnn.enabled)
        if torch.backends.cudnn.enabled:
            raise RuntimeError('cuDNN error: CUDNN_STATUS_NOT_SUPPORTED')
        return np.ones((len(feat), 5))
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: None)
    monkeypatch.setattr(train, 'original_model', lambda _: Model())
    monkeypatch.setattr(train, 'run', run)
    out = list(absolute.model_runs(job, 'cpu', 0, job['songs']))
    assert len(out) == 2 and calls == [True, False, True, False] and torch.backends.cudnn.enabled
    monkeypatch.setattr(train, 'run', lambda *a: (_ for _ in ()).throw(RuntimeError('other')))
    with pytest.raises(RuntimeError, match='other'):
        list(absolute.model_runs(job, 'cpu', 0, job['songs']))


@pytest.mark.parametrize('mix', [np.zeros((0, 5)), np.zeros((20, 4)), np.full((20, 5), np.nan)])
def test_unusable_mix_is_recorded_not_fatal(tmp_path, mix):
    np.save(tmp_path / 'activations.mix.npy', mix)
    assert scorer.unusable_mix(tmp_path).startswith('unusable')
    np.save(tmp_path / 'activations.mix.npy', np.zeros((20, 5)))
    assert scorer.unusable_mix(tmp_path) is None


@pytest.mark.parametrize('path', ['.eval-cache/ft/solos/x.npy', '.eval-cache/vision_pilot/x.json',
                                  'truth/harvest/manifest.jsonl', 'truth/harvest/songs/x/truth.json'])
def test_write_guard_rejects_protected_destinations(env, path):
    with guard(writes=[env.directory, absolute.VAULT_TRUTH]):
        with pytest.raises(PermissionError, match='write outside'):
            write(env.root / path, {})
