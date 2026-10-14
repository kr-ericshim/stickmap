"""Offline safety checks; all fixture writes stay under pytest's temp directory."""
from __future__ import annotations
import json
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys
from threading import Barrier, Event, Lock
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'harvest')]
import harvest
import harvest_transfer as transfer
import harvest_read as reader
import harvest_wsl as controller
from chops_common import fold_policy, partitions


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    out = root / 'truth/harvest'
    out.mkdir(parents=True)
    for module in (harvest, transfer.vault):
        monkeypatch.setattr(module, 'ROOT', root)
        monkeypatch.setattr(module, 'OUT', out)
        monkeypatch.setattr(module, 'SONGS', out / 'songs')
        monkeypatch.setattr(module, 'MANIFEST', out / 'manifest.jsonl')
    monkeypatch.setattr(transfer.vault, 'CONFIG', out / 'vault.json')
    monkeypatch.setattr(transfer.vault, 'LOG', out / 'vault_log.jsonl')
    config = {'cutoff': 100, 'video_ids': [], 'train_channels': ['dailychops']}
    (out / 'vault.json').write_text(json.dumps(config))
    return out


def bundle(tmp_path, dataset, manifest):
    path = tmp_path / 'bundle'
    path.mkdir()
    (path / 'manifest.jsonl').write_bytes(manifest)
    (path / 'vault.json').write_bytes((dataset / 'vault.json').read_bytes())
    (path / 'vault_log.jsonl').write_bytes(b'')
    return path


def test_migration_and_pull_preserve_first_rows_and_bytes(dataset, tmp_path):
    first = b'{"video":"abcdefghijk","channel":"dailychops","at":200,"status":"fetched"}\n'
    later = b'{"video":"abcdefghijk","at":300,"status":"read"}\n'
    path = bundle(tmp_path, dataset, first)
    transfer.import_bundle(path, initialize=True)
    assert (dataset / 'manifest.jsonl').read_bytes() == first
    (dataset / 'manifest.jsonl').write_bytes(first + later)
    transfer.import_bundle(path)
    assert (dataset / 'manifest.jsonl').read_bytes() == first + later
    assert not transfer.vault.Vault().is_vault('abcdefghijk')
    # Mac is now the older mirror; publishing the remote journal is replacement,
    # not append. A repeated pull must not duplicate the fetched/first row.
    (path / 'manifest.jsonl').write_bytes(first + later)
    (dataset / 'manifest.jsonl').write_bytes(first)
    transfer.publish_mirror(path)
    transfer.publish_mirror(path)
    assert (dataset / 'manifest.jsonl').read_bytes() == first + later
    assert (dataset / '.wsl-mirror').exists()


def test_divergence_and_partial_journal_fail_without_changes(dataset, tmp_path):
    initial = b'{"video":"abcdefghijk","at":100}\n'
    (dataset / 'manifest.jsonl').write_bytes(initial)
    path = bundle(tmp_path, dataset, initial + b'{"video":"localbranch","at":200}\n')
    with pytest.raises(ValueError, match='diverged'):
        transfer.import_bundle(path)
    assert (dataset / 'manifest.jsonl').read_bytes() == initial
    (path / 'manifest.jsonl').write_bytes(b'{"video":')
    with pytest.raises(ValueError, match='unfinished'):
        transfer.import_bundle(path, initialize=True)
    assert (dataset / 'manifest.jsonl').read_bytes() == initial


def test_initial_extension_is_allowed_only_before_wsl_authority(dataset, tmp_path):
    first = b'{"video":"abcdefghijk","at":100}\n'
    extra = b'{"video":"lmnopqrstuv","at":200}\n'
    (dataset / 'manifest.jsonl').write_bytes(first)
    path = bundle(tmp_path, dataset, first + extra)
    transfer.import_bundle(path, initialize=True)
    assert (dataset / 'manifest.jsonl').read_bytes() == first + extra
    (path / 'manifest.jsonl').write_bytes(first + extra + b'{"video":"macbranch01","at":300}\n')
    with pytest.raises(ValueError, match='diverged'):
        transfer.import_bundle(path, initialize=True)


def test_snapshot_only_canonical_compacted_outputs(dataset, tmp_path):
    song = dataset / 'songs/song-abcdefghijk'
    for folder in ('truth', 'truth-history/old', 'frames'):
        (song / folder).mkdir(parents=True)
    for name in ('video.mp4', 'audio.wav', 'drums.htdemucs.opus', 'attacks.htdemucs.npy',
                 'truth/truth.json', 'truth/alignment.json', 'truth/rejection.log',
                 'truth/crop.png', 'truth-history/old/truth.json', 'frames/frame.png'):
        (song / name).write_bytes(b'fixture')
    row = {'video': 'abcdefghijk', 'channel': 'dailychops', 'at': 200,
           'status': 'compacted', 'slug': song.name}
    (dataset / 'manifest.jsonl').write_text(json.dumps(row) + '\n')
    destination = tmp_path / 'snapshot'
    transfer.snapshot(destination, ['dailychops'])
    files = {str(p.relative_to(destination / 'songs' / song.name))
             for p in (destination / 'songs' / song.name).rglob('*') if p.is_file()}
    assert files == {'drums.htdemucs.opus', 'attacks.htdemucs.npy',
                     'truth/truth.json', 'truth/alignment.json', 'truth/rejection.log'}
    assert (destination / 'manifest.jsonl').read_bytes() == (dataset / 'manifest.jsonl').read_bytes()


def test_fetch_additional_limit_device_and_this_venv(dataset, tmp_path, monkeypatch):
    channels = tmp_path / 'channels.json'
    channels.write_text(json.dumps({'channels': [{'name': 'dailychops', 'layout': 'dailychops'}]}))
    monkeypatch.setattr(harvest, 'CHANNELS', channels)
    monkeypatch.setattr(harvest, 'candidates', lambda _: [('abcdefghijk', 100, 'one'), ('lmnopqrstuv', 100, 'two')])
    monkeypatch.setattr(harvest, 'free_gb', lambda: 100)
    monkeypatch.setattr(harvest, 'executable', lambda name: '/fake/' + name)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if 'groovemap.cli' in argv:
            video = argv[argv.index('transcribe') + 1].split('=')[-1]
            song = harvest.SONGS / ('song-' + video)
            song.mkdir()
            (song / 'video.mp4').write_bytes(b'video')
        elif argv[0].endswith('yt-dlp'):
            Path(argv[argv.index('-o') + 1]).write_bytes(b'video 1080p')
        return subprocess.CompletedProcess(argv, 0, 'ok', '')

    monkeypatch.setattr(harvest.subprocess, 'run', run)
    harvest.fetch(2, ['dailychops'], 1, device='cuda', additional=True, limit=1)
    assert len(harvest.manifest()) == 1
    assert calls[0][:3] == [sys.executable, '-m', 'groovemap.cli']
    assert calls[0][-2:] == ['--device', 'cuda']
    assert 'uv' not in calls[0]
    # The pipeline's 720p video is replaced by the default 1080p one.
    assert calls[1][0].endswith('yt-dlp') and harvest.VIDEO_FORMAT in calls[1]


@pytest.mark.parametrize('succeeds', [False, True])
def test_fetch_sharper_video_keeps_720p_until_success(dataset, tmp_path, monkeypatch, succeeds):
    channels = tmp_path / 'channels.json'
    channels.write_text(json.dumps({'channels': [{'name': 'dailychops', 'layout': 'dailychops'}]}))
    monkeypatch.setattr(harvest, 'CHANNELS', channels)
    monkeypatch.setattr(harvest, 'candidates', lambda _: [('abcdefghijk', 100, 'one')])
    monkeypatch.setattr(harvest, 'free_gb', lambda: 100)
    monkeypatch.setattr(harvest, 'executable', lambda name: '/fake/' + name)
    sleeps, downloads = [], []
    monkeypatch.setattr(harvest.time, 'sleep', sleeps.append)
    song = harvest.SONGS / 'song-abcdefghijk'

    def run(argv, **kwargs):
        if 'groovemap.cli' in argv:
            song.mkdir()
            (song / 'video.mp4').write_bytes(b'original 720p')
        else:
            downloads.append(argv)
            assert (song / 'video.mp4').read_bytes() == b'original 720p'
            target = Path(argv[argv.index('-o') + 1])
            assert target.name == 'video.1080p.part.mp4'
            if succeeds:
                target.write_bytes(b'replacement 1080p')
            return subprocess.CompletedProcess(argv, 0 if succeeds else 1, '', 'mock download failure')
        return subprocess.CompletedProcess(argv, 0, '', '')

    monkeypatch.setattr(harvest.subprocess, 'run', run)
    harvest.fetch(1, ['dailychops'], 1, max_height=1080)
    assert (song / 'video.mp4').read_bytes() == (b'replacement 1080p' if succeeds else b'original 720p')
    assert harvest.manifest()['abcdefghijk']['status'] == 'fetched'
    assert len(downloads) == (1 if succeeds else 3)
    assert sleeps == ([] if succeeds else [20, 40])
    assert not (song / 'video.1080p.part.mp4').exists()


def test_writer_lock_and_mac_mirror_guard(dataset):
    with harvest.writer_lock():
        with pytest.raises(RuntimeError, match='another harvest'):
            with harvest.writer_lock():
                pass
    (dataset / '.wsl-mirror').touch()
    with pytest.raises(RuntimeError, match='WSL owns'):
        with harvest.writer_lock():
            pass


def test_fold_policy_holds_new_channels_and_known_same_song_out():
    folds = fold_policy(['dailychops'])
    assert 'dailychops' in folds[1]
    assert 'dailychops' not in folds[0] + folds[2]
    songs = [{'slug': 'pop-last-christmas-abcdefghijk', 'channel': 'swartz'},
             {'slug': 'new-chop-lmnopqrstuv', 'channel': 'dailychops'},
             {'slug': 'independent', 'channel': 'yeol'}]
    # Timing-only references still protect against a known duplicate in training.
    refs = songs + [{'slug': 'other-last-christmas-ABCDEFGHIJK', 'channel': 'dailychops'}]
    result = partitions(songs, refs, folds)
    assert result[1]['held'] == ['new-chop-lmnopqrstuv']
    assert result[1]['train'] == ['independent']
    assert 'new-chop-lmnopqrstuv' in result[0]['train']


def test_comparison_report_uses_counts_and_requires_matching_completion(tmp_path):
    import chops_compare as compare
    from chops_common import sha
    tags = [{'tag': f'test_{recipe}_s{seed}', 'recipe': recipe, 'seed': seed}
            for recipe in ('v2', 'v5b') for seed in (0, 1)]
    plan = {'name': 'test', 'tags': tags, 'songs': [1], 'references': [1],
            'solos': [1], 'channels': ['dailychops']}
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    flat = {f'{subset}.{inst}.{key}': value for subset in ('all', 'fast')
            for inst in ('kick', 'snare', 'tom') for key, value in zip(('tp', 'est', 'ref'), (2, 4, 5))}
    global_counts = {inst: [2, 4, 5] for inst in ('kick', 'snare', 'tom', 'hihat', 'cymbal')}
    for item in tags:
        out = tmp_path / item['tag']
        out.mkdir()
        (out / 'done.json').write_text(json.dumps({'plan_sha256': sha(tmp_path / 'plan.json')}))
        for metric in ('loose', 'strict', 'solos', 'global'):
            (out / f'{metric}.json').write_text(json.dumps(global_counts if metric == 'global' else flat))
    compare.report(plan, tmp_path)
    output = json.loads((tmp_path / 'comparison.json').read_text())
    assert len(output['rows']) == 40
    assert output['rows'][0]['prf']['kick'] == [0.5, 0.4, 4 / 9]
    assert output['rows'][9]['all_song_counts'] == [10, 20, 25]
    assert output['rows'][9]['all_song_pr'] == [0.5, 0.4]
    (tmp_path / tags[0]['tag'] / 'done.json').write_text('{"plan_sha256":"stale"}')
    with pytest.raises(ValueError, match='incomplete scoring'):
        compare.report(plan, tmp_path)


@pytest.mark.parametrize('recipe', ['v2', 'v5b'])
def test_recipe_loader_includes_dailychops_but_refuses_explicit_vault(dataset, tmp_path, monkeypatch, recipe):
    # Exercise the real wrapper + synthetic known-channel filter, without model
    # loading, tensors, CUDA kernels, downloads, or the training loop.
    sys.path.insert(0, str(HERE / 'chops_recipe'))
    import train_recipe as runner
    import torch
    if recipe == 'v2':
        import train
    else:
        import train_synth as synth
        train = synth.train
        for name in ('_load_songs', 'P_UNISON_KICK', 'P_UNISON_TOM', 'P_DOUBLE'):
            monkeypatch.setattr(synth, name, getattr(synth, name))
        monkeypatch.setattr(synth, 'bank_for', lambda song: {0: [1], 2: [1]})
    for name in ('FOLDS', 'MODELS', 'load_songs', 'same_song'):
        monkeypatch.setattr(train, name, getattr(train, name))
    monkeypatch.setattr(runner.splits, 'FOLDS', runner.splits.FOLDS)
    cache = tmp_path / 'ft'
    meta = [{'slug': 'old-abcdefghijk', 'video_id': 'abcdefghijk', 'channel': 'swartz'},
            {'slug': 'new-lmnopqrstuv', 'video_id': 'lmnopqrstuv', 'channel': 'dailychops'}]
    for row in meta:
        path = cache / row['slug']
        path.mkdir(parents=True)
        (path / 'truth.json').write_text(json.dumps(row))
    folds = fold_policy(['dailychops'])
    plan = {'channels': [c for f in folds for c in f], 'songs': meta,
            'folds': partitions(meta, meta, folds)}
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'CACHE', cache)
    monkeypatch.setattr(runner, 'load_plan', lambda path: plan)
    monkeypatch.setattr(runner, 'check_plan', lambda data: None)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(train, 'original_model', lambda device: SimpleNamespace(eval=lambda: None))
    loaded = []

    def song(folder, teacher, device, **kw):
        row = json.loads((folder / 'truth.json').read_text())
        loaded.append(row['slug'])
        if recipe == 'v2':
            assert kw['withheld_mask'] is False
        return SimpleNamespace(**row)

    monkeypatch.setattr(train, 'Song', song)

    def main():
        songs = train.load_songs('cuda')
        assert {s.channel for s in songs} == {'swartz', 'dailychops'}
        train.MODELS.mkdir()
        (train.MODELS / 'test-fold1.json').write_text(json.dumps({'args': {}, 'held': ['new-lmnopqrstuv']}))

    monkeypatch.setattr(train, 'main', main)
    wrapper_args = ['train_recipe', '--recipe', recipe, '--plan', str(plan_path),
                    '--tag', 'test', '--fold', '1', '--seed', '0']
    monkeypatch.setattr(sys, 'argv', wrapper_args[:])
    runner.main()
    assert set(loaded) == {'old-abcdefghijk', 'new-lmnopqrstuv'}
    config = json.loads((dataset / 'vault.json').read_text())
    config['video_ids'] = ['lmnopqrstuv']
    (dataset / 'vault.json').write_text(json.dumps(config))
    loaded.clear()
    monkeypatch.setattr(sys, 'argv', wrapper_args[:])
    with pytest.raises(ValueError, match='vault/cohort'):
        runner.main()
    assert 'new-lmnopqrstuv' not in loaded  # sealed features/teacher were never loaded


@pytest.fixture
def reread_dataset(dataset, monkeypatch):
    (dataset / '.wsl-authority').touch()
    monkeypatch.setattr(reader, 'ROOT', dataset.parents[1])
    monkeypatch.setattr(reader.batch, 'channel_profile', lambda channel: (
        {'name': channel, 'enabled': True, 'rules': {}}, Path('/unused/profile.json')))
    monkeypatch.setattr(reader.batch, 'report', lambda root: None)
    monkeypatch.setattr(reader.shutil, 'disk_usage', lambda path: SimpleNamespace(free=100e9))
    monkeypatch.setattr(harvest, 'executable', lambda name: '/fake/' + name)
    monkeypatch.setattr(reader.time, 'sleep', lambda seconds: None)

    def offline(argv, **kwargs):
        pytest.fail(f'unmocked subprocess in offline test: {argv}')

    monkeypatch.setattr(reader.subprocess, 'run', offline)
    return dataset


def reread_song(out, video='abcdefghijk', channel='moyin', status='compacted', with_video=False):
    row = {'video': video, 'channel': channel, 'status': status, 'at': 200,
           'slug': 'song-' + video, 'seconds': 120}
    song = out / 'songs' / row['slug']
    (song / 'truth').mkdir(parents=True)
    for name, content in {'attacks.htdemucs.npy': b'attacks', 'drums.htdemucs.opus': b'stem',
                          'activations.mix.npy': b'cache', 'truth/truth.json': b'previous labels',
                          'truth/result.json': b'{"complete":true,"signature":"old"}'}.items():
        (song / name).write_bytes(content)
    if with_video:
        (song / 'video.mp4').write_bytes(b'original video')
    with (out / 'manifest.jsonl').open('a') as stream:
        stream.write(json.dumps(row) + '\n')
    return row, song


def tree_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def remote_bundle(tmp_path, dataset, contents):
    path = bundle(tmp_path, dataset, b'{"video":"abcdefghijk","at":100,"status":"compacted"}\n')
    files = []
    for name, data in contents.items():
        target = path / 'songs' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files.append([name, len(data), int(target.stat().st_mtime), transfer.digest(target)])
    (path / 'snapshot.json').write_text(json.dumps({'bytes': sum(row[1] for row in files), 'files': files}))
    return path


def offline_pull(monkeypatch, remote, *, failure=None, during_transfer=None):
    calls = []
    monkeypatch.setattr(controller, 'ssh', lambda command: calls.append(('ssh', command)))
    monkeypatch.setattr(controller.shutil, 'disk_usage', lambda path: SimpleNamespace(free=100e9))

    def rsync(source, destination, *options):
        calls.append(('rsync', source, destination))
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        if '--exclude=songs/' in options:
            for name in ('manifest.jsonl', 'vault.json', 'vault_log.jsonl', 'snapshot.json'):
                shutil.copy2(remote / name, destination / name)
        else:
            assert source.endswith('/songs/')
            shutil.copytree(remote / 'songs', destination, dirs_exist_ok=True)
            if during_transfer:
                during_transfer()
            if failure:
                raise RuntimeError(failure)

    monkeypatch.setattr(controller, 'rsync', rsync)
    return calls


@pytest.mark.parametrize('change', ['edit', 'delete'])
def test_pull_detects_local_divergence_with_hashes_and_explicit_override(dataset, tmp_path, monkeypatch, change):
    name = 'song-abcdefghijk/options.json'
    remote = remote_bundle(tmp_path, dataset, {name: b'original', 'song-abcdefghijk/truth/truth.json': b'old truth'})
    offline_pull(monkeypatch, remote)
    controller.pull(None)
    local = dataset / 'songs' / name
    stat = local.stat()
    if change == 'edit':
        local.write_bytes(b'edited!!')  # same length AND mtime, still a conflict
        transfer.os.utime(local, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    else:
        local.unlink()
    (local.parent / 'local-only.txt').write_text('keep me')
    before = tree_bytes(dataset)
    with pytest.raises(ValueError, match='local song files changed; nothing overwritten'):
        controller.pull(None)
    assert tree_bytes(dataset) == before
    controller.pull(None, overwrite_local=True)
    assert local.read_bytes() == b'original'
    assert (local.parent / 'local-only.txt').read_text() == 'keep me'
    assert transfer.pulled_files()[name] == transfer.digest(local)
    assert not list(dataset.glob('.wsl-pull-*'))


@pytest.mark.parametrize('matches', [False, True])
def test_first_pull_only_bootstraps_matching_existing_files(dataset, tmp_path, monkeypatch, matches):
    name = 'song-abcdefghijk/options.json'
    remote = remote_bundle(tmp_path, dataset, {name: b'remote'})
    local = dataset / 'songs' / name
    local.parent.mkdir(parents=True)
    local.write_bytes(b'remote' if matches else b'local')
    offline_pull(monkeypatch, remote)
    if matches:
        controller.pull(None)
        assert transfer.pulled_files()[name] == transfer.digest(local)
    else:
        before = tree_bytes(dataset)
        with pytest.raises(ValueError, match='overwrite-local'):
            controller.pull(None)
        assert tree_bytes(dataset) == {**before, '.harvest-writer.lock': b''}


def test_pull_refuses_legacy_snapshot_without_hashes(dataset, tmp_path, monkeypatch):
    remote = remote_bundle(tmp_path, dataset, {'song/options.json': b'remote'})
    index = json.loads((remote / 'snapshot.json').read_text())
    index['files'] = [row[:3] for row in index['files']]
    (remote / 'snapshot.json').write_text(json.dumps(index))
    offline_pull(monkeypatch, remote)
    before = tree_bytes(dataset)
    with pytest.raises(ValueError, match='snapshot lacks SHA-256 hashes'):
        controller.pull(None, overwrite_local=True)
    assert tree_bytes(dataset) == {**before, '.harvest-writer.lock': b''}


@pytest.mark.parametrize('failure', ['transfer', 'hash', 'edit_during_transfer'])
def test_pull_staging_failure_never_publishes_song_or_journals(dataset, tmp_path, monkeypatch, failure):
    name = 'song-abcdefghijk/options.json'
    remote = remote_bundle(tmp_path, dataset, {name: b'original', 'other-song/score.json': b'score'})
    offline_pull(monkeypatch, remote)
    controller.pull(None)
    local = dataset / 'songs' / name
    before = tree_bytes(dataset)
    if failure == 'hash':
        (remote / 'songs' / name).write_bytes(b'bad received bytes')
    offline_pull(monkeypatch, remote, failure='fixture transfer failure' if failure == 'transfer' else None,
                 during_transfer=(lambda: local.write_bytes(b'edited')) if failure == 'edit_during_transfer' else None)
    with pytest.raises((RuntimeError, ValueError)):
        controller.pull(None)
    expected = {**before, 'songs/' + name: b'edited'} if failure == 'edit_during_transfer' else before
    assert tree_bytes(dataset) == expected


def test_song_publication_swaps_whole_directory_and_only_updates_selected_index(dataset, tmp_path, monkeypatch):
    contents = {'song-a/options.json': b'old options', 'song-a/truth/truth.json': b'old truth',
                'song-b/score.json': b'unchanged'}
    remote = remote_bundle(tmp_path, dataset, contents)
    offline_pull(monkeypatch, remote)
    controller.pull(None)
    previous_b = transfer.pulled_files()['song-b/score.json']
    # A channel-limited snapshot must retain the other channel's sync baseline.
    (remote / 'songs/song-b/score.json').unlink()
    index = json.loads((remote / 'snapshot.json').read_text())
    index['files'] = [row for row in index['files'] if row[0].startswith('song-a/')]
    for row in index['files']:
        path = remote / 'songs' / row[0]
        path.write_bytes(b'new options' if path.name == 'options.json' else b'new truth')
        row[1] = path.stat().st_size
        row[3] = transfer.digest(path)
    index['bytes'] = sum(row[1] for row in index['files'])
    (remote / 'snapshot.json').write_text(json.dumps(index))
    real_swap = transfer.atomic_song
    seen = []

    def swap(stage, target):
        assert tree_bytes(target) == {'options.json': b'old options', 'truth/truth.json': b'old truth'}
        real_swap(stage, target)
        assert tree_bytes(target) == {'options.json': b'new options', 'truth/truth.json': b'new truth'}
        assert tree_bytes(stage) == {'options.json': b'old options', 'truth/truth.json': b'old truth'}
        seen.append(target.name)

    monkeypatch.setattr(transfer, 'atomic_song', swap)
    controller.pull(['dailychops'])
    assert seen == ['song-a']
    assert transfer.pulled_files()['song-b/score.json'] == previous_b


def staged_push(tmp_path, dataset):
    path = bundle(tmp_path, dataset, b'{"video":"abcdefghijk","at":100}\n')
    for name in ('pyproject.toml', 'uv.lock', 'src/module.py', 'tools/template.png',
                 'inputs/truth/input.json', 'inputs/.eval-cache/ft/song/feat.npy'):
        target = path / (name if name.startswith('inputs/') else 'code/' + name)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'new content')
    return path


def test_push_writer_lock_refusal_preserves_live_code_inputs_and_journals(dataset, tmp_path, monkeypatch):
    root = dataset.parents[1]
    monkeypatch.setattr(transfer, 'ROOT', root)
    path = staged_push(tmp_path, dataset)
    (root / 'tools').mkdir()
    (root / 'tools/template.png').write_bytes(b'old template')
    before = tree_bytes(root)
    monkeypatch.setattr(transfer.subprocess, 'run', lambda *a, **kw: pytest.fail('locked push published files'))
    with harvest.writer_lock():
        with pytest.raises(RuntimeError, match='another harvest.*refusing to publish'):
            transfer.publish_push(path, initialize=True, inputs=True)
    assert tree_bytes(root) == {**before, 'truth/harvest/.harvest-writer.lock': b''}


def test_staged_push_cli_uses_live_writer_lock_without_installed_helpers(dataset, tmp_path):
    root = dataset.parents[1]
    path = staged_push(tmp_path, dataset)
    for name in ('gpu/harvest_transfer.py', 'harvest/vault.py', 'harvest/manifest_io.py'):
        target = path / 'code/tools' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE.parent / name, target)
    before = tree_bytes(root)
    # This really executes the staged CLI on a temp repo, without SSH or rsync.
    with harvest.writer_lock():
        result = subprocess.run([sys.executable, str(path / 'code/tools/gpu/harvest_transfer.py'),
                                 'publish', str(path), '--root', str(root), '--init', '--inputs'],
                                capture_output=True, text=True)
    assert result.returncode != 0
    assert 'another harvest writer/snapshot/comparison is running; refusing to publish' in result.stderr
    assert tree_bytes(root) == {**before, 'truth/harvest/.harvest-writer.lock': b''}


def test_push_validation_precedes_code_and_lock_covers_inputs_rebase_import(dataset, tmp_path, monkeypatch):
    root = dataset.parents[1]
    monkeypatch.setattr(transfer, 'ROOT', root)
    path = staged_push(tmp_path, dataset)
    calls = []

    def publish(argv, **kwargs):
        with pytest.raises(RuntimeError, match='another harvest'):
            with harvest.writer_lock():
                pass
        calls.append(argv)
        if argv[0] == 'rsync':
            shutil.copytree(argv[2], argv[3], dirs_exist_ok=True)
        else:
            assert argv[-1] == '--rebase'
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(transfer.subprocess, 'run', publish)
    transfer.publish_push(path, initialize=True, inputs=True)
    assert len(calls) == 3
    assert (root / 'tools/template.png').read_bytes() == b'new content'
    assert (root / 'truth/input.json').read_bytes() == b'new content'
    assert (dataset / '.wsl-authority').exists()
    assert (dataset / 'manifest.jsonl').read_bytes() == (path / 'manifest.jsonl').read_bytes()
    before = tree_bytes(root)
    (path / 'manifest.jsonl').write_bytes(b'{"video":"localbranch","at":110}\n')
    calls.clear()
    with pytest.raises(ValueError, match='diverged'):
        transfer.publish_push(path, inputs=True)
    assert not calls and tree_bytes(root) == before


@pytest.mark.parametrize('inputs', [False, True])
def test_push_controller_sends_every_file_to_stage_before_publication(dataset, monkeypatch, inputs):
    calls, destinations = [], []
    groups = b'{"groups": [["train", "sealed"]], "source": "offline audio", "generated": "fixture"}\n'
    (dataset / 'same_performance.json').write_bytes(groups)
    sent = []
    monkeypatch.setattr(controller, 'ssh', calls.append)
    def rsync(source, destination, *opts):
        destinations.append(destination)
        if (Path(source) / 'code-parts.txt').is_file():
            path = Path(source) / 'same_performance.json'
            if path.is_file():
                sent.append(path.read_bytes())
        if '/inputs/truth/' in destination:
            assert '--exclude=harvest/same_performance.json' in opts
    monkeypatch.setattr(controller, 'rsync', rsync)
    monkeypatch.setattr(controller.subprocess, 'check_output', lambda *a, **kw: '/external/song\n')
    controller.push(True, inputs)
    assert destinations and all('/.harvest-transfer/' in destination for destination in destinations)
    publish = next(command for command in calls if ' publish ' in command)
    assert '/code/tools/gpu/harvest_transfer.py publish ' in publish
    assert '--root . --init' in publish
    assert ('--inputs' in publish) == inputs
    assert calls[-1].startswith('rm -rf -- .harvest-transfer/')
    assert (dataset / '.wsl-mirror').exists()
    assert sent == [groups]


def test_publish_push_installs_same_performance_and_pull_never_overwrites(dataset, tmp_path, monkeypatch):
    root = dataset.parents[1]
    monkeypatch.setattr(transfer, 'ROOT', root)
    path = staged_push(tmp_path, dataset)
    groups = b'{"groups": [["train", "sealed"]], "source": "Mac owner"}\n'
    (path / 'same_performance.json').write_bytes(groups)
    def publish(argv, **kwargs):
        assert argv[0] == 'rsync'
        shutil.copytree(argv[2], argv[3], dirs_exist_ok=True)
    monkeypatch.setattr(transfer.subprocess, 'run', publish)
    transfer.publish_push(path, initialize=True)
    assert (dataset / 'same_performance.json').read_bytes() == groups
    transfer.publish_mirror(path)
    assert (dataset / 'same_performance.json').read_bytes() == groups
    # Pull refuses owner-table divergence rather than overwriting Mac edits.
    (path / 'same_performance.json').write_text('{"groups": []}')
    before = tree_bytes(root)
    with pytest.raises(ValueError, match='same_performance.json differs'):
        transfer.publish_mirror(path)
    assert tree_bytes(root) == before


@pytest.mark.parametrize('problem', ['malformed', 'missing'])
def test_push_group_table_errors_precede_any_publication(dataset, tmp_path, monkeypatch, problem):
    root = dataset.parents[1]
    monkeypatch.setattr(transfer, 'ROOT', root)
    path = staged_push(tmp_path, dataset)
    (dataset / 'same_performance.json').write_text('{"groups": [["train", "sealed"]]}')
    if problem == 'malformed':
        (path / 'same_performance.json').write_text('{"groups": ["bad"]}')
    before = tree_bytes(root)
    monkeypatch.setattr(transfer.subprocess, 'run', lambda *a, **kw: pytest.fail('bad groups published code'))
    with pytest.raises(ValueError, match='same_performance'):
        transfer.publish_push(path, initialize=True)
    assert tree_bytes(root) == {**before, 'truth/harvest/.harvest-writer.lock': b''}


def final_plan(tmp_path):
    from chops_common import load_performance_table, sha
    plan = {'tags': [{'tag': 'frozen'}]}
    table, digest = load_performance_table()
    plan.update(same_performance=[list(g) for g in table], same_performance_sha256=digest)
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    (tmp_path / 'frozen').mkdir()
    (tmp_path / 'frozen/done.json').write_text(json.dumps({'plan_sha256': sha(tmp_path / 'plan.json')}))
    return plan


def test_compare_final_records_durable_shared_opening_before_first_sealed_read(dataset, tmp_path, monkeypatch):
    import chops_compare as compare
    monkeypatch.setattr(compare, 'ROOT', dataset.parents[1])
    plan = final_plan(tmp_path)

    def read(**kwargs):
        records = [json.loads(line) for line in transfer.vault.LOG.read_text().splitlines()]
        assert len(records) == 1 and records[0]['model_tag'] == 'frozen'
        attempt = json.loads((tmp_path / 'final_once.json').read_text())
        assert attempt['opening_id'] == records[0]['opening_id']
        raise RuntimeError('fixture preparation failure after opening')

    monkeypatch.setattr(compare, 'score_rows', read)
    with pytest.raises(RuntimeError, match='preparation failure'):
        compare.final(plan, 'frozen', tmp_path, 1)
    assert len(transfer.vault.LOG.read_text().splitlines()) == 1
    with pytest.raises(FileExistsError):
        compare.final(plan, 'frozen', tmp_path, 1)


@pytest.mark.parametrize('problem', ['unfinished', 'append_failure', 'mirror'])
def test_compare_final_failed_shared_opening_never_reads_truth(dataset, tmp_path, monkeypatch, problem):
    import chops_compare as compare
    plan = final_plan(tmp_path)
    if problem == 'unfinished':
        transfer.vault.LOG.write_bytes(b'{"timestamp":')
    elif problem == 'mirror':
        (dataset / '.wsl-mirror').touch()
    else:
        def fail(*args):
            raise OSError('fixture vault append failure')
        monkeypatch.setattr(transfer.vault.os, 'write', fail)
    monkeypatch.setattr(compare, 'score_rows', lambda **kw: pytest.fail('failed opening read sealed truth'))
    with pytest.raises((ValueError, OSError, RuntimeError)):
        compare.final(plan, 'frozen', tmp_path, 1)


def test_compare_final_passes_same_opening_id_to_scorer(dataset, tmp_path, monkeypatch):
    import chops_compare as compare
    monkeypatch.setattr(compare, 'ROOT', dataset.parents[1])
    import numpy as np
    import torch
    plan = final_plan(tmp_path)
    folder = tmp_path / 'features'
    folder.mkdir()
    np.save(folder / 'feat.npy', np.zeros((10, 1)))
    monkeypatch.setattr(compare, 'score_rows', lambda **kw: [{'slug': 'sealed'}])
    monkeypatch.setattr(compare, 'feature', lambda song: folder)
    monkeypatch.setattr(compare, 'checkpoint', lambda *args: Path('fake-checkpoint'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: None)
    monkeypatch.setattr(torch, 'set_num_threads', lambda count: None)
    monkeypatch.setattr(torch, 'load', lambda *a, **kw: {})
    monkeypatch.setattr(transfer.vault.os, 'getppid', transfer.vault.os.getpid)
    model = SimpleNamespace(load_state_dict=lambda state: None, eval=lambda: None)
    monkeypatch.setitem(sys.modules, 'train', SimpleNamespace(
        original_model=lambda device: model, run=lambda *args: np.zeros((10, 5))))
    commands = []

    def score(argv, log, tag):
        opening_id = argv[argv.index('--opening-id') + 1]
        transfer.vault.resume_final(opening_id, argv[argv.index('--model-tag') + 1])
        assert opening_id == json.loads((tmp_path / 'final_once.json').read_text())['opening_id']
        commands.append(argv)

    monkeypatch.setattr(compare, 'command', score)
    compare.final(plan, 'frozen', tmp_path, 4)
    assert len(commands) == 1
    assert len(list(folder.glob('student-frozen_final-fold*.npy'))) == 3
    assert len(transfer.vault.LOG.read_text().splitlines()) == 1


def test_pull_cli_forwards_explicit_local_override(monkeypatch):
    calls = []
    monkeypatch.setattr(controller, 'pull', lambda only, overwrite_local: calls.append((only, overwrite_local)))
    monkeypatch.setattr(sys, 'argv', ['harvest_wsl.py', 'pull', '--only', 'dailychops', '--overwrite-local'])
    controller.main()
    assert calls == [(['dailychops'], True)]


def successful_read(monkeypatch, calls):
    def process(root, row, profile, profile_path, timeout):
        calls.append(row['video'])
        assert row['status'] == 'fetched'
        assert reader.batch.manifest_read(root / 'manifest.jsonl')[row['video']]['status'] == 'fetched'
        song = reader.batch.song_path(root, row)
        assert not (song / 'truth').exists()  # even identical inputs get a new attempt
        assert list(song.glob('truth.prev-*'))
        (song / 'truth').mkdir()
        (song / 'truth/truth.json').write_text('new labels')
        reader.batch.manifest_append(root / 'manifest.jsonl', video=row['video'], status='read')
        return {'complete': True}

    monkeypatch.setattr(reader.batch, 'process', process)


def test_reread_selection_uses_latest_status_channels_and_ids(reread_dataset, capsys):
    out = reread_dataset
    a, _ = reread_song(out, status='read')
    b, _ = reread_song(out, video='lmnopqrstuv', status='unreadable')
    reread_song(out, video='newvideo001', status='fetched')
    reread_song(out, video='newvideo002', status='compacted')
    reread_song(out, video='otherchan01', channel='dailychops', status='read')
    reader.batch.manifest_append(out / 'manifest.jsonl', video=a['video'], status='failed')
    # A previously failed row is eligible once its LATEST status is fetched.
    reader.batch.manifest_append(out / 'manifest.jsonl', video=b['video'], status='failed')
    reader.batch.manifest_append(out / 'manifest.jsonl', video=b['video'], status='fetched')
    selected = reader.reread_rows(out, ['moyin'], None)
    assert [row['video'] for row in selected] == ['lmnopqrstuv', 'newvideo001', 'newvideo002']
    assert 'skipped latest status: failed' in capsys.readouterr().out
    assert reader.reread_rows(out, ['moyin'], ['lmnopqrstuv', 'lmnopqrstuv']) == [selected[0]]
    with pytest.raises(ValueError, match='IDs missing'):
        reader.reread_rows(out, ['moyin'], ['otherchan01'])


def test_reread_dry_run_is_offline_and_preserves_all_bytes(reread_dataset, monkeypatch, capsys):
    out = reread_dataset
    reread_song(out)
    reread_song(out, video='lmnopqrstuv', status='read', with_video=True)
    before = tree_bytes(out)
    monkeypatch.setattr(reader.batch, 'process', lambda *args: pytest.fail('dry-run called reader'))
    monkeypatch.setattr(reader.batch, 'report', lambda *args: pytest.fail('dry-run wrote report'))
    monkeypatch.setattr(harvest, 'executable', lambda *args: pytest.fail('dry-run looked up yt-dlp'))
    monkeypatch.setattr(sys, 'argv', ['harvest_read', '--reread', '--only', 'moyin',
                                    '--max-height', '1080', '--dry-run'])
    reader.main()
    assert tree_bytes(out) == before
    output = capsys.readouterr().out
    assert 'abcdefghijk compacted: re-read' in output
    assert 'lmnopqrstuv read: re-read; reuse existing video' in output
    assert '2 songs; 1 video downloads; estimated download total 0.270 GB' in output


def test_reread_video_only_preserves_journal_first_row_and_seal(reread_dataset, monkeypatch):
    out = reread_dataset
    row, song = reread_song(out)
    first = (out / 'manifest.jsonl').read_bytes()
    before_audio = {name: (song / name).read_bytes() for name in
                    ('attacks.htdemucs.npy', 'drums.htdemucs.opus', 'activations.mix.npy')}
    config = (out / 'vault.json').read_bytes()
    assert transfer.vault.Vault().is_vault(row['video'])  # moyin is outside train_channels
    reads, downloads, heights = [], [], []
    successful_read(monkeypatch, reads)

    def video_format(height):
        heights.append(height)
        return f'shared-format-{height}'

    def run(argv, **kwargs):
        downloads.append(argv)
        assert argv[0] == '/fake/yt-dlp'
        assert argv[argv.index('-f') + 1] == 'shared-format-1080'
        assert kwargs['env'] == harvest.command_env()
        Path(argv[argv.index('-o') + 1]).write_bytes(b'redownloaded video')
        return subprocess.CompletedProcess(argv, 0, '', '')

    monkeypatch.setattr(harvest, 'video_format', video_format)
    monkeypatch.setattr(reader.subprocess, 'run', run)
    reader.reread(out, ['moyin'], [row['video']], 1080, False, 123, 10)
    assert reads == [row['video']] and len(downloads) == 1 and heights == [1080]
    assert (song / 'video.mp4').read_bytes() == b'redownloaded video'
    assert not list(song.glob('.reread-video-*'))
    assert (song / 'truth/truth.json').read_text() == 'new labels'
    assert len(list(song.glob('truth.prev-*'))) == 1
    assert next(song.glob('truth.prev-*/truth.json')).read_bytes() == b'previous labels'
    assert before_audio == {name: (song / name).read_bytes() for name in before_audio}
    journal = (out / 'manifest.jsonl').read_bytes()
    assert journal.startswith(first)
    assert [json.loads(line)['status'] for line in journal.splitlines()] == ['compacted', 'fetched', 'read']
    seal = transfer.vault.Vault()
    assert seal.first[row['video']] == 200 and seal.is_vault(row['video'])
    assert (out / 'vault.json').read_bytes() == config
    assert not (out / 'vault_log.jsonl').exists()
    harvest.compact(False, ['moyin'])
    assert not (song / 'video.mp4').exists()
    assert next(song.glob('truth.prev-*/truth.json')).read_bytes() == b'previous labels'
    assert reader.batch.manifest_read(out / 'manifest.jsonl')[row['video']]['status'] == 'compacted'
    assert (out / 'manifest.jsonl').read_bytes().startswith(first)
    assert transfer.vault.Vault().first[row['video']] == 200
    assert transfer.vault.Vault().is_vault(row['video'])


@pytest.mark.parametrize('status', ['read', 'unreadable', 'fetched', 'compacted'])
def test_reread_reuses_existing_video_and_accepts_explicit_seal(reread_dataset, monkeypatch, status):
    out = reread_dataset
    row, song = reread_song(out, channel='dailychops', status=status, with_video=True)
    config = json.loads((out / 'vault.json').read_text())
    config['video_ids'] = [row['video']]
    (out / 'vault.json').write_text(json.dumps(config))
    reads = []
    successful_read(monkeypatch, reads)
    reader.reread(out, ['dailychops'], None, 2160, False, 1200, 10)
    assert reads == [row['video']]
    assert (song / 'video.mp4').read_bytes() == b'original video'
    assert transfer.vault.Vault().is_vault(row['video'])


@pytest.mark.parametrize('outcome', ['success', 'gate_rejection', 'worker_failure'])
def test_reread_real_batch_forces_new_attempt_and_retains_worse_truth(reread_dataset, monkeypatch, outcome):
    out = reread_dataset
    row, song = reread_song(out, status='read', with_video=True)
    profile, _ = reader.batch.channel_profile('moyin')
    old = {'complete': True, 'signature': reader.batch.signature(song, profile),
           'scoring_systems': 1, 'scoring_notes': 2}
    (song / 'truth/result.json').write_text(json.dumps(old))
    worker_calls = []

    def worker(argv, **kwargs):
        worker_calls.append(argv)
        assert '--worker' in argv and '--reuse' not in argv
        assert argv[:2] == [sys.executable, str(reader.batch.HERE / 'batch.py')]
        assert kwargs['timeout'] == 42
        attempt = Path(argv[argv.index('--worker') + 2])
        if outcome == 'worker_failure':
            return subprocess.CompletedProcess(argv, 1)
        notes = int(outcome == 'success')
        result = {'complete': True, 'signature': reader.batch.signature(song, profile),
                  'strict_systems': notes, 'scoring_systems': notes,
                  'strict_seconds': notes, 'scoring_seconds': notes,
                  'strict_notes': notes, 'scoring_notes': notes,
                  'rejection_counts': {} if notes else {'ordinary gate': 1}}
        (attempt / 'result.json').write_text(json.dumps(result))
        (attempt / 'truth.json').write_text('[]')
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(reader.subprocess, 'run', worker)
    if outcome == 'worker_failure':
        with pytest.raises(SystemExit, match='did not complete'):
            reader.reread(out, ['moyin'], None, 720, False, 42, 10)
    else:
        reader.reread(out, ['moyin'], None, 720, False, 42, 10)
    assert len(worker_calls) == 1  # identical successful fingerprint did not skip the reader
    result = json.loads((song / 'truth/result.json').read_text())
    assert result['complete'] == (outcome != 'worker_failure')
    assert reader.batch.manifest_read(out / 'manifest.jsonl')[row['video']]['status'] == (
        'read' if outcome == 'success' else 'unreadable')
    assert json.loads(next(song.glob('truth.prev-*/result.json')).read_text()) == old
    assert next(song.glob('truth.prev-*/truth.json')).read_bytes() == b'previous labels'
    assert (song / 'video.mp4').exists()
    assert not (song / 'truth-history').exists()  # no duplicate archive
    harvest.compact(False, ['moyin'])
    assert (song / 'video.mp4').exists() == (outcome != 'success')
    assert reader.batch.manifest_read(out / 'manifest.jsonl')[row['video']]['status'] == (
        'compacted' if outcome == 'success' else 'unreadable')
    assert next(song.glob('truth.prev-*/truth.json')).read_bytes() == b'previous labels'


@pytest.mark.parametrize('failure', ['exit', 'empty', 'timeout'])
def test_reread_download_retries_fail_without_changing_truth_or_manifest(reread_dataset, monkeypatch, failure):
    out = reread_dataset
    row, song = reread_song(out)
    before = tree_bytes(out)
    calls, sleeps = [], []

    def run(argv, **kwargs):
        calls.append(argv)
        target = Path(argv[argv.index('-o') + 1])
        if failure == 'timeout':
            target.write_bytes(b'partial')
            raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
        target.write_bytes(b'partial' if failure == 'exit' else b'')
        return subprocess.CompletedProcess(argv, int(failure == 'exit'), '', 'fixture error')

    monkeypatch.setattr(reader.subprocess, 'run', run)
    monkeypatch.setattr(reader.time, 'sleep', sleeps.append)
    with pytest.raises(RuntimeError, match='video download failed after 3 attempts'):
        reader.reread(out, ['moyin'], [row['video']], 720, False, 1200, 10)
    assert len(calls) == 3 and sleeps == [20, 40]
    assert len({argv[argv.index('-o') + 1] for argv in calls}) == 3
    assert tree_bytes(out) == {**before, f'songs/{song.name}/.truth.lock': b''}


def test_reread_download_retry_can_succeed(reread_dataset, monkeypatch):
    out = reread_dataset
    row, song = reread_song(out)
    calls, reads = [], []
    successful_read(monkeypatch, reads)

    def run(argv, **kwargs):
        calls.append(argv)
        Path(argv[argv.index('-o') + 1]).write_bytes(b'video')
        return subprocess.CompletedProcess(argv, int(len(calls) < 3), '', 'retry fixture')

    monkeypatch.setattr(reader.subprocess, 'run', run)
    reader.reread(out, ['moyin'], None, 720, False, 1200, 10)
    assert len(calls) == 3 and reads == [row['video']]
    assert (song / 'video.mp4').read_bytes() == b'video'


@pytest.mark.parametrize('on_retry', [False, True])
def test_reread_reserve_checked_before_each_download(reread_dataset, monkeypatch, on_retry):
    out = reread_dataset
    row, song = reread_song(out)
    first = (out / 'manifest.jsonl').read_bytes()
    free = [100e9 if on_retry else 10.1e9]  # 120s @720 => .12 GB; .24 GB workspace
    calls = []
    monkeypatch.setattr(reader.shutil, 'disk_usage', lambda path: SimpleNamespace(free=free[0]))

    def run(argv, **kwargs):
        calls.append(argv)
        free[0] = 10.1e9
        return subprocess.CompletedProcess(argv, 1, '', 'retry fixture')

    monkeypatch.setattr(reader.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='disk reserve reached'):
        reader.reread(out, ['moyin'], None, 720, False, 1200, 10)
    assert len(calls) == int(on_retry)
    assert (out / 'manifest.jsonl').read_bytes() == first
    assert (song / 'truth/truth.json').read_bytes() == b'previous labels'
    assert not list(song.glob('truth.prev-*')) and not (song / 'video.mp4').exists()


def test_reread_append_failure_restores_canonical_truth(reread_dataset, monkeypatch):
    out = reread_dataset
    _, song = reread_song(out, with_video=True)
    first = (out / 'manifest.jsonl').read_bytes()

    def fail(*args, **kwargs):
        raise ValueError('manifest has an unfinished final record')

    monkeypatch.setattr(reader.batch, 'manifest_append', fail)
    with pytest.raises(ValueError, match='unfinished'):
        reader.reread(out, ['moyin'], None, 720, False, 1200, 10)
    assert (song / 'truth/truth.json').read_bytes() == b'previous labels'
    assert not list(song.glob('truth.prev-*'))
    assert (out / 'manifest.jsonl').read_bytes() == first


@pytest.mark.parametrize('problem', ['disabled', 'missing_attacks', 'missing_song', 'unsafe_slug', 'symlink'])
def test_reread_preflight_fails_without_downloads_or_truth_changes(reread_dataset, monkeypatch, problem):
    out = reread_dataset
    row, song = reread_song(out)
    if problem == 'disabled':
        monkeypatch.setattr(reader.batch, 'channel_profile', lambda channel: ({'enabled': False}, None))
    elif problem == 'missing_attacks':
        (song / 'attacks.htdemucs.npy').unlink()
    elif problem == 'missing_song':
        row['slug'] = 'missing-song'
    elif problem == 'unsafe_slug':
        row['slug'] = '../escape'
    else:
        (song / 'video.mp4').symlink_to(song / 'attacks.htdemucs.npy')
    if problem in ('missing_song', 'unsafe_slug'):
        (out / 'manifest.jsonl').write_text(json.dumps(row) + '\n')
    before = tree_bytes(out)
    with pytest.raises((ValueError, FileNotFoundError)):
        reader.reread(out, ['moyin'], None, 720, False, 1200, 10)
    assert tree_bytes(out) == before


def test_reread_song_lock_and_authority_guards(reread_dataset):
    out = reread_dataset
    _, song = reread_song(out, with_video=True)
    with reader.batch.lock(song / '.truth.lock') as acquired:
        assert acquired
        with pytest.raises(RuntimeError, match='song lock'):
            reader.reread(out, ['moyin'], None, 720, False, 1200, 10)
    assert not list(song.glob('truth.prev-*'))
    with transfer.lock():
        with pytest.raises(RuntimeError, match='another harvest'):
            with harvest.writer_lock():
                pass
    (out / '.wsl-authority').unlink()
    with pytest.raises(RuntimeError, match='WSL authority'):
        reader.reread(out, ['moyin'], None, 720, True, 1200, 10)
    (out / '.wsl-authority').touch()
    (out / '.wsl-mirror').touch()
    with pytest.raises(RuntimeError, match='WSL authority'):
        reader.reread(out, ['moyin'], None, 720, True, 1200, 10)


@pytest.mark.parametrize('args', [[], ['--reread'], ['--reread', '--latest-fetch', '--only', 'moyin'],
                                  ['--latest-fetch', '--ids', 'abcdefghijk'],
                                  ['--latest-fetch', '--dry-run'], ['--latest-fetch', '--max-height', '1080'],
                                  ['--reread', '--only', 'moyin', '--max-height', '100'],
                                  ['--reread', '--only', 'moyin', '--timeout', 'nan'],
                                  ['--reread', '--only', 'moyin', '--min-free-gb', '-1'],
                                  ['--latest-fetch', '--jobs', '0'],
                                  ['--reread', '--only', 'moyin', '--jobs', '-2'],
                                  ['--latest-fetch', '--jobs', '1.5']])
def test_read_cli_rejects_invalid_combinations(monkeypatch, args):
    monkeypatch.setattr(sys, 'argv', ['harvest_read', *args])
    with pytest.raises(SystemExit) as exit:
        reader.main()
    assert exit.value.code == 2


@pytest.mark.parametrize('duration', [None, 'NA', -1, float('nan'), float('inf')])
def test_reread_estimate_has_finite_unknown_duration_fallback(duration):
    assert reader.download_estimate({'seconds': duration}, 1080) == 1_080_000_000


def test_latest_fetch_selection_and_already_attempted_regression(reread_dataset, monkeypatch):
    out = reread_dataset
    fetched, _ = reread_song(out, status='fetched', with_video=True)
    attempted, _ = reread_song(out, video='lmnopqrstuv', status='read', with_video=True)
    other, _ = reread_song(out, video='newvideo001', channel='dailychops', status='fetched', with_video=True)
    reread_song(out, video='newvideo002', status='fetched', with_video=True)  # older pending
    logs = reader.ROOT / 'logs/harvest'
    logs.mkdir(parents=True)
    (logs / 'fetch.json').write_text(json.dumps({'exit': 0, 'videos': [fetched['video'], attempted['video'], other['video']]}))
    (logs / 'latest_fetch.txt').write_text('logs/harvest/fetch.json\n')
    calls = []
    monkeypatch.setattr(reader.batch, 'process', lambda root, row, *args: calls.append(row['video']) or {'complete': True})
    monkeypatch.setattr(sys, 'argv', ['harvest_read', '--latest-fetch', '--only', 'moyin'])
    reader.main()
    assert calls == [fetched['video']]
    assert not list(out.glob('songs/*/truth.prev-*'))
    (logs / 'fetch.json').write_text(json.dumps({'exit': 1, 'videos': [fetched['video']]}))
    with pytest.raises(ValueError, match='did not finish successfully'):
        reader.main()
    assert calls == [fetched['video']]


@pytest.mark.parametrize('flags,worker', [(['--reread', '--only', 'moyin', '--dry-run', '--jobs', '6'], 'tools/gpu/harvest_read.py'),
                                        (['--latest-fetch', '--jobs', '8'], 'tools/gpu/harvest_read.py'),
                                        (['--only', 'dailychops', '--limit', '2'], 'tools/truth_video/batch.py')])
def test_read_controller_keeps_detached_job_and_shared_lock(monkeypatch, flags, worker):
    calls = []
    monkeypatch.setattr(controller, 'ssh', calls.append)
    controller.detached('read', flags)
    assert len(calls) == 1
    command = shlex.split(calls[0])
    assert command[command.index('setsid'):command.index('nohup') + 1] == ['setsid', '-f', 'nohup']
    assert 'tools/gpu/harvest_job.py' in command
    assert command[command.index('--lock') + 1] == '--'
    argv = command[command.index('--') + 1:command.index('>')]
    assert argv == ['.venv/bin/python', '-u', worker, '--min-free-gb', '10', *flags]
    assert command[-3:] == ['2>&1', '<', '/dev/null']
    assert '&' not in command and '--wait' not in command


def latest_fetch(out, videos):
    logs = out.parents[1] / 'logs/harvest'
    logs.mkdir(parents=True, exist_ok=True)
    (logs / 'fetch.json').write_text(json.dumps({'exit': 0, 'videos': videos}))
    (logs / 'latest_fetch.txt').write_text('logs/harvest/fetch.json\n')


def run_read_cli(monkeypatch, mode, jobs):
    monkeypatch.setattr(sys, 'argv', ['harvest_read', '--' + mode, '--only', 'moyin',
                                    '--jobs', str(jobs), '--timeout', '42'])
    reader.main()


@pytest.mark.parametrize('mode', ['reread', 'latest-fetch'])
def test_parallel_real_batch_isolates_logs_and_appends_committed_rows(reread_dataset, monkeypatch, mode):
    out = reread_dataset
    songs = [reread_song(out, video=f'video{i:06d}', status='fetched', with_video=True)
             for i in range(6)]
    first = (out / 'manifest.jsonl').read_bytes()
    latest_fetch(out, [row['video'] for row, _ in songs])
    together = Barrier(3)
    counters = Lock()
    active = peak = 0
    calls, reports = [], []

    def worker(argv, **kwargs):
        nonlocal active, peak
        song, attempt, snapshot = map(Path, argv[argv.index('--worker') + 1:][:3])
        profile = json.loads(snapshot.read_text())
        video = song.name.removeprefix('song-')
        assert kwargs['timeout'] == 42
        assert kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'
        assert (song / '.truth.lock').exists()
        assert ('--reuse' in argv) == (mode == 'latest-fetch')
        kwargs['stdout'].write(video + '\n')
        with counters:
            calls.append(attempt)
            active += 1
            peak = max(peak, active)
        try:
            together.wait(timeout=5)  # would fail if readers were still serial
            notes = int(video != 'video000001')  # ordinary gate rejection completes
            result = dict(complete=True, signature=reader.batch.signature(song, profile),
                          strict_systems=notes, scoring_systems=notes, systems=1,
                          strict_seconds=notes, scoring_seconds=notes,
                          strict_notes=notes, scoring_notes=notes,
                          strict_labels={}, scoring_labels={},
                          rejection_counts={} if notes else {'ordinary gate': 1})
            (attempt / 'result.json').write_text(json.dumps(result))
            (attempt / 'truth.json').write_text('[]')
        finally:
            with counters:
                active -= 1
        return subprocess.CompletedProcess(argv, 0)

    def report(root):
        assert active == 0 and len(calls) == 6
        assert len(reader.batch.manifest_read(root / 'manifest.jsonl')) == 6
        reports.append(root)

    monkeypatch.setattr(reader.subprocess, 'run', worker)
    monkeypatch.setattr(reader.batch, 'report', report)
    run_read_cli(monkeypatch, mode, 3)
    assert peak == 3 and len(set(calls)) == 6
    assert reports == [out]
    journal = (out / 'manifest.jsonl').read_bytes()
    assert journal.startswith(first) and journal.endswith(b'\n')
    committed = [json.loads(line) for line in journal.splitlines()]
    assert len(committed) == (18 if mode == 'reread' else 12)
    for row, song in songs:
        expected = 'unreadable' if row['video'] == 'video000001' else 'read'
        assert reader.batch.manifest_read(out / 'manifest.jsonl')[row['video']]['status'] == expected
        assert (song / 'truth/run.log').read_text() == row['video'] + '\n'
        assert not list(song.glob('.truth-work-*'))
        if mode == 'reread':
            assert next(song.glob('truth.prev-*/truth.json')).read_bytes() == b'previous labels'
        else:
            assert next(song.glob('truth-history/*/truth.json')).read_bytes() == b'previous labels'


@pytest.mark.parametrize('mode', ['reread', 'latest-fetch'])
def test_parallel_failure_stops_admission_and_joins_running_songs(reread_dataset, monkeypatch, mode):
    out = reread_dataset
    songs = [reread_song(out, video=f'video{i:06d}', status='fetched', with_video=True)
             for i in range(5)]
    latest_fetch(out, [row['video'] for row, _ in songs])
    started = Barrier(2)
    stopped = Event()
    monkeypatch.setattr(reader, 'Event', lambda: stopped)
    finished, calls = [], []

    def process(root, row, *args):
        calls.append(row['video'])
        started.wait(timeout=5)
        if row['video'] == 'video000000':
            raise ValueError('fixture reader failure')
        assert stopped.wait(timeout=5)  # scheduler has actually published failure
        finished.append(row['video'])
        return {'complete': True}

    monkeypatch.setattr(reader.batch, 'process', process)
    monkeypatch.setattr(reader.batch, 'report', lambda root: pytest.fail('failed run wrote report'))
    with pytest.raises(ValueError, match='reader fail'):
        run_read_cli(monkeypatch, mode, 2)
    assert set(calls) == {'video000000', 'video000001'}
    assert finished == ['video000001']  # executor joined this song before raising
    for _, song in songs[2:]:
        assert (song / 'truth/truth.json').read_bytes() == b'previous labels'
        assert not list(song.glob('truth.prev-*'))


@pytest.mark.parametrize('mode', ['reread', 'latest-fetch'])
@pytest.mark.parametrize('failure', ['incomplete', 'none'])
def test_incomplete_song_does_not_stop_the_run(reread_dataset, monkeypatch, mode, failure):
    # batch records a per-video reader error as unreadable; the other songs are still read.
    out = reread_dataset
    songs = [reread_song(out, video=f'video{i:06d}', status='fetched', with_video=True)
             for i in range(5)]
    latest_fetch(out, [row['video'] for row, _ in songs])
    calls, reports = [], []

    def process(root, row, *args):
        calls.append(row['video'])
        if row['video'] == 'video000000':
            return None if failure == 'none' else {'complete': False}
        return {'complete': True}

    monkeypatch.setattr(reader.batch, 'process', process)
    monkeypatch.setattr(reader.batch, 'report', lambda root: reports.append(root))
    with pytest.raises(SystemExit, match='1 song'):
        run_read_cli(monkeypatch, mode, 2)
    assert sorted(calls) == [row['video'] for row, _ in songs]
    assert len(reports) == 1


def test_scheduler_fills_free_slot_without_waiting_for_slowest_song():
    first_running, release_first = Event(), Event()
    calls = []

    def read_one(row):
        calls.append(row['video'])
        if row['video'] == 0:
            first_running.set()
            assert release_first.wait(timeout=5)
        else:
            assert first_running.wait(timeout=5)
            if row['video'] == 3:
                release_first.set()

    reader.read_songs([{'video': i} for i in range(4)], read_one, 2)
    assert calls == [0, 1, 2, 3]


def test_parallel_reread_downloads_are_serial_while_readers_overlap(reread_dataset, monkeypatch):
    out = reread_dataset
    songs = [reread_song(out, video=f'video{i:06d}') for i in range(4)]
    download_lock = Lock()
    together = Barrier(2)
    downloads, reads = [], []
    successful_read(monkeypatch, reads)
    real_process = reader.batch.process

    def download(song, row, height, min_free_gb):
        assert download_lock.acquire(blocking=False)
        try:
            assert height == 1080 and min_free_gb == 20
            downloads.append(row['video'])
            (song / 'video.mp4').write_bytes(b'downloaded')
        finally:
            download_lock.release()

    def process(*args):
        # The next song can be prepared/downloaded while this reader is active.
        together.wait(timeout=5)
        return real_process(*args)

    monkeypatch.setattr(reader, 'download_video', download)
    monkeypatch.setattr(reader.batch, 'process', process)
    reader.reread(out, ['moyin'], None, 1080, False, 42, 20, jobs=2)
    assert set(downloads) == set(reads) == {row['video'] for row, _ in songs}
    assert len(downloads) == len(reads) == 4


def test_parallel_disk_reserve_failure_drains_active_reader(reread_dataset, monkeypatch):
    out = reread_dataset
    songs = [reread_song(out, video=f'video{i:06d}', status='fetched', with_video=True)
             for i in range(4)]
    latest_fetch(out, [row['video'] for row, _ in songs])
    running, stopped = Event(), Event()
    monkeypatch.setattr(reader, 'Event', lambda: stopped)
    calls, finished = [], []
    checks = 0
    counter = Lock()

    def disk(path):
        nonlocal checks
        with counter:
            checks += 1
            if checks == 1:
                return SimpleNamespace(free=100e9)
        # Keep later reserve checks pending until the first reader is active.
        assert running.wait(timeout=5)
        return SimpleNamespace(free=9e9)

    def process(root, row, *args):
        calls.append(row['video'])
        running.set()
        assert stopped.wait(timeout=5)
        finished.append(row['video'])
        return {'complete': True}

    monkeypatch.setattr(reader.shutil, 'disk_usage', disk)
    monkeypatch.setattr(reader.batch, 'process', process)
    with pytest.raises(RuntimeError, match='disk reserve reached'):
        run_read_cli(monkeypatch, 'latest-fetch', 2)
    assert len(calls) == 1 and finished == calls
    for _, song in songs[2:]:
        assert (song / 'truth/truth.json').read_bytes() == b'previous labels'
        assert not list(song.glob('truth.prev-*'))


def test_job_wrapper_holds_global_lock_through_parallel_read_and_report(reread_dataset, monkeypatch):
    out = reread_dataset
    songs = [reread_song(out, video=f'video{i:06d}', status='fetched', with_video=True)
             for i in range(2)]
    latest_fetch(out, [row['video'] for row, _ in songs])
    together = Barrier(2)
    reports = []

    def locked():
        with pytest.raises(RuntimeError, match='another harvest'):
            with harvest.writer_lock():
                pass

    def process(*args):
        locked()
        together.wait(timeout=5)
        locked()
        return {'complete': True}

    def report(root):
        locked()
        reports.append(root)

    def run(command, **kwargs):
        assert command == ['fixture-reader']
        assert kwargs['stdin'] == subprocess.DEVNULL and kwargs['close_fds'] is True
        locked()
        run_read_cli(monkeypatch, 'latest-fetch', 2)
        locked()
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(reader.batch, 'process', process)
    monkeypatch.setattr(reader.batch, 'report', report)
    monkeypatch.setattr(subprocess, 'run', run)
    job = out.parents[1] / 'logs/harvest/read_fixture.json'
    monkeypatch.setattr(sys, 'argv', ['harvest_job', '--job', str(job), '--lock', '--', 'fixture-reader'])
    with pytest.raises(SystemExit) as exit:
        runpy.run_path(str(HERE / 'harvest_job.py'), run_name='__main__')
    assert exit.value.code == 0 and reports == [out]
    assert json.loads(job.read_text())['exit'] == 0
    with harvest.writer_lock():
        pass  # released only after the reader (including its report) returned


@pytest.mark.parametrize('kind', ['read', 'fetch', 'compact'])
def test_detached_launch_shape_disconnects_stdio_and_reports_paths(monkeypatch, capsys, kind):
    calls = []
    monkeypatch.setattr(controller, 'call', lambda argv, **kwargs: calls.append((argv, kwargs)))
    controller.detached(kind, ['--only', 'moyin'])
    assert len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[:len(controller.SSH)] == controller.SSH
    assert kwargs['stdin'] == subprocess.DEVNULL
    remote = shlex.split(argv[-1])
    assert remote[remote.index('setsid'):remote.index('nohup') + 1] == ['setsid', '-f', 'nohup']
    assert '&' not in remote and '--wait' not in remote and '-w' not in remote
    assert remote[-3:] == ['2>&1', '<', '/dev/null']
    job_path = remote[remote.index('--job') + 1]
    log_path = remote[remote.index('>') + 1]
    assert log_path == str(Path(job_path).with_suffix('.log'))
    output = capsys.readouterr().out
    assert f'Job {Path(job_path).stem}' in output
    assert f'log: ~/{controller.REMOTE}/{log_path}' in output
    command = remote[remote.index('--') + 1:remote.index('>')]
    if kind == 'read':
        assert '--lock' in remote
    else:
        # Fetch and compact hold harvest.py's own writer lock; nesting would fail.
        assert '--lock' not in remote
        assert command[:4] == ['.venv/bin/python', '-u', 'tools/harvest/harvest.py', kind]


def test_compact_cli_uses_detached_job_and_still_rejects_force(monkeypatch):
    calls = []
    monkeypatch.setattr(controller, 'detached', lambda kind, args: calls.append((kind, args)))
    monkeypatch.setattr(sys, 'argv', ['harvest_wsl', 'compact', '--only', 'moyin'])
    controller.main()
    assert calls == [('compact', ['--only', 'moyin'])]
    monkeypatch.setattr(sys, 'argv', ['harvest_wsl', 'compact', '--force'])
    with pytest.raises(SystemExit) as exit:
        controller.main()
    assert exit.value.code == 2 and len(calls) == 1
