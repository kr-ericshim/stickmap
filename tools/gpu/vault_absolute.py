"""One logged, absolute evaluation of chopround3 arm A; no selection/training.

Exact WSL commands (run from the WSL writer checkout, never the Mac mirror):

    cd ~/groovemap
    .venv/bin/python -B tools/gpu/vault_absolute.py run --plan scores/chops/chopround3/plan.json --arm A --dry-run
    .venv/bin/python -B tools/gpu/vault_absolute.py run --plan scores/chops/chopround3/plan.json --arm A

Dry-run is read-only on every host: metadata and frozen checkpoint hashes only.
It checks plan structure using check_plan with payload hashes/feature stamps
deferred; the real preflight checks every frozen development input. No sealed
truth, notation, media or activations are read until open_final has fsynced its
log row. A failed real attempt consumes once.json; there is no retry/resume CLI.
Canonical CV and scoring are direct children and verify the runner's opening.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
# librosa's numba JIT cache must land in a guard-allowed folder: under the write guard the default
# __pycache__ locator is refused ("cannot cache function ... no locator available"). Set before numba loads;
# direct children inherit it through the environment.
NUMBA_CACHE = HERE.parents[1] / '.eval-cache/numba-vault-absolute'
os.environ['NUMBA_CACHE_DIR'] = str(NUMBA_CACHE)
sys.path.insert(0, str(HERE))
import chops_compare as compare
from chops_common import vault
from vault_absolute_io import guard, hashes, load, safe, sha, sync_directory, tree, verify, write
# Bind harvest's flat score module before the vision reader puts its own
# identically named score.py on sys.path. This import reads no song payloads.
sys.path.insert(0, str(HERE.parent / 'harvest'))
import score as harvest_score

ROOT = compare.ROOT
SPLIT = ROOT / '.eval-cache/chops_split/split_final_olliewoods_shorts.json'
EXCLUSIONS = vault.OUT / 'vault_final_exclusions.json'
VAULT_TRUTH = ROOT / 'vault_truth/olliewoods_shorts'
RELEASE = Path('/proc/sys/kernel/osrelease')
EXPECTED_OLLIE = 30
CLASSES = ('kick', 'snare', 'tom')
# Earlier openings of this same evaluation that died before any inference or score (no result was seen).
PREVIOUS_FAILURES = {
    'cd068aac75184c65ae983babf3e5a373': 'failed after truth generation, before inference/scoring: the write guard '
    'blocked numba JIT cache writes from librosa ("cannot cache function ... no locator available"); fixed by '
    'NUMBA_CACHE_DIR in a guard-allowed folder (and feature() tmp-file path allowed). Outputs preserved in '
    'scores/chops/chopround3/vault_absolute.attempt1-failed and vault_truth.attempt1/. Rerun approved.',
}
DEVELOPMENT = {'channels': ['moyin', 'dailychops', 'olliewoods', 'alexblack', 'olliewoods_shorts'],
               'strict_held_out_fast_f1_percent': dict(zip(CLASSES, (83.03, 82.42, 82.89))),
               'provenance': 'user supplied frozen chop development result; not recomputed here'}


def require_writer():
    vault.require_final_writer()
    if not RELEASE.is_file() or 'microsoft' not in RELEASE.read_text().lower():
        raise RuntimeError('real absolute evaluation requires WSL authority; use --dry-run on the Mac')


def video_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{11}', value):
        raise ValueError(f'unsafe video ID: {value}')
    return value


def manual_exclusions():
    data = load(safe(EXCLUSIONS, ROOT))
    rows = data.get('exclusions') if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError('manual exclusions must contain an exclusions list')
    result = {}
    for row in rows:
        video = video_id(row['video'])
        if video in result or not isinstance(row.get('reason'), str) or not row['reason'].strip():
            raise ValueError('duplicate manual exclusion or missing reason')
        result[video] = row
    return result


def manifest_rows(seal, manual):
    rows, mixed = vault.filter_final(harvest_score.songs(None, final=True, loose=True), seal=seal, log=False)
    rows = [r for r in rows if not (seal.song_ids(r) & manual.keys())]
    return rows, mixed


def sealed_paths(seal):
    folders = [ROOT / 'vault_truth']
    for video, row in seal.rows.items():
        if seal.is_vault(video) and row.get('slug'):
            folders.append(safe(vault.SONGS / row['slug'], ROOT))
            folders.append(safe(compare.CACHE / row['slug'], ROOT))
            folders.append(safe(ROOT / 'sources' / row['channel'] / video, ROOT))
    for slug, video in seal.slugs.items():
        if seal.is_vault(video):
            folders.append(safe(compare.CACHE / slug, ROOT))
    folders.extend(safe(ROOT / 'sources/olliewoods_shorts' / video_id(v), ROOT)
                   for v in seal.explicit)
    return folders


def runtime_files():
    paths = [p for sub in ('tools/gpu', 'tools/vision_read', 'tools/truth_video', 'tools/finetune',
                          'tools/harvest', 'src/stickmap') for p in (ROOT / sub).rglob('*.py')]
    paths += list((ROOT / 'tools/truth_video/legends').glob('*.json'))
    for name in ('templates', 'harvest_templates'):
        paths += [p for p in (ROOT / 'tools/truth_video' / name).rglob('*')
                  if p.is_file() and p.suffix in ('.png', '.json')]
    return paths


def runtime_preflight():
    """No model inference; missing local runtime must not consume the opening."""
    import torch
    import adtof_pytorch as adtof
    import cv2  # noqa: F401
    import soundfile  # noqa: F401
    from adtof_pytorch.audio import process_audio_file  # noqa: F401
    weights = adtof.get_default_weights_path()
    if not weights or not Path(weights).is_file():
        raise FileNotFoundError('local original ADTOF weights missing; no downloads permitted')
    if shutil.which('ffmpeg') is None:
        raise FileNotFoundError('local ffmpeg missing; no vault opening attempted')


def preflight(plan_path, arm='A', dry_run=False):
    if arm != 'A':
        raise ValueError('absolute evaluation measures only frozen arm A')
    plan_path = safe(plan_path, ROOT)
    plan = load(plan_path)
    name = plan.get('name')
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_]+', name):
        raise ValueError('unsafe/missing plan name')
    directory = safe(ROOT / 'scores/chops' / name / 'vault_absolute', ROOT)
    if plan_path != directory.parent / 'plan.json':
        raise ValueError('plan must be scores/chops/<name>/plan.json')
    if not dry_run and (directory / 'once.json').exists():
        raise FileExistsError(f'once.json already exists; no second vault attempt: {directory / "once.json"}')
    if not dry_run and directory.exists() and any(directory.iterdir()):
        raise ValueError('absolute output contains stale files without once.json; refuse before opening')
    selected = [e for e in plan['tags'] if e.get('arm') == 'A']
    if (len(selected) != 2 or {type(e.get('seed')) for e in selected} != {int} or
            {e['seed'] for e in selected} != {0, 1} or len(plan['folds']) != 2 or
            any(e['tag'] != f'{name}_A_s{e["seed"]}' for e in selected)):
        raise ValueError('A requires exactly name_A_s0/name_A_s1 and folds 0/1')
    if 'E' in plan.get('arms', {}):
        raise ValueError('arm E plans forbid vault evaluation')
    selected.sort(key=lambda e: e['seed'])
    seal = vault.Vault()
    # Even a malformed plan cannot induce a sealed payload hash read here.
    forbidden = sealed_paths(seal)
    with guard(reads_forbidden=forbidden):
        for relative in (*plan['hashes'], *plan.get('feature_stamps', {})):
            safe(ROOT / relative, ROOT)
        structural = {**plan, 'hashes': {}, 'feature_stamps': {}} if dry_run else plan
        compare.check_plan(structural)
        manual = manual_exclusions()
        rows, mixed = manifest_rows(seal, manual)
        split = load(safe(SPLIT, ROOT))
        candidates = split.get('vault')
        if (not isinstance(candidates, list) or len(set(candidates)) != len(candidates) or
                any(video_id(v) != v for v in candidates)):
            raise ValueError('invalid split vault list')
        ids = sorted((set(candidates) & seal.explicit) - set(manual) - set(mixed))
        if len(ids) != EXPECTED_OLLIE:
            raise ValueError(f'expected {EXPECTED_OLLIE} Ollie vault folders, got {len(ids)}')
        if any(v in seal.rows for v in ids):
            raise ValueError('Ollie split IDs must be source-only, absent from manifest (avoid double counting)')
        songs = []
        for row in rows:
            slug = row.get('slug')
            if not isinstance(slug, str) or Path(slug).name != slug or slug in ('.', '..'):
                raise ValueError('invalid manifest slug')
            song = tree(vault.SONGS / slug, ROOT)
            if not song.is_dir():
                raise FileNotFoundError(f'missing manifest vault folder: {song}')
            # Existence only (stat, never opened): scorer/feature inputs must be present before the opening.
            needed = [song / 'activations.mix.npy', song / 'attacks.htdemucs.npy']
            if not any((song / n).is_file() for n in ('drums.htdemucs.wav', 'drums.htdemucs.opus')):
                raise FileNotFoundError(f'missing drum stem: {song}')
            for path in needed:
                if not path.is_file():
                    raise FileNotFoundError(f'missing scoring input: {path}')
            songs.append({'cohort': 'manifest', 'video': row['video'], 'slug': slug,
                          'channel': row['channel'], 'song': song.relative_to(ROOT).as_posix(),
                          'cache': (directory / 'cache/manifest' / slug).relative_to(ROOT).as_posix()})
        for video in ids:
            source = tree(ROOT / 'sources/olliewoods_shorts' / video, ROOT)
            if not source.is_dir():
                raise FileNotFoundError(f'missing Ollie source folder: {source}')
            for filename in ('drums.htdemucs.wav', 'attacks.htdemucs.npy', 'meta.json'):
                if not (source / filename).is_file():
                    raise FileNotFoundError(source / filename)
            from media import video_path
            video_path(source)  # names/existence only, never opens media
            folder = tree(VAULT_TRUTH / video, ROOT)
            if folder.exists() and any(folder.iterdir()):
                raise ValueError(f'vault-only output already exists: {folder}')
            songs.append({'cohort': 'ollie', 'video': video, 'slug': video,
                          'channel': 'olliewoods_shorts', 'source': source.relative_to(ROOT).as_posix(),
                          'song': folder.relative_to(ROOT).as_posix(), 'cache': folder.relative_to(ROOT).as_posix()})
        if len({s['song'] for s in songs}) != len(songs):
            raise ValueError('duplicate scoring song folders')
        checkpoints, metadata = {}, [plan_path, vault.CONFIG, vault.MANIFEST, SPLIT, EXCLUSIONS]
        for experiment in selected:
            done_path = safe(directory.parent / experiment['tag'] / 'done.json', ROOT)
            done = load(done_path)
            if done.get('tag') != experiment['tag'] or done.get('plan_sha256') != sha(plan_path):
                raise ValueError(f'incomplete/stale routine done.json: {experiment["tag"]}')
            expected = {}
            for fold in (0, 1):
                path = safe(compare.checkpoint(plan, experiment['tag'], fold), ROOT)
                expected[path.relative_to(ROOT).as_posix()] = sha(path)
                metadata.append(path.with_suffix('.json'))
            if done.get('checkpoints') != expected:
                raise ValueError(f'checkpoint hashes do not match routine done.json: {experiment["tag"]}')
            checkpoints.update(expected)
            metadata.append(done_path)
        performance = safe(vault.OUT / 'same_performance.json', ROOT)
        if performance.exists():
            metadata.append(performance)
        tree(directory, ROOT)
        return {'plan': plan, 'plan_path': plan_path.relative_to(ROOT).as_posix(), 'arm': 'A',
                'model_tag': f'{name}_A_absolute', 'directory': directory.relative_to(ROOT).as_posix(),
                'songs': songs, 'selected': selected, 'checkpoints': checkpoints,
                'metadata_hashes': hashes(metadata, ROOT),
                'runtime_hashes': {} if dry_run else hashes(runtime_files(), ROOT),
                'absent_metadata': [] if performance.exists() else [performance.relative_to(ROOT).as_posix()],
                'excluded': {'mixed_performance': mixed, 'manual': manual},
                'split_vault_count': len(candidates), 'explicit_intersection_count': len(set(candidates) & seal.explicit),
                'dry_run': dry_run, 'development_payload_hashes_checked': not dry_run}


def verify_metadata(job):
    verify(job['metadata_hashes'], ROOT)
    verify(job['checkpoints'], ROOT)
    verify(job['runtime_hashes'], ROOT)
    if not job['dry_run'] and sorted(p.relative_to(ROOT).as_posix() for p in runtime_files()) != sorted(job['runtime_hashes']):
        raise ValueError('runtime code/legend/template inventory changed after preflight')
    for relative in job['absent_metadata']:
        if safe(ROOT / relative, ROOT).exists():
            raise ValueError(f'metadata added after preflight: {relative}')


def child_job(request, opening_id, model_tag):
    """Called FIRST by direct children, before reading request or sealed media."""
    require_writer()
    vault.resume_final(opening_id, model_tag)
    job = load(safe(request, ROOT))
    directory = safe(ROOT / job['directory'], ROOT)
    if directory != ROOT / 'scores/chops' / job['plan']['name'] / 'vault_absolute':
        raise ValueError('invalid final directory')
    if (job['dry_run'] or job['opening_id'] != opening_id or job['model_tag'] != model_tag or
            model_tag != f'{job["plan"]["name"]}_A_absolute'):
        raise ValueError('request does not match durable opening')
    once = load(directory / 'once.json')
    if once.get('opening_id') != opening_id or once.get('model_tag') != model_tag:
        raise ValueError('request does not match once.json')
    if sha(request) != once.get('request_sha256'):
        raise ValueError('request changed since runner froze it')
    verify_metadata(job)
    return job


def dry_summary(job):
    result = {k: job[k] for k in ('model_tag', 'directory', 'split_vault_count', 'explicit_intersection_count',
                                  'excluded', 'checkpoints', 'development_payload_hashes_checked')}
    result.update(dry_run=True, vault_opened=False, sealed_payloads_read=False,
                  manifest_vault_count=sum(s['cohort'] == 'manifest' for s in job['songs']),
                  ollie_vault_count=sum(s['cohort'] == 'ollie' for s in job['songs']),
                  would_score=[{k: s[k] for k in ('cohort', 'video', 'slug', 'song', 'cache')} for s in job['songs']],
                  truth_coverage='unknown until logged opening; no truth/notation read',
                  authority='real run requires WSL and absence of truth/harvest/.wsl-mirror',
                  ensemble='A only: seed mean within fold0/1, then mean of both fold files')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def command(script, job, opening_id, log):
    request = ROOT / job['directory'] / 'request.json'
    argv = [str(ROOT / '.venv/bin/python'), '-B', str(ROOT / script), '--request', str(request),
            '--opening-id', opening_id, '--model-tag', job['model_tag']]
    env = {k: v for k, v in os.environ.items() if not (k.startswith('FT_') and k.endswith('_SCALE'))}
    env.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    with log.open('w') as stream:
        stream.write(shlex.join(argv) + '\n'); stream.flush()
        subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def truth_snapshot(job):
    folders = [tree(ROOT / s['song'] / 'truth', ROOT) for s in job['songs']]
    files = [p for folder in folders for p in folder.glob('*.json')]
    return {'opening_id': job['opening_id'], 'truth_hashes': hashes(files, ROOT),
            'truth_inventory': {f.relative_to(ROOT).as_posix(): sorted(p.name for p in f.glob('*.json')) for f in folders},
            'absent_audits': [(f / 'kst_audit.json').relative_to(ROOT).as_posix() for f in folders
                              if not (f / 'kst_audit.json').exists()],
            'checkpoints': job['checkpoints'], 'metadata_hashes': job['metadata_hashes'],
            'code_hashes': job['runtime_hashes'], 'songs': job['songs'], 'excluded': job['excluded']}


def verify_inputs(job, snapshot):
    verify_metadata(job)
    for key in ('truth_hashes', 'checkpoints', 'code_hashes', 'audio_hashes'):
        verify(snapshot.get(key, {}), ROOT)
    for relative, names in snapshot['truth_inventory'].items():
        if sorted(p.name for p in tree(ROOT / relative, ROOT).glob('*.json')) != names:
            raise ValueError(f'sealed truth inventory changed: {relative}')


def prepare_audio(job):
    """CPU only; source audio prefers audio.m4a exactly as import_source."""
    import numpy as np
    from adtof_pytorch.audio import process_audio_file
    from media import video_path
    for item in job['songs']:
        song, cache = tree(ROOT / item['song'], ROOT), tree(ROOT / item['cache'], ROOT)
        cache.mkdir(parents=True, exist_ok=True)
        if item['cohort'] == 'manifest':
            # Existing final also writes this one canonical feature cache.
            original = compare.feature(song) / 'feat.npy'
            shutil.copyfile(original, cache / 'feat.npy')
        else:
            source = tree(ROOT / item['source'], ROOT)
            for name in ('drums.htdemucs.wav', 'attacks.htdemucs.npy'):
                shutil.copyfile(source / name, song / name)
            audio = source / 'audio.m4a' if (source / 'audio.m4a').is_file() else video_path(source)
            subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-n', '-i', str(audio), '-vn',
                            '-ar', '44100', '-ac', '2', str(song / 'audio.wav')], check=True)
            feat, _ = process_audio_file(str(song / 'drums.htdemucs.wav'))
            np.save(cache / 'feat.npy', feat[:, :, 0].astype(np.float16))


GPU_BUDGET = 2 * 1024 ** 3  # the GPU is shared with higher-priority jobs: stay within ~2 GB
GPU_MIN_FREE = 3 * 1024 ** 3


def pick_device():
    """CUDA capped at GPU_BUDGET when enough VRAM is free, else CPU (same model, same weights)."""
    import torch
    if not torch.cuda.is_available():
        return 'cpu', 'CUDA unavailable'
    free, total = torch.cuda.mem_get_info()
    if free < GPU_MIN_FREE:
        return 'cpu', f'only {free / 2**30:.1f} GiB VRAM free'
    torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_BUDGET / total))
    return 'cuda', f'{free / 2**30:.1f} GiB free; capped at {GPU_BUDGET / 2**30:.0f} GiB'


def out_of_memory(error):
    import torch
    return isinstance(error, torch.cuda.OutOfMemoryError) or 'out of memory' in str(error).lower()


def threads(device):
    import torch
    torch.set_num_threads(1 if device == 'cuda' else 4)


def mix_activations(job, device):
    """Original-model mix activations for the source-only Ollie folders (scorer input, as import_source)."""
    import numpy as np
    from stickmap import models
    threads(device)
    for item in job['songs']:
        if item['cohort'] == 'ollie':
            song = ROOT / item['song']
            np.save(song / 'activations.mix.npy', models.drum_activations(song / 'audio.wav', device=device))


def model_runs(job, device, fold, songs):
    """Yield (seed_index, item, activation) for frozen A checkpoints on device."""
    import numpy as np
    import torch
    from train import original_model, run
    for seed_index, experiment in enumerate(job['selected']):
        model = original_model(device)
        model.load_state_dict(torch.load(compare.checkpoint(job['plan'], experiment['tag'], fold),
                                        map_location=device, weights_only=True))
        model.eval()
        for item in songs:
            feat = np.load(ROOT / item['cache'] / 'feat.npy', allow_pickle=False)
            try:
                act = run(model, feat, device)
            except RuntimeError as error:
                # As models.drum_activations: cuDNN's RNN kernels refuse very long inputs; PyTorch's own do not.
                if 'CUDNN_STATUS_NOT_SUPPORTED' not in str(error):
                    raise
                with torch.backends.cudnn.flags(enabled=False):
                    act = run(model, feat, device)
            yield seed_index, item, act.astype(np.float32)
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()


def predict(job, device):
    import numpy as np
    threads(device)
    for fold in (0, 1):
        for seed_index, item, act in model_runs(job, device, fold, job['songs']):
            output = ROOT / item['cache'] / f'student-{job["model_tag"]}-fold{fold}.npy'
            act = act / 2
            if seed_index:
                previous = np.load(output, allow_pickle=False)
                if previous.shape != act.shape:
                    raise ValueError('seed activation shape mismatch')
                act += previous
            if act.ndim != 2 or act.shape[1] != 5 or not np.isfinite(act).all():
                raise ValueError('invalid A activations')
            np.save(output, act)


def inference(job):
    """Mix activations + A ensemble; a CUDA OOM reruns the whole phase on CPU so one device made every array."""
    device, reason = pick_device()
    record = {'device': device, 'reason': reason}
    try:
        mix_activations(job, device)
        predict(job, device)
    except RuntimeError as error:
        if device != 'cuda' or not out_of_memory(error):
            raise
        import torch
        torch.cuda.empty_cache()
        record.update(device='cpu', reason=f'CUDA OOM under the {GPU_BUDGET / 2**30:.0f} GiB cap: {error}'[:300])
        mix_activations(job, 'cpu')
        predict(job, 'cpu')
    return record


def device_check(job, record, count=2):
    """When CUDA made the arrays, recompute a few songs on CPU (memory only) and record the difference."""
    import numpy as np
    if record['device'] != 'cuda':
        return {'checked': False, 'reason': 'all inference already on CPU'}
    from stickmap import models
    songs = ([s for s in job['songs'] if s['cohort'] == 'manifest'][:count] +
             [s for s in job['songs'] if s['cohort'] == 'ollie'][:count])
    threads('cpu')
    rows = {s['slug']: {} for s in songs}
    for fold in (0, 1):
        sums = {}
        for seed_index, item, act in model_runs(job, 'cpu', fold, songs):
            sums[item['slug']] = sums.get(item['slug'], 0) + act / 2
        for item in songs:
            gpu = np.load(ROOT / item['cache'] / f'student-{job["model_tag"]}-fold{fold}.npy', allow_pickle=False)
            diff = np.abs(gpu - sums[item['slug']])
            rows[item['slug']][f'student_fold{fold}'] = {'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
                                                         'peak_frames_differ': int(((gpu > .5) != (sums[item['slug']] > .5)).sum())}
    for item in songs:
        if item['cohort'] == 'ollie':
            song = ROOT / item['song']
            gpu = np.load(song / 'activations.mix.npy', allow_pickle=False)
            cpu = models.drum_activations(song / 'audio.wav', device='cpu')
            diff = np.abs(gpu - cpu)
            rows[item['slug']]['mix'] = {'max_abs': float(diff.max()), 'mean_abs': float(diff.mean())}
    return {'checked': True, 'songs': rows,
            'max_abs': max(v['max_abs'] for r in rows.values() for v in r.values())}


def metrics(counts):
    return {subset: {cls: {'counts': [int(counts.get(f'{subset}.{cls}.{key}', 0)) for key in ('tp', 'est', 'ref')],
                           'prf': compare.prf(*[counts.get(f'{subset}.{cls}.{key}', 0) for key in ('tp', 'est', 'ref')])}
                     for cls in CLASSES} for subset in ('all', 'fast')}


def report(result):
    def cell(values):
        return '/'.join('—' if v is None else f'{100*v:.2f}' for v in values)
    lines = ['**결론: 동결한 arm A의 단일 vault 절대 평가를 완료했다.** 레시피·seed 선택이나 학습은 수행하지 않았다.', '',
             f'Opening ID: `{result["opening_id"]}`. Ensemble: seeds 0/1 평균을 fold별로 저장하고 두 fold를 평균했다.', '',
             '| 집단 | 고려한 곡 | strict truth가 있는 곡 | loose notation이 있는 곡 | strict fill 점수에 기여한 곡 | 입력 불량(채점 제외) |',
             '|---|---:|---:|---:|---:|---:|']
    for cohort, coverage in result['coverage'].items():
        lines.append(f'| {cohort} | {coverage["considered"]} | {coverage["with_strict_truth"]} | '
                     f'{coverage["with_notation"]} | {coverage["with_strict_fills"]} | {coverage.get("unusable_inputs", 0)} |')
    lines += ['', 'P/R/F는 백분율이며 곡별 평균이 아니라 정수 TP/est/ref를 합산했다. 분모 0은 —다.', '',
              '| 지표 / 집단 | 부분집합 | Kick P/R/F | Snare P/R/F | Tom P/R/F |',
              '|---|---|---|---|---|']
    for cohort, summary in result['strict'].items():
        for subset in ('all', 'fast'):
            lines.append(f'| strict / {cohort} | {subset} | ' +
                         ' | '.join(cell(summary[subset][c]['prf']) for c in CLASSES) + ' |')
    for scope, summary in result['loose_manifest'].items():
        for subset in ('all', 'fast'):
            lines.append(f'| loose manifest / {scope} | {subset} | ' +
                         ' | '.join(cell(summary[subset][c]['prf']) for c in CLASSES) + ' |')
    lines += ['', '| A 비교 기준: strict held-out fast F1 | Kick | Snare | Tom |', '|---|---:|---:|---:|',
              '| 개발 지표 (모델 고정 시점) | 83.03 | 82.42 | 82.89 |']
    for cohort, summary in result['strict'].items():
        lines.append(f'| vault {cohort} | ' + ' | '.join(cell([summary['fast'][c]['prf'][2]]) for c in CLASSES) + ' |')
    device = result.get('device', {})
    check = device.get('cpu_check', {})
    lines += ['', f'추론 장치: {device.get("device")} ({device.get("reason")}). GPU를 썼다면 일부 곡을 CPU로 다시 계산해 '
              f'최대 절대 차이 {check.get("max_abs", "—")}를 기록했다 (device.json).' if check.get('checked') else
              f'추론 장치: {device.get("device")} ({device.get("reason")}). CPU 재계산 비교는 생략했다: {check.get("reason")}.']
    lines += ['', '개발 채널: ' + ', '.join(DEVELOPMENT['channels']) + '.', '',
              'Strict는 기존 chops_scoring --scope strict와 동일하다: strict truth를 load하고 '
              'gap ≤140ms, ≥4 K/S/T 순간, tom을 포함하는 fill 중 median gap <90ms가 fast다. '
              'all은 이 fill들의 전체이며 곡 전체의 모든 strict note를 뜻하지 않는다.', '',
              'Loose는 fill_seq.notated(0.75), nearest K/S/T neighbour <90ms, '
              'order DP ±60ms다. complete/partial/all_notation은 기존 vault scorer와 동일한 '
              'whole-system audit 범위를 사용한다. strict와 loose의 fast 수치를 직접 비교하지 않는다.', '',
              '제외: final_exclusions()의 mixed-performance 목록과 수동 Eric Moore 목록을 모두 적용했다. '
              '기존 scorer는 vault_final_exclusions.json을 읽지 않으므로 이 스크립트가 직접 적용했다.']
    for kind, rows in result['excluded'].items():
        for video, item in sorted(rows.items()):
            lines.append(f'- {kind}: {video}: {item["reason"]}')
    lines += ['', 'CV gate 통과는 독립적으로 검증한 악기 정답이나 전체 악보 coverage를 보장하지 않는다. '
              'partial notation의 실제 미기보 연주는 FP로 계산될 수 있다. 반복·겹치는 clip은 상관된 표본이다. '
              'truth가 없거나 strict fill이 없는 곡도 coverage에 포함되며 유효 분모가 없으면 결과는 미정이다.', '',
              '단일 A 절대 측정이며 bootstrap, arm 비교, 승격 판정은 하지 않았다. '
              '입력 truth/체크포인트/코드/오디오 해시를 예측 전 저장하고 결과 작성 전에 다시 확인했다.', '',
              *([f'이전 opening {len(result["previous_openings"])}개 (점수 산출 전 실패, 결과 미확인):', '```json',
                 *[json.dumps(r, ensure_ascii=False) for r in result['previous_openings']], '```', '']
                if result.get('previous_openings') else []),
              '`truth/harvest/vault_log.jsonl`의 이번 durable row:', '```json',
              json.dumps(result['vault_log_row'], ensure_ascii=False, indent=2), '```', '']
    if result.get('reader_rejections'):
        lines += ['Scanner가 native 악보 영역/staff를 증명하지 못한 영상은 unknown으로 유지했다. '
                  'truth event를 생성하지 않았고 점수 분모에 넣지 않았다. 고려한 곡 수에는 포함된다.', '']
        lines += [f'- {r["video"]}: {r["reason"]}' for r in result['reader_rejections']]
    return '\n'.join(lines)


def run(plan_path, arm='A', dry_run=False):
    if not dry_run:
        require_writer()
    job = preflight(plan_path, arm, dry_run)
    if dry_run:
        return dry_summary(job)
    directory = ROOT / job['directory']
    with guard(reads_forbidden=sealed_paths(vault.Vault())):
        runtime_preflight()
    directory.mkdir(parents=True, exist_ok=True)
    once = directory / 'once.json'
    with once.open('x') as stream:
        json.dump({'model_tag': job['model_tag'], 'plan_sha256': sha(ROOT / job['plan_path']),
                   'command': shlex.join(sys.argv), 'state': 'attempted'}, stream)
        stream.flush(); os.fsync(stream.fileno())
    sync_directory(directory)
    verify_metadata(job)
    opening_id = vault.open_final(job['model_tag'])
    job['opening_id'] = opening_id
    NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
    destinations = [directory, NUMBA_CACHE, *[ROOT / s['song'] for s in job['songs'] if s['cohort'] == 'ollie'],
                    *[compare.CACHE / s['slug'] for s in job['songs'] if s['cohort'] == 'manifest']]  # feature() writes feat.npy.tmp then renames
    with guard(writes=destinations):
        write(directory / 'request.json', job)
        write(once, {'model_tag': job['model_tag'], 'opening_id': opening_id,
                     'request_sha256': sha(directory / 'request.json'), 'state': 'opened'})
        command('tools/vision_read/vault_read.py', job, opening_id, directory / 'truth.log')
        verify_metadata(job)
        snapshot = truth_snapshot(job)
        write(directory / 'inputs.json', snapshot)  # immediately after truth, before any model inference
        prepare_audio(job)
        audio = [ROOT / s['cache'] / 'feat.npy' for s in job['songs']]
        audio += [ROOT / s['song'] / name for s in job['songs'] for name in
                  ('activations.mix.npy', 'attacks.htdemucs.npy', 'drums.htdemucs.wav', 'drums.htdemucs.opus')
                  if (ROOT / s['song'] / name).is_file() and not (s['cohort'] == 'ollie' and name == 'activations.mix.npy')]
        snapshot['audio_hashes'] = hashes(audio, ROOT)
        write(directory / 'inputs.json', snapshot)
        verify_inputs(job, snapshot)
        record = inference(job)
        record['cpu_check'] = device_check(job, record)
        write(directory / 'device.json', record)
        snapshot['audio_hashes'].update(hashes([ROOT / s['song'] / 'activations.mix.npy' for s in job['songs']
                                                if s['cohort'] == 'ollie'], ROOT))
        write(directory / 'inputs.json', snapshot)
        verify_inputs(job, snapshot)
        command('tools/gpu/vault_absolute_score.py', job, opening_id, directory / 'score.log')
        result = load(directory / 'scored.json')
        verify(result['prediction_hashes'], ROOT)
        verify_inputs(job, snapshot)
        rows = [json.loads(line) for line in vault.LOG.read_bytes().splitlines()]
        opening = [r for r in rows if r.get('opening_id') == opening_id and r.get('model_tag') == job['model_tag']
                   and r.get('pid') == os.getpid()]
        if len(opening) != 1:
            raise ValueError('missing/ambiguous durable opening row')
        previous = [r for r in rows if r.get('model_tag') == job['model_tag'] and r.get('opening_id') != opening_id]
        result['previous_openings'] = [{**r, 'outcome': PREVIOUS_FAILURES.get(r.get('opening_id'), 'unknown')}
                                       for r in previous]
        result.update(plan=job['plan']['name'], arm='A', model_tag=job['model_tag'], opening_id=opening_id,
                      vault_log_row=opening[0], excluded=job['excluded'], development=DEVELOPMENT,
                      ensemble='seed mean per fold, then fold mean', device=record, inputs_sha256=sha(directory / 'inputs.json'))
        write(directory / 'final.json', result)
        (directory / 'REPORT.md').write_text(report(result))
        write(once, {**load(once), 'state': 'complete', 'final_sha256': sha(directory / 'final.json')})
    print(f'Absolute A complete: {directory / "final.json"}', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['run'])
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--arm', choices=['A'], required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        run(args.plan, args.arm, args.dry_run)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(2, f'vault_absolute refused: {error}\n')


if __name__ == '__main__':
    main()
