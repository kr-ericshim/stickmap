"""Offline representative-frame CV read of explicitly allowed fetched sources.

Inputs are read-only. Every output, including native crops and future reread
jobs, lives under .eval-cache/vision_pilot/channel_read/<channel>/<video>.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import hashlib
import multiprocessing
import os
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT, dump, load, sha
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools/truth_video'))
from channel_read_core import VERSION, relative

BASE = DATA / 'channel_read'
INPUT_NAMES = ('drums.htdemucs.wav', 'attacks.htdemucs.npy', 'onsets.bands.htdemucs.npy', 'meta.json')


def video_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{11}', value):
        raise ValueError('unsafe video id (expected 11 YouTube ID characters): ' + str(value))
    return value


def system_id(video, index):
    video_id(video)
    if type(index) is not int or index < 0:
        raise ValueError('system index must be a nonnegative integer')
    return f'{video}_src_{index:04d}'


def no_symlinks(path):
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(ROOT):
        raise ValueError('path must be repo-relative: ' + str(path))
    for part in (path, *path.parents):
        if part == ROOT:
            break
        if part.is_symlink():
            raise ValueError('symlink input/output forbidden: ' + relative(part.parent) + '/' + part.name)
    return path


def output_path(path):
    if DATA != ROOT / '.eval-cache/vision_pilot':
        raise ValueError('channel_read requires DATA under .eval-cache/vision_pilot')
    path = no_symlinks(path)
    if not path.is_relative_to(BASE):
        raise ValueError('channel_read output escapes its scratch subtree')
    # Upstream readers may overwrite pre-existing PNG/JSON children directly.
    if path.exists() and any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError('symlink within output tree forbidden')
    return path


def read_allow(path):
    if path is None:
        raise ValueError('--allow FILE is required')
    value = load(path)
    if not isinstance(value, dict) or not isinstance(value.get('train'), list):
        raise ValueError('allow file must be JSON {"train": [ids]}')
    ids = [video_id(v) for v in value['train']]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate ids in allow train list')
    return frozenset(ids)


def require_allowed(video, allowed):
    """Fresh, independent seal checks BEFORE source metadata or media access."""
    video_id(video)
    if video not in allowed:
        raise ValueError('video is not in --allow train list: ' + video)
    from tools.harvest.vault import Vault, config
    explicit = set(config()['video_ids'])
    seal = Vault()
    if video in explicit or seal.is_vault(video):
        raise ValueError('vault input forbidden: ' + video)


def source(root, channel, video, allowed):
    require_allowed(video, allowed)
    root = no_symlinks(root)
    # Accept sources/<channel>/<id> and a direct cohort root such as the
    # legacy .eval-cache/pilot_new/ollieshorts/<id> verification input.
    cohort = root / channel if (root / channel).is_dir() else root
    song = no_symlinks(cohort / video)
    if not song.is_dir():
        raise ValueError('missing source folder: ' + relative(song))
    if any(p.is_symlink() for p in song.rglob('*')):
        raise ValueError('symlink within source tree forbidden: ' + video)
    for name in INPUT_NAMES:
        if not (song / name).is_file():
            raise ValueError('missing fetched input: ' + relative(song / name))
    meta = load(song / 'meta.json')
    if meta.get('video_id') != video or meta.get('channel') != channel:
        raise ValueError('source metadata identity/channel mismatch: ' + video)
    from media import video_path, VIDEO_NAMES
    if meta.get('video') not in (None, video, *VIDEO_NAMES):
        raise ValueError('source metadata video identity mismatch: ' + video)
    video_path(song)  # existence/uniqueness only; no media open
    return song


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): file_hash(p)
            for p in sorted(folder.rglob('*')) if p.is_file()}


@contextmanager
def write_guard(folder):
    """Fail closed for Python writes outside one video; offline by construction.

    OpenCV writes bypass Python audit hooks, so every native destination is
    supplied explicitly under the symlink-checked folder as well.
    """
    active = True

    def check(path):
        if isinstance(path, (str, bytes, os.PathLike)):
            resolved = Path(os.fsdecode(path)).resolve()
            if not resolved.is_relative_to(folder):
                raise PermissionError('write outside channel_read video scratch forbidden')
        else:
            raise PermissionError('unverifiable write destination')

    def audit(event, args):
        if not active:
            return
        if event == 'open':
            _, mode, flags = args
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+'))
            writing |= bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing:
                check(args[0])
        elif event in ('os.rename', 'os.link'):
            check(args[0]); check(args[1])
        elif event == 'os.symlink':
            raise PermissionError('symlink creation forbidden')
        elif event in ('os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.chown', 'os.utime',
                       'os.setxattr', 'os.removexattr', 'os.truncate'):
            check(args[0])
        elif event in ('socket.connect', 'socket.getaddrinfo', 'subprocess.Popen'):
            raise PermissionError('network/subprocess access forbidden in offline channel_read')

    sys.addaudithook(audit)
    try:
        yield
    finally:
        active = False


def process(task):
    root, channel, video, allowed = task
    os.chdir(ROOT)  # upstream scan provenance may already use repo-relative paths
    song = source(Path(root), channel, video, allowed)
    folder = output_path(BASE / channel / video)
    before = snapshot(song)
    dependencies = {}
    for directory in ('tools/truth_video', 'tools/vision_read', 'tools/harvest'):
        for p in sorted((ROOT / directory).rglob('*')):
            if p.is_file() and p.suffix in ('.py', '.json', '.png') and '__pycache__' not in p.parts:
                dependencies[relative(p)] = file_hash(p)
    inputs = dict(version=VERSION, source=relative(song), files=before, dependencies=dependencies)
    summary_path = folder / 'summary.json'
    if summary_path.exists():
        result = load(summary_path)
        if result['inputs'] != inputs:
            raise ValueError('completed output has different inputs/code; use a fresh channel_read tree: ' + video)
        for name, digest in result['output_sha256'].items():
            if sha(folder / name) != digest:
                raise ValueError('completed output integrity mismatch: ' + video + '/' + name)
        print(video + ': reuse verified output', flush=True)
        return result
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('incomplete output exists; archive this video scratch folder before retry: ' + video)
    folder.mkdir(parents=True, exist_ok=True)
    from channel_read_core import notation, read_cv, extract, prepare, stage, gate, counts
    legend = ROOT / f'tools/truth_video/legends/{channel}.json'
    import cv2
    cv2.setNumThreads(1)
    with write_guard(folder), (folder / 'run.log').open('w') as log, redirect_stdout(log), redirect_stderr(log):
        work = folder / 'work'
        scan, read = work / 'notation', work / 'truth-reviewed'
        require_allowed(video, allowed)
        notation(song, scan, legend)
        require_allowed(video, allowed)
        read_cv(song, read, legend, scan)
        cv_only = counts(read)
        require_allowed(video, allowed)
        inventory = extract(song, channel, work / 'native', read, scan, system_id)
        require_allowed(video, allowed)
        prepare(inventory, work)
        staging = stage(inventory, work, folder / 'cv')
        require_allowed(video, allowed)
        variant = gate(song, folder / 'cv')
        after = snapshot(song)
        if before != after:
            raise ValueError('source files changed during read: ' + video)
        result = dict(video=video, channel=channel, version=VERSION, variant='cv', vault=False,
                      source=relative(song), work=relative(work), cv=relative(folder / 'cv'),
                      strip_systems=inventory['strip_systems'], systems_found=inventory['systems_found'],
                      systems_read=staging['systems_read'], systems=inventory['systems'],
                      source_readonly_verified=True, cv_only=cv_only, **variant, inputs=inputs,
                      gate_sha256={p: sha(ROOT / 'tools/truth_video' / p) for p in ('align.py', 'bands.py', 'tiers.py')})
        dump(folder / 'cv/summary.json', {k: v for k, v in result.items() if k != 'inputs'})
    result['output_sha256'] = snapshot(folder)
    dump(summary_path, result)
    print(f"{video}: strip/jobs {result['strip_systems']}/{result['systems_found']}; "
          f"CV-only {cv_only['strict_notes']}/{cv_only['scoring_notes']}; "
          f"representative CV {result['strict_notes']}/{result['scoring_notes']}", flush=True)
    return result


def totals(rows):
    keys = ('strict_pass', 'scoring_pass', 'strict_notes', 'scoring_notes')
    answer = {k: sum(r[k] for r in rows) for k in keys}
    for key in ('fast_strict', 'fast_scoring'):
        answer[key] = {kind: sum(r[key][kind] for r in rows) for kind in ('K', 'S', 'T')}
    return answer


def report(channel, rows, errors, allow_path, root):
    folder = output_path(BASE / channel)
    summary = dict(version=VERSION, channel=channel, allow=relative(allow_path), sources_root=relative(root),
                   videos=rows, errors=errors, totals=totals(rows), cv_only_totals=totals([r['cv_only'] for r in rows]),
                   gpt_invocations=0, detector_invocations=0)
    dump(folder / 'summary.json', summary)
    t, baseline = summary['totals'], summary['cv_only_totals']
    lines = [f'**결론: {len(rows)}개 허용·비봉인 영상의 대표 프레임 CV 재검출을 완료했다.** ' +
             f"strict/scoring {t['strict_notes']}/{t['scoring_notes']} notes, " +
             f"CV-only {baseline['strict_notes']}/{baseline['scoring_notes']} notes다. " +
             (f'{len(errors)}개 실행 실패가 있어 해당 영상은 합계에서 제외했다.' if errors else '실행 실패는 없다.'), '',
             '| 영상 | strip / jobs | CV-only strict/scoring 시스템 | CV-only notes | 대표 CV strict/scoring 시스템 | 대표 CV notes | strict fast K/S/T | 주요 거부 사유 |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        b = r['cv_only']; fast = '/'.join(str(r['fast_strict'][k]) for k in ('K', 'S', 'T'))
        reasons = '; '.join(f'{reason} ({n})' for reason, n in r['top_reasons'][:3])
        lines.append(f"| {r['video']} | {r['strip_systems']}/{r['systems_found']} | "
                     f"{b['strict_pass']}/{b['scoring_pass']} | {b['strict_notes']}/{b['scoring_notes']} | "
                     f"{r['strict_pass']}/{r['scoring_pass']} | {r['strict_notes']}/{r['scoring_notes']} | {fast} | {reasons} |")
    lines += ['', '대표 프레임 선택·좌표 검증은 channel_pilot과 같다. scan의 정확한 crop과 '
              '15/50/85% 프레임을 사용하고, CV 오류·staff 사유·head 수·선명도로 순위화한다. '
              'skip_reasons, match_geometry, convert_cursor(cv_sanity=True)를 직접 재사용한다.', '',
              '기존 strip의 cursor/colour evidence와 제외 행의 문맥을 유지하고 notes만 대표 프레임 CV로 교체한다. '
              'build_truth(song, out, bands=bands.load(song, cache=False))와 tiers.write를 그대로 호출한다. '
              'Fast K/S/T는 solos_gate.fast_counts의 동일 시스템 내 core 순간 간격 <90ms 정의다.', '',
              'allow train 목록과 Vault().is_vault 및 vault.json.video_ids를 미디어 접근 전 확인한다. '
              '각 단계에서 봉인을 다시 확인하고 원본 전체의 전후 해시를 비교했다. '
              'Python 쓰기는 영상별 scratch 밖에서 거부하며 OpenCV 출력도 같은 경로로 제한한다. '
              '모든 산출물 경로는 저장소 기준 상대 경로다.', '',
              'jobs는 work/hybrid/jobs/<id>_src_NNNN에 있다. 대표 CV에서 strict가 거부된 준비 job은 '
              '추후 선택적 GPT 재독의 입력으로 사용할 수 있다. 이 실행은 네트워크·GPT/Gemini·드럼 검출기·훈련을 사용하지 않는다.', '',
              '게이트 통과량은 완전한 음악 정답 또는 청취 검증을 뜻하지 않는다. '
              '단일 프레임과 median strip의 head/centroid 차이에 따라 CV-only와 수치가 달라질 수 있다. '
              '기존 미검증 악기·표시는 unknown이며 학습용 음성 negative를 만들지 않는다.', '']
    for error in errors:
        lines.append(f"실패 {error['video']}: {error['error']}")
    (folder / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    return summary


def write_readme():
    output_path(BASE).mkdir(parents=True, exist_ok=True)
    (BASE / 'README.md').write_text('''# Channel read: WSL 재현

저장소 전체의 unchanged reader/gate/template/legend, truth/harvest/vault.json과
manifest.jsonl, 허용 파일 및 source 폴더를 같은 상대 경로로 전송한다.
Python 환경에는 numpy/scipy/OpenCV/soundfile이 필요하다. 네트워크 호출은 없다.

```bash
cd ~/groovemap
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -B tools/vision_read/test_channel_read.py
.venv/bin/python -B tools/vision_read/channel_read.py \\
  --channel olliewoods_shorts \\
  --sources-root .eval-cache/pilot_new/ollieshorts \\
  --allow .eval-cache/vision_pilot/channel_read/allow-ollie-pilots.json --jobs 3
```

새 fetch_source 폴더에서는 --sources-root sources가 기본값이다.
allow 파일은 JSON {"train": ["VIDEO_ID_11"]} 형식이며 실제 승인한 ID만 넣는다.
아래는 allow 파일 경로를 .eval-cache/vision_pilot/channel_read/allow-train.json으로
준비한 뒤 실행하는 정확한 명령이다.

```bash
cd ~/groovemap
.venv/bin/python -B tools/vision_read/channel_read.py \\
  --channel olliewoods_shorts --sources-root sources \\
  --allow .eval-cache/vision_pilot/channel_read/allow-train.json --jobs 3
```

--jobs는 1..6 영상 프로세스이고 OpenCV/BLAS는 각 1 thread다.
--videos ID [ID ...]로 allow의 일부를 선택할 수 있다. allow에 없는 ID,
명시적/시간 기준 vault ID, 다른 meta identity/channel, symlink는 거부한다.
미디어를 읽기 전에 전체 선택 목록을 검사한다. 기본 선택은 allow 전체다.
source가 없는 ID도 오류로 처리하므로 채널별 allow 파일을 준비한다.

완료 결과는 입력/reader/template 해시와 산출물 해시가 같을 때만 재사용한다.
코드나 입력이 바뀌었거나 중단된 폴더가 있으면 해당 channel_read/<channel>/<id>
scratch 폴더를 이 subtree 안에 보관 이동한 후 다시 실행한다.
Mac 산출물을 rsync한 뒤 WSL에서 새 계산을 하려면 Mac의 채널 결과 폴더를
이 subtree 안의 별도 archive 폴더로 옮긴다. 공통 DATA의 다른 폴더는 건드리지 않는다.

영상별 work/notation은 pilot_new notation과 동일한 scan/strip/signature,
work/truth-reviewed는 batch.worker CV-only 결과, work/native는 대표 프레임,
work/hybrid/jobs/<id>_src_NNNN은 hybrid.prepare job이다. cv/는 대표 CV staging,
alignment와 strict/scoring truth이고 채널 summary.json/REPORT.md는 합계다.
기존 GPT runner의 기본 jobs 위치와 다르므로 그대로 --systems만 넘기지 않는다.
향후 runner가 이 준비 경로를 명시적으로 받아야 하며 본 도구는 GPT를 실행하지 않는다.
''')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', required=True)
    parser.add_argument('--allow', type=Path, required=True)
    parser.add_argument('--sources-root', type=Path, default=ROOT / 'sources')
    parser.add_argument('--jobs', type=int, default=3)
    parser.add_argument('--videos', nargs='+')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.channel) or not 1 <= args.jobs <= 6:
        parser.error('safe channel name and --jobs 1..6 required')
    try:
        allow = no_symlinks(args.allow)
        allowed = read_allow(allow)
        selected = args.videos if args.videos is not None else sorted(allowed)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError('select at least one video, without duplicates')
        root = no_symlinks(args.sources_root)
        if not (ROOT / f'tools/truth_video/legends/{args.channel}.json').is_file():
            raise ValueError('channel legend does not exist')
        # Finish the entire cohort's seal preflight before any worker starts.
        for video in selected:
            require_allowed(video, allowed)
        for video in selected:
            source(root, args.channel, video, allowed)
        output_path(BASE / args.channel)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    rows, errors = [], []
    tasks = [(str(root), args.channel, video, allowed) for video in selected]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = {pool.submit(process, task): task[2] for task in tasks}
        for future in concurrent.futures.as_completed(pending):
            video = pending[future]
            try:
                rows.append(future.result())
            except Exception as error:
                message = (type(error).__name__ + ': ' + str(error)).replace(str(ROOT) + '/', '')
                errors.append(dict(video=video, error=message))
                print(video + ': ' + message, file=sys.stderr, flush=True)
    rows.sort(key=lambda r: selected.index(r['video']))
    report(args.channel, rows, errors, allow, root)
    write_readme()
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
