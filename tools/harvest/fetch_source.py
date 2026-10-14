"""Detector-free source fetch for sealed/split channels (video + audio only).

    .venv/bin/python tools/harvest/fetch_source.py fetch --channel olliewoods_shorts [--ids ID ...] [--limit N]
    .venv/bin/python tools/harvest/fetch_source.py copy --channel olliewoods_shorts --source .eval-cache/pilot_new/ollieshorts
    .venv/bin/python tools/harvest/fetch_source.py prepare --channel olliewoods_shorts [--device cuda] [--jobs 2]
    .venv/bin/python tools/harvest/fetch_source.py status --channel olliewoods_shorts

Files land in sources/<channel>/<id>/ (video.mp4, info.json, fetch.json; after
prepare: drums.htdemucs.wav, attacks.htdemucs.npy, onsets.bands.htdemucs.npy,
meta.json). Nothing here touches truth/harvest, the manifest, the vault or the
harvest writer lock, so a fetched video is invisible to training, scoring and
readers until it is deliberately imported. Prepare separates the drum stem with
cached htdemucs weights (no download) and computes class-agnostic attacks and
band onsets; it never imports the drum detector, the transcription pipeline or
a scorer.

Rate limits: YouTube waits 8-15 s between requests and stops after two
consecutive failures. Instagram waits 45-90 s, rests at least 10 min after every
20 requests, fetches at most 60 per KST day and stops at the FIRST failure or
suspicious response. A stop writes STOP.json; later runs refuse until
--clear-stop is given (Instagram: and the stop is at least an hour old).
No cookies, browser profiles or logins are used.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / 'sources'
CHANNELS = ROOT / 'tools/harvest/channels.json'
LISTS = ROOT / 'truth/harvest/lists'
KST = timezone(timedelta(hours=9))
VIDEO_NAMES = ('video.mp4', 'video.webm', 'video.mkv')

# OpenCV on WSL cannot decode AV1 (no hardware decoder), so prefer H.264, then any SDR non-AV1 stream.
FORMAT = 'bv*[width<=1080][vcodec^=avc1]+ba/bv*[width<=1080][dynamic_range=SDR][vcodec!^=av01]+ba'
POLICY = {
    'youtube': dict(gap=(8.0, 15.0), max_consecutive_failures=2, stop_on_first=False,
                    rest_every=None, rest_seconds=0, daily_cap=None, stop_wait=0,
                    url='https://www.youtube.com/shorts/{id}',
                    format=FORMAT),
    'instagram': dict(gap=(45.0, 90.0), max_consecutive_failures=1, stop_on_first=True,
                      rest_every=20, rest_seconds=600, daily_cap=60, stop_wait=3600,
                      url='https://www.instagram.com/reel/{id}/',
                      format=FORMAT),
}
# Any of these in yt-dlp output means "slow down / blocked", never "retry now".
SUSPICIOUS = re.compile(r'(?i)login|log in|sign in|429|too many requests|please wait|rate.?limit|'
                        r'checkpoint|not available|empty media|private|consent|captcha|403|forbidden')


def channel_config(name: str) -> dict:
    for c in json.loads(CHANNELS.read_text())['channels']:
        if c['name'] == name:
            return c
    raise SystemExit(f'unknown channel: {name}')


def platform(config: dict) -> str:
    return 'instagram' if 'instagram.com' in config.get('url_template', '') else 'youtube'


def listing(name: str) -> list[tuple[str, str]]:
    rows = []
    for line in (LISTS / f'{name}.txt').read_text().splitlines():
        if line.strip():
            vid, _, title = line.split('|', 2)
            rows.append((vid, title))
    return rows


def folder(channel: str, vid: str) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9_-]{6,20}', vid):
        raise ValueError(f'unsafe video id: {vid!r}')
    return SOURCES / channel / vid


def video_file(path: Path) -> Path | None:
    found = [path / n for n in VIDEO_NAMES if (path / n).is_file() and (path / n).stat().st_size > 0]
    return found[0] if len(found) == 1 else None


def done(channel: str, vid: str) -> bool:
    path = folder(channel, vid)
    meta = path / 'fetch.json'
    return meta.exists() and json.loads(meta.read_text()).get('status') == 'ok' and video_file(path) is not None


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def log(channel: str, row: dict) -> None:
    path = SOURCES / channel / 'fetch_log.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as out:
        out.write(json.dumps({'at': datetime.now(timezone.utc).isoformat(), **row}, ensure_ascii=False) + '\n')


def check_stop(channel: str, policy: dict, clear: bool, now: float) -> None:
    stop = SOURCES / channel / 'STOP.json'
    if not stop.exists():
        return
    data = json.loads(stop.read_text())
    age = now - data.get('unix', now)
    if not clear:
        raise SystemExit(f'{stop} exists ({data.get("reason")}); inspect it, then rerun with --clear-stop')
    if policy['stop_wait'] and age < policy['stop_wait']:
        raise SystemExit(f'stop is only {age/60:.0f} min old; wait at least {policy["stop_wait"]/60:.0f} min')
    stop.rename(stop.with_name(f'STOP.cleared-{int(now)}.json'))


def daily_count(channel: str, day: str) -> int:
    path = SOURCES / channel / 'state.json'
    return json.loads(path.read_text()).get('daily', {}).get(day, 0) if path.exists() else 0


def bump_daily(channel: str, day: str) -> None:
    path = SOURCES / channel / 'state.json'
    state = json.loads(path.read_text()) if path.exists() else {}
    state.setdefault('daily', {})[day] = state.get('daily', {}).get(day, 0) + 1
    write_json(path, state)


def download(url: str, fmt: str, target: Path) -> tuple[bool, str]:
    """One yt-dlp attempt into a scratch dir; the video is moved in only when complete."""
    work = target / '.part'
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    cmd = [str(Path(sys.executable).with_name('yt-dlp')) if Path(sys.executable).with_name('yt-dlp').exists() else 'yt-dlp',
           '--no-cookies', '--no-playlist', '--no-progress', '--retries', '1', '--fragment-retries', '3',
           '-f', fmt, '--merge-output-format', 'mp4', '--write-info-json',
           '-o', str(work / 'video.%(ext)s'), url]
    try:
        run = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        shutil.rmtree(work, ignore_errors=True)
        return False, 'timeout'
    text = (run.stderr or '') + (run.stdout or '')
    video = work / 'video.mp4'
    info = work / 'video.info.json'
    if run.returncode != 0 or not video.exists() or video.stat().st_size == 0:
        shutil.rmtree(work, ignore_errors=True)
        return False, text[-1500:] or f'exit {run.returncode}, no video'
    os.replace(video, target / 'video.mp4')
    if info.exists():
        os.replace(info, target / 'info.json')
    shutil.rmtree(work, ignore_errors=True)
    return True, text[-500:]


def height_format(height: int) -> str:
    """Landscape videos: cap the HEIGHT (a width cap would give ~608p for 16:9), same codec preference."""
    return (f'bv*[height<={height}][vcodec^=avc1]+ba/'
            f'bv*[height<={height}][dynamic_range=SDR][vcodec!^=av01]+ba')


def fetch(channel: str, ids: list[str] | None, limit: int | None, clear_stop: bool,
          sleep=time.sleep, now=time.time, rng=random.random, downloader=download,
          max_height: int | None = None) -> dict:
    config = channel_config(channel)
    policy = POLICY[platform(config)]
    rows = listing(channel)
    titles = dict(rows)
    if ids:
        unknown = [v for v in ids if v not in titles]
        if unknown:
            raise SystemExit(f'ids not in {channel} list: {unknown}')
        order = ids
    else:
        order = [v for v, _ in rows]
    todo = [v for v in order if not done(channel, v)]
    if limit is not None:
        todo = todo[:limit]
    SOURCES.mkdir(exist_ok=True)
    lock = (SOURCES / '.fetch.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another fetch_source run holds sources/.fetch.lock')
    check_stop(channel, policy, clear_stop, now())
    url_template = config.get('url_template') or policy['url']
    fmt = policy['format']
    if max_height is not None:  # regular (landscape) YouTube videos rather than Shorts
        if platform(config) != 'youtube':
            raise SystemExit('--max-height applies to YouTube videos only')
        fmt = height_format(max_height)
        url_template = config.get('url_template') or 'https://www.youtube.com/watch?v={id}'
    result = dict(channel=channel, requested=len(todo), ok=[], failed=[], stopped=None)
    consecutive = 0
    for i, vid in enumerate(todo):
        day = datetime.fromtimestamp(now(), KST).date().isoformat()
        if policy['daily_cap'] is not None and daily_count(channel, day) >= policy['daily_cap']:
            result['stopped'] = f'daily cap {policy["daily_cap"]} reached for {day} KST'
            break
        if i:
            low, high = policy['gap']
            sleep(low + (high - low) * rng())
            if policy['rest_every'] and i % policy['rest_every'] == 0:
                sleep(policy['rest_seconds'] + 120 * rng())
        target = folder(channel, vid)
        target.mkdir(parents=True, exist_ok=True)
        started = now()
        ok, detail = downloader(url_template.format(id=vid), fmt, target)
        if policy['daily_cap'] is not None:
            bump_daily(channel, day)
        suspicious = bool(SUSPICIOUS.search(detail)) if not ok else False
        record = dict(video=vid, channel=channel, title=titles[vid], url=url_template.format(id=vid), format=fmt,
                      status='ok' if ok else 'failed', seconds=round(now() - started, 1),
                      detail=detail[-500:], cookies=False, source='network', detector_free=True)
        write_json(target / 'fetch.json', record)
        log(channel, {k: record[k] for k in ('video', 'status', 'seconds', 'detail')})
        print(json.dumps({k: record[k] for k in ('video', 'status', 'seconds')}), flush=True)
        if ok:
            consecutive = 0
            result['ok'].append(vid)
            continue
        consecutive += 1
        result['failed'].append(vid)
        if policy['stop_on_first'] or suspicious and platform(config) == 'instagram' \
                or consecutive >= policy['max_consecutive_failures']:
            reason = f'{consecutive} consecutive failure(s); last {vid}' + (' (suspicious response)' if suspicious else '')
            write_json(SOURCES / channel / 'STOP.json', dict(reason=reason, video=vid, detail=detail[-1500:],
                                                             unix=now(), at=datetime.now(timezone.utc).isoformat()))
            result['stopped'] = reason
            break
    write_json(SOURCES / channel / f'last_fetch.json', result)
    return result


def copy_existing(channel: str, source: Path) -> list[str]:
    """Reuse already-downloaded pilot media instead of requesting it again."""
    titles = dict(listing(channel))
    copied = []
    for path in sorted(p for p in source.iterdir() if p.is_dir() and p.name in titles):
        src = video_file(path)
        if src is None or done(channel, path.name):
            continue
        target = folder(channel, path.name)
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target / src.name)
        for name in ('audio.m4a', 'drums.htdemucs.wav', 'attacks.htdemucs.npy', 'onsets.bands.htdemucs.npy'):
            if (path / name).exists():
                shutil.copy2(path / name, target / name)
        write_json(target / 'fetch.json', dict(video=path.name, channel=channel, title=titles[path.name],
                                               status='ok', source=f'copied from {path.relative_to(ROOT)}',
                                               cookies=False, detector_free=True))
        copied.append(path.name)
    return copied


def attack_function():
    """groovemap.fills.attack_times, compiled alone so the detector package is never imported."""
    import numpy as np
    import soundfile as sf
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    from scipy.signal import find_peaks, stft
    tree = ast.parse((ROOT / 'src/groovemap/fills.py').read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'attack_times'
            or isinstance(n, ast.Assign) and any(getattr(t, 'id', None) in ('_HOP', '_BANDS') for t in n.targets)]
    if len(keep) != 3:
        raise RuntimeError('src/groovemap/fills.py no longer has attack_times/_HOP/_BANDS as expected')
    namespace = dict(np=np, sf=sf, stft=stft, find_peaks=find_peaks, maximum_filter1d=maximum_filter1d,
                     uniform_filter1d=uniform_filter1d, Path=Path)
    exec(compile(ast.Module(body=keep, type_ignores=[]), 'fills.attack_times', 'exec'), namespace)
    return namespace['attack_times']


def prepare_one(path: Path, device: str) -> dict:
    import numpy as np
    import soundfile as sf
    import torch
    torch.set_num_threads(1)

    def offline(*args, **kwargs):
        raise RuntimeError('fetch_source prepare: htdemucs weights are not cached; refusing to download')
    torch.hub.download_url_to_file = offline
    video = video_file(path)
    if video is None:
        raise ValueError(f'no single video in {path}')
    stem = path / 'drums.htdemucs.wav'
    if not stem.exists():
        mix = path / 'audio.wav'
        source = path / 'audio.m4a' if (path / 'audio.m4a').exists() else video
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-threads', '1', '-y', '-i', str(source),
                        '-vn', '-ar', '44100', '-ac', '2', str(mix)], check=True)
        from demucs.api import Separator
        audio, sr = sf.read(mix, dtype='float32', always_2d=True)
        separator = Separator(model='htdemucs', device=device, progress=False)
        _, stems = separator.separate_tensor(torch.from_numpy(audio.T.copy()), sr)
        part = path / 'drums.htdemucs.part.wav'
        sf.write(part, stems['drums'].cpu().numpy().T, separator.samplerate)
        part.replace(stem)
        mix.unlink()
    attacks = path / 'attacks.htdemucs.npy'
    if not attacks.exists():
        np.save(attacks, attack_function()(stem))
    sys.path.insert(0, str(ROOT / 'tools/truth_video'))
    import bands  # band-limited STFT flux only; no detector labels
    bands.load(path)
    import cv2
    cap = cv2.VideoCapture(str(video))
    native = dict(width=int(cap.get(3)), height=int(cap.get(4)), fps=cap.get(5), frames=int(cap.get(7)))
    cap.release()
    fetched = json.loads((path / 'fetch.json').read_text())
    info = json.loads((path / 'info.json').read_text()) if (path / 'info.json').exists() else {}
    meta = dict(video_id=path.name, channel=fetched['channel'], title=info.get('title') or fetched.get('title'),
                description=info.get('description'), duration=info.get('duration') or native['frames'] / max(native['fps'], 1e-9),
                uploader=info.get('uploader'), upload_date=info.get('upload_date'), native_video=native,
                video=video.name, detector_free=True)
    write_json(path / 'meta.json', meta)
    return dict(video=path.name, attacks=int(len(np.load(attacks))), native=native)


def prepare(channel: str, device: str, jobs: int) -> list[dict]:
    paths = [p for p in sorted((SOURCES / channel).iterdir()) if p.is_dir() and done(channel, p.name)
             and not (p / 'meta.json').exists()]
    if jobs <= 1:
        return [prepare_one(p, device) for p in paths]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(prepare_one, paths, [device] * len(paths)))


def status(channel: str) -> dict:
    rows = listing(channel)
    have = [v for v, _ in rows if done(channel, v)]
    prepared = [v for v in have if (folder(channel, v) / 'meta.json').exists()]
    stop = SOURCES / channel / 'STOP.json'
    return dict(channel=channel, listed=len(rows), fetched=len(have), prepared=len(prepared),
                stop=json.loads(stop.read_text()) if stop.exists() else None)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('fetch')
    f.add_argument('--channel', required=True)
    f.add_argument('--ids', nargs='+')
    f.add_argument('--ids-file', type=Path, help='one video ID per line (IDs may start with "-")')
    f.add_argument('--limit', type=int)
    f.add_argument('--clear-stop', action='store_true')
    f.add_argument('--max-height', type=int, help='landscape videos: cap height instead of width (e.g. 1080)')
    c = sub.add_parser('copy')
    c.add_argument('--channel', required=True)
    c.add_argument('--source', required=True, type=Path)
    r = sub.add_parser('prepare')
    r.add_argument('--channel', required=True)
    r.add_argument('--device', default='cpu')
    r.add_argument('--jobs', type=int, default=1)
    s = sub.add_parser('status')
    s.add_argument('--channel', required=True)
    args = p.parse_args(argv)
    if args.command == 'fetch':
        ids = (args.ids or []) + (args.ids_file.read_text().split() if args.ids_file else [])
        out = fetch(args.channel, ids or None, args.limit, args.clear_stop, max_height=args.max_height)
        print(json.dumps({**out, 'ok': len(out['ok']), 'failed': out['failed']}))
        raise SystemExit(1 if out['stopped'] and 'daily cap' not in out['stopped'] else 0)
    if args.command == 'copy':
        print(json.dumps(copy_existing(args.channel, args.source.resolve())))
    elif args.command == 'prepare':
        if not 1 <= args.jobs <= 4:
            p.error('jobs must be 1..4')
        for row in prepare(args.channel, args.device, args.jobs):
            print(json.dumps(row), flush=True)
    else:
        print(json.dumps(status(args.channel), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
