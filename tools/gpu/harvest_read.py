"""Read the latest fetch or re-read selected WSL songs with unchanged batch gates.

harvest_wsl.sh read --latest-fetch [--only dailychops] [--jobs N]
harvest_wsl.sh read --reread --only moyin [--ids VIDEO_ID ...] [--dry-run] [--jobs N]
The job wrapper holds the same global lock as fetch/compact/snapshot.
Re-reading includes sealed songs without changing their vault membership.
"""
from __future__ import annotations
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from threading import Event, Lock
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/harvest'))
import harvest
sys.path.insert(0, str(ROOT / 'tools/truth_video'))
import batch

REREAD_STATUSES = {'read', 'unreadable', 'compacted', 'fetched'}
DOWNLOAD_TIMEOUT = 1200


class ReaderIncomplete(RuntimeError):
    """batch.process ran but the reader did not complete this song (recorded by batch; the run goes on)."""


def read_songs(rows: list[dict], read_one, jobs: int) -> list[str]:
    """Bound admission, stop on failure, and join every admitted song.

    A song whose reader did not complete (ReaderIncomplete) does not stop the run: batch records
    it, and its message is returned. Any other error stops admission as before.

    Threads only coordinate batch.process: its CPU-heavy reader is a separate
    subprocess. Never enqueue the whole selection, or a failure would leave
    additional songs scheduled. Admission and failure publication share a lock.
    """
    if jobs < 1:
        raise ValueError('jobs must be positive')
    incomplete = []

    def attempt(row):
        try:
            read_one(row)
        except ReaderIncomplete as error:
            print(f'  {error}', flush=True)
            incomplete.append(str(error))

    if jobs == 1:
        for row in rows:
            attempt(row)
        return incomplete
    stopped = Event()
    admission = Lock()
    selection = iter(rows)

    def run(row):
        try:
            attempt(row)
        except BaseException:
            with admission:
                stopped.set()
            raise

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        pending = set()

        def submit_next():
            with admission:
                if stopped.is_set():
                    return
                row = next(selection, None)
                if row is not None:
                    pending.add(pool.submit(run, row))

        for _ in range(jobs):
            submit_next()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            # Check every completed result before filling any newly free slot.
            # On error the executor context joins running songs before raising.
            for future in done:
                future.result()
            for _ in done:
                submit_next()
    return incomplete


def download_estimate(row: dict, max_height: int) -> int:
    """Offline planning estimate, not a bound or a yt-dlp metadata lookup.

    Budget 8 Mbps at 720p, scaled by pixel area. Unknown durations use the
    harvester's upper duration default. Remux/partial files need extra room.
    """
    try:
        seconds = float(row.get('seconds') or 0)
    except (TypeError, ValueError):
        seconds = 0
    if not math.isfinite(seconds) or seconds <= 0:
        seconds = harvest.DURATION[1]
    return math.ceil(seconds * 1_000_000 * (max_height / 720) ** 2)


def reserve(song: Path, min_free_gb: float, needed: int = 0) -> None:
    free = shutil.disk_usage(song).free
    if free - needed < min_free_gb * 1e9:
        raise RuntimeError(f'disk reserve reached: need {needed / 1e9:.2f} GB plus '
                           f'{min_free_gb:g} GB reserve; free {free / 1e9:.2f} GB; '
                           'no source/truth deletion performed')


def reread_rows(root: Path, only: list[str], ids: list[str] | None) -> list[dict]:
    rows = batch.manifest_read(root / 'manifest.jsonl')
    if ids:
        missing = [video for video in ids if video not in rows or rows[video].get('channel') not in only]
        if missing:
            raise ValueError(f'IDs missing from selected manifest channels: {missing}')
    selected = []
    for video, row in rows.items():
        if row.get('channel') not in only or (ids and video not in ids):
            continue
        if row.get('status') not in REREAD_STATUSES:
            print(video, 'skipped latest status:', row.get('status'), flush=True)
            continue
        selected.append(row)
    if not selected:
        raise ValueError('no eligible manifest videos matching this re-read selection')
    return selected


def download_video(song: Path, row: dict, max_height: int, min_free_gb: float) -> None:
    """Fetch only the video; publish it after a successful nonempty download."""
    video = row['video']
    if not re.fullmatch(r'[A-Za-z0-9_-]{11}', video):
        raise ValueError(f'invalid YouTube video ID: {video!r}')
    needed = 2 * download_estimate(row, max_height)
    error = ''
    for attempt in range(3):
        # Check on every retry too; other jobs/filesystems may consume space.
        reserve(song, min_free_gb, needed)
        with tempfile.TemporaryDirectory(prefix='.reread-video-', dir=song) as temp:
            target = Path(temp) / 'video.mp4'
            argv = [harvest.executable('yt-dlp'), '-q', '--no-playlist',
                    '-f', harvest.video_format(max_height), '--merge-output-format', 'mp4',
                    '-o', str(target), f'https://www.youtube.com/watch?v={video}']
            try:
                run = subprocess.run(argv, cwd=ROOT, env=harvest.command_env(),
                                     capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT)
                error = run.stderr or run.stdout or f'yt-dlp exit {run.returncode}'
                if run.returncode == 0 and target.is_file() and target.stat().st_size:
                    target.replace(song / 'video.mp4')
                    return
            except subprocess.TimeoutExpired:
                error = f'yt-dlp timed out after {DOWNLOAD_TIMEOUT}s'
        if attempt < 2:
            time.sleep(20 * (attempt + 1))
    # Preserve the previous manifest status/truth on video-only fetch failure.
    raise RuntimeError(f'{video}: video download failed after 3 attempts: {error[-400:]}')


def reread(root: Path, only: list[str], ids: list[str] | None, max_height: int,
           dry_run: bool, timeout: float, min_free_gb: float, jobs: int = 1) -> None:
    if not (root / '.wsl-authority').exists() or (root / '.wsl-mirror').exists():
        raise RuntimeError('re-read requires WSL authority after push --init; use harvest_wsl.sh')
    rows = reread_rows(root, only, ids)
    estimate = 0
    downloads = 0
    preparation = Lock()

    def read_one(row):
        nonlocal estimate, downloads
        # Video downloads/backups are serial; already-started readers continue
        # while the next admitted song is prepared. No extra songs are queued.
        with preparation:
            profile, profile_path = batch.channel_profile(row['channel'])
            if not profile.get('enabled'):
                raise ValueError(f'{row["channel"]} reader disabled; no gate override')
            song = batch.song_path(root, row)
            if not song.is_dir():
                raise FileNotFoundError(f'{song}: existing song folder missing; push --inputs first')
            if not (song / 'attacks.htdemucs.npy').is_file():
                raise FileNotFoundError(f'{song}: missing attacks.htdemucs.npy; no audio pipeline is run')
            video = song / 'video.mp4'
            out = song / 'truth'
            if video.is_symlink() or out.is_symlink():
                raise ValueError(f'{song}: video/canonical truth must not be a symlink')
            needs_video = not video.is_file() or video.stat().st_size == 0
            size = download_estimate(row, max_height) if needs_video else 0
            downloads += int(needs_video)
            estimate += size
            print(f'[{row["channel"]}] {row["video"]} {row["status"]}: re-read; '
                  + (f'download video <= {max_height}p, estimate {size / 1e9:.3f} GB'
                     if needs_video else 'reuse existing video (height unchanged)'), flush=True)
            if dry_run:
                return
            with batch.lock(song / '.truth.lock') as acquired:
                if not acquired:
                    raise RuntimeError(f'{row["video"]}: another reader holds the song lock')
                if needs_video:
                    download_video(song, row, max_height, min_free_gb)
                reserve(song, min_free_gb)
                # Identical signatures short-circuit in batch. Moving canonical
                # truth forces an actual read, without any gate override.
                previous = None
                if out.exists():
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                    previous = song / ('truth.prev-' + stamp)
                    out.rename(previous)
                try:
                    # Append only: vault timestamps/channel membership stay intact.
                    batch.manifest_append(root / 'manifest.jsonl', video=row['video'], status='fetched')
                except Exception:
                    if previous is not None:
                        previous.rename(out)
                    raise
                row = batch.manifest_read(root / 'manifest.jsonl')[row['video']]
                if previous is not None:
                    print(f'  previous truth: {previous}', flush=True)
        # The job wrapper still holds the global writer lock. batch acquires its
        # own per-song lock and publishes read/unreadable through its normal path.
        reserve(song, min_free_gb)
        result = batch.process(root, row, profile, profile_path, timeout)
        if result is None or not result.get('complete'):
            raise ReaderIncomplete(f'{row["video"]}: reader did not complete; source kept, see '
                                   'truth/run.log and truth.prev-*')
    incomplete = read_songs(rows, read_one, 1 if dry_run else jobs)
    if dry_run:
        print(f'DRY RUN: {len(rows)} songs; {downloads} video downloads; '
              f'estimated download total {estimate / 1e9:.3f} GB. '
              'Each download requires twice its estimate plus the disk reserve; '
              'bitrate/duration and reader/history space may differ. No changes made.', flush=True)
    else:
        batch.report(root)
        if incomplete:
            raise SystemExit(f'{len(incomplete)} song(s) did not complete (others were read): ' + '; '.join(incomplete))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--latest-fetch', action='store_true')
    mode.add_argument('--reread', action='store_true')
    p.add_argument('--only', nargs='+')
    p.add_argument('--ids', nargs='+', action='extend',
                   help='re-read only these IDs within --only channels; an ID starting with "-" as --ids=-ID')
    p.add_argument('--max-height', type=int, help='missing video height limit (default: harvest.MAX_HEIGHT)')
    p.add_argument('--dry-run', action='store_true', help='list re-reads and offline video download estimates')
    p.add_argument('--timeout', type=float, default=1200)
    p.add_argument('--min-free-gb', type=float, default=10)
    p.add_argument('--jobs', type=int, default=1, help='maximum concurrent song readers (default: 1)')
    a = p.parse_args()
    if a.jobs < 1:
        p.error('jobs must be positive')
    if not math.isfinite(a.timeout) or a.timeout <= 0:
        p.error('timeout must be finite and positive')
    if not math.isfinite(a.min_free_gb) or a.min_free_gb < 0:
        p.error('min-free-gb must be finite and nonnegative')
    if a.reread and not a.only:
        p.error('--reread requires --only CHANNEL ...')
    if a.max_height is not None and a.max_height < 144:
        p.error('max-height must be at least 144')
    if not a.reread and (a.ids is not None or a.max_height is not None or a.dry_run):
        p.error('--ids, --max-height and --dry-run require --reread')
    root = ROOT / 'truth/harvest'
    if (root / '.wsl-mirror').exists():
        raise RuntimeError('WSL owns this manifest; use harvest_wsl.sh')
    if a.reread:
        reread(root, a.only, a.ids, a.max_height or harvest.MAX_HEIGHT,
               a.dry_run, a.timeout, a.min_free_gb, a.jobs)
        return
    path = Path((ROOT / 'logs/harvest/latest_fetch.txt').read_text().strip())
    if not path.is_absolute():
        path = ROOT / path
    job = json.loads(path.read_text())
    if job['exit'] != 0:
        raise ValueError('latest fetch did not finish successfully; inspect its log/status')
    rows = batch.manifest_read(root / 'manifest.jsonl')
    videos = [v for v in job['videos'] if not a.only or rows[v]['channel'] in a.only]
    if not videos:
        raise ValueError('latest fetch added no videos matching this selection')
    def read_one(row):
        video = row['video']
        if row['status'] in ('read', 'unreadable', 'compacted'):
            print(video, 'already attempted:', row['status'])
            return
        if row['status'] != 'fetched':
            raise ValueError(f'{video}: fetch failed')
        profile, profile_path = batch.channel_profile(row['channel'])
        if not profile.get('enabled'):
            raise ValueError(f'{row["channel"]} reader disabled; no gate override')
        if shutil.disk_usage(root).free / 1e9 < a.min_free_gb:
            raise RuntimeError('reader disk reserve reached; no deletion performed')
        result = batch.process(root, row, profile, profile_path, a.timeout)
        if result is None or not result.get('complete'):
            raise ReaderIncomplete(f'{video}: reader did not complete; source kept, see truth/run.log')
    # Fetch job IDs should be unique, but never admit the same song twice if a
    # status file repeats an ID. Manifest selection remains limited to this job.
    incomplete = read_songs([rows[v] for v in dict.fromkeys(videos)], read_one, a.jobs)
    batch.report(root)
    if incomplete:
        raise SystemExit(f'{len(incomplete)} song(s) did not complete (others were read): ' + '; '.join(incomplete))


if __name__ == '__main__':
    main()
