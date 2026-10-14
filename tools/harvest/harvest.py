"""Fetch drum-notation videos from several channels for ground truth.

Each chosen video is transcribed into truth/harvest/songs/<slug> with the normal
pipeline, which keeps the video, the drum stem and the detector caches. The
notation is read afterwards (tools/truth_video); `compact` then drops what only
the reading needed, so thousands of songs fit on the disk.

    .venv/bin/python tools/harvest/harvest.py fetch --per-channel 5
    .venv/bin/python tools/harvest/harvest.py compact
    .venv/bin/python tools/harvest/harvest.py status

Everything stays on this machine (truth/ is not in git): the videos and their
notation belong to their channels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
import fcntl
import sys
from contextlib import contextmanager
from pathlib import Path
from manifest_io import read as read_manifest, append as append_manifest

ROOT = Path(__file__).resolve().parents[2]
CHANNELS = Path(__file__).with_name("channels.json")
OUT = ROOT / "truth" / "harvest"
SONGS = OUT / "songs"
MANIFEST = OUT / "manifest.jsonl"
LISTS = OUT / "lists"
# 1080p by default: at 720p many staffs fall under the reader's 8 px line spacing (truth/harvest/PILOT1080.md).
VIDEO_FORMAT = "bv*[height=1080][vcodec^=avc1]/bv*[height=1080]/bv*[height<=1080]"
MAX_HEIGHT = 1080
PIPELINE_HEIGHT = 720  # the pipeline's own video download (src/groovemap/fetch.py) is 720p
MIN_FREE_GB = 6.0  # stop fetching below this much free disk
DURATION = (75, 480)  # seconds: skips shorts and long streams


def video_format(max_height: int) -> str:
    if max_height == MAX_HEIGHT:
        return VIDEO_FORMAT
    return f"bv*[height<={max_height}][vcodec^=avc1]/bv*[height<={max_height}]"


def free_gb() -> float:
    # truth/ can live on a different filesystem from the source checkout.
    path = SONGS
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def command_env() -> dict[str, str]:
    """Use this interpreter's tools, including from a non-login SSH shell."""
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join([str(Path(sys.executable).absolute().parent),
                                  str(Path.home() / '.deno/bin'), env.get('PATH', '')])
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), env.get('PYTHONPATH', '')])
    return env


def executable(name: str) -> str:
    path = shutil.which(name, path=command_env()['PATH'])
    if not path:
        raise RuntimeError(f'{name} not found; install it in this venv or on PATH')
    return path


@contextmanager
def writer_lock():
    """POSIX flock works on macOS and WSL's Linux filesystem (use ~/groovemap)."""
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / '.wsl-mirror').exists():
        raise RuntimeError('WSL owns this manifest; run fetch/read/compact through harvest_wsl.sh')
    with (OUT / '.harvest-writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError('another harvest writer/snapshot/comparison is running') from e
        yield


def listing(channel: dict, refresh: bool = False) -> list[tuple[str, int, str]]:
    """The channel's uploads, newest first: (video id, seconds, title)."""
    LISTS.mkdir(parents=True, exist_ok=True)
    path = LISTS / f"{channel['name']}.txt"
    if channel.get("listing") == "file":  # hand-collected IDs (e.g. Instagram reels), used as is
        refresh = False
        if not path.exists():
            raise FileNotFoundError(f"{path}: listing 'file' needs this list")
    if refresh or not path.exists():
        # A channel may name playlists (e.g. a transcription playlist) instead of its whole upload list.
        # or a listing URL such as https://www.youtube.com/@handle/shorts.
        urls = [f"https://www.youtube.com/playlist?list={p}" for p in channel.get("playlists", [])] \
            or ([channel["listing_url"]] if channel.get("listing_url") else []) \
            or [f"https://www.youtube.com/channel/{channel['id']}/videos"]
        out = subprocess.run(
            [executable('yt-dlp'), "--flat-playlist", "--print", "%(id)s|%(duration)s|%(title)s", *urls],
            cwd=ROOT, env=command_env(), capture_output=True, text=True, check=True,
        ).stdout
        path.write_text(out)
    rows = []
    for line in path.read_text().splitlines():
        vid, seconds, title = line.split("|", 2)
        rows.append((vid, int(float(seconds)) if seconds not in ("", "NA", "None") else 0, title))
    return rows


DRUMLESS = json.loads(CHANNELS.read_text()).get("exclude")


def candidates(channel: dict) -> list[tuple[str, int, str]]:
    rows = listing(channel)
    rows = rows[: max(1, int(len(rows) * channel.get("newest", 1.0)))]
    keep = []
    for vid, seconds, title in rows:
        # 0 = duration unknown (a hand-collected list): no duration filter.
        if seconds and not channel.get("min", DURATION[0]) <= seconds <= channel.get("max", DURATION[1]):
            continue
        if "include" in channel and not re.search(channel["include"], title, re.I):
            continue
        if "exclude" in channel and re.search(channel["exclude"], title, re.I):
            continue
        if DRUMLESS and re.search(DRUMLESS, title, re.I):
            continue
        keep.append((vid, seconds, title))
    return keep


def manifest() -> dict[str, dict]:
    return read_manifest(MANIFEST)


def record(**row) -> None:
    append_manifest(MANIFEST, **row)


def fetch(per_channel: int, only: list[str] | None, seed: int, *, device: str = 'auto',
          min_free_gb: float = MIN_FREE_GB, limit: int | None = None,
          additional: bool = False, max_height: int = MAX_HEIGHT, ids: list[str] | None = None) -> None:
    """`ids` fetches exactly these videos (still looked up in each channel's listing), ignoring the per-channel
    target and the random choice. `max_height` other than 720 replaces the pipeline's 720p video."""
    config = json.loads(CHANNELS.read_text())["channels"]
    done = manifest()
    SONGS.mkdir(parents=True, exist_ok=True)
    executable('yt-dlp')
    executable('ffmpeg')
    attempted = 0
    failures = []
    for channel in config:
        if only and channel["name"] not in only:
            continue
        if channel.get("fetch_enabled") is False:  # e.g. reader on hold, or a test-only channel
            print(f"[{channel['name']}] fetch disabled: {channel.get('fetch_note', '')[:120]}", flush=True)
            continue
        have = sum(1 for row in done.values() if row.get("channel") == channel["name"] and row.get("status") != "failed")
        # Named IDs may retry an earlier failed fetch (e.g. a dropped YouTube download).
        pool = [c for c in candidates(channel)
                if c[0] not in done or (ids is not None and c[0] in ids and done[c[0]].get("status") == "failed")]
        if ids is not None:
            pool = sorted((c for c in pool if c[0] in ids), key=lambda c: ids.index(c[0]))
            chosen = pool
        else:
            random.Random(f"{seed}-{channel['name']}").shuffle(pool)
            chosen = pool[: per_channel if additional else max(0, per_channel - have)]
        for vid, seconds, title in chosen:
            if limit is not None and attempted >= limit:
                if failures:
                    raise RuntimeError(f'fetch failed for {failures}; see manifest errors')
                return
            if free_gb() < min_free_gb:
                print(f"stopping: {free_gb():.1f} GB free")
                if failures:
                    raise RuntimeError(f'fetch failed for {failures}; see manifest errors')
                return
            print(f"[{channel['name']}] {vid} {title[:70]}", flush=True)
            url = channel.get("url_template", "https://www.youtube.com/watch?v={id}").format(id=vid)
            if attempted and channel.get("sleep"):  # e.g. Instagram rate limits
                time.sleep(float(channel["sleep"]))
            for attempt in range(3):  # YouTube's CDN refuses now and then (HTTP 403)
                run = subprocess.run([sys.executable, '-m', 'groovemap.cli', "transcribe", url,
                                      "--out", str(SONGS), "--no-pdf", '--device', device],
                                     cwd=ROOT, env=command_env(), capture_output=True, text=True)
                if run.returncode == 0:
                    break
                if attempt < 2:
                    time.sleep(20 * (attempt + 1))
            slug = next((p.name for p in SONGS.iterdir() if p.name.endswith(vid)), None)
            video = SONGS / slug / "video.mp4" if slug else None
            video_error = ''
            # The pipeline fetched a 720p video; at another height it is replaced only once the new one has
            # arrived, so a failed re-download keeps the 720p video. A missing video is fetched at max_height.
            sharper = max_height != PIPELINE_HEIGHT and video is not None and video.exists() and run.returncode == 0
            target = video.with_name(f"video.{max_height}p.part.mp4") if sharper else video
            for attempt in range(3):  # the audio can arrive while the video download failed
                if not video or (target.exists() and target.stat().st_size > 0):
                    break
                target.unlink(missing_ok=True)
                video_run = subprocess.run([executable('yt-dlp'), "-q", "-f", channel.get("video_format") or video_format(max_height),
                                            "--merge-output-format", "mp4", "-o", str(target), url], cwd=ROOT, env=command_env(), capture_output=True, text=True)
                video_error = video_run.stderr or video_run.stdout
                if not target.exists() and attempt < 2:
                    time.sleep(20 * (attempt + 1))
            if sharper:
                if target.exists() and target.stat().st_size > 0:
                    os.replace(target, video)
                else:
                    target.unlink(missing_ok=True)
                    print(f"  {max_height}p video failed; keeping the 720p one: {video_error[-200:]}", flush=True)
            ok = run.returncode == 0 and video is not None and video.exists()
            record(video=vid, channel=channel["name"], layout=channel["layout"], title=title, seconds=seconds,
                   slug=slug, status="fetched" if ok else "failed",
                   error=None if ok else ((run.stderr or run.stdout) if run.returncode else video_error)[-400:])
            if not ok:
                failures.append(vid)
            attempted += 1
            done = manifest()
    if failures:
        raise RuntimeError(f'fetch failed for {failures}; see manifest errors')


# Kept after reading: what scoring and training need. Everything else goes.
KEEP = {"meta.json", "options.json", "beats.json", "score.json", "truth", ".truth.lock"}
KEEP_SUFFIX = (".npy",)


def compact(force: bool, only: list[str] | None = None) -> None:
    """Drop the video and full mix of every song whose notation has been read; shrink the stem."""
    rows = manifest()
    for row in rows.values():
        if only and row.get('channel') not in only:
            continue
        if row.get("status") != "read" and not (force and row.get("status") == "fetched"):
            continue
        folder = SONGS / (row.get("slug") or "")
        if not row.get("slug") or not folder.is_dir():
            continue
        if folder.resolve().parent != SONGS.resolve():
            raise ValueError('unsafe compact song path')
        with (folder/'.truth.lock').open('a') as lock:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: continue
            current=manifest().get(row['video'],{})
            if current.get('status')!='read' and not(force and current.get('status')=='fetched'):
                continue
            stems = list(folder.glob("drums.*.wav"))
            for stem in stems:
                opus = stem.with_suffix(".opus")
                temporary=stem.with_suffix('.compact.opus')
                subprocess.run([executable('ffmpeg'), "-loglevel", "error", "-y", "-i", str(stem), "-c:a", "libopus",
                                "-b:a", "128k", str(temporary)], check=True)
                temporary.replace(opus)
                stem.unlink()
            for path in folder.iterdir():
                # Preserve all directories (truth, history, interrupted work)
                # and the small source caches. Never recurse into truth.
                if path.is_dir() or path.name in KEEP or path.suffix in KEEP_SUFFIX or path.suffix == ".opus":
                    continue
                path.unlink()
            record(video=row["video"], status="compacted")


def status() -> None:
    rows = manifest()
    by = {}
    for row in rows.values():
        by.setdefault(row.get("channel"), {}).setdefault(row.get("status"), 0)
        by[row.get("channel")][row.get("status")] += 1
    for name, counts in sorted(by.items()):
        print(f"{name:14s} {counts}")
    size = sum(p.stat().st_size for p in SONGS.rglob("*") if p.is_file()) / 1e9 if SONGS.exists() else 0
    print(f"{len(rows)} videos, {size:.1f} GB in {SONGS}, {free_gb():.1f} GB free")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--per-channel", type=int, default=5)
    f.add_argument("--only", nargs="*")
    f.add_argument("--seed", type=int, default=1)
    f.add_argument('--device', choices=('auto', 'cuda', 'mps', 'cpu'), default='auto',
                   help='auto keeps the pipeline policy: MPS on Mac, CUDA on WSL when available')
    f.add_argument('--min-free-gb', type=float, default=MIN_FREE_GB)
    f.add_argument('--limit', type=int, help='maximum attempted videos in this invocation, across channels')
    f.add_argument('--additional', action='store_true', help='per-channel NEW videos, rather than a total target')
    f.add_argument('--max-height', type=int, default=MAX_HEIGHT,
                   help='video height (default 1080); other than 720 the pipeline\'s 720p video is fetched again at this height')
    f.add_argument('--ids', nargs='+', action='extend',
                   help='fetch only these video IDs (no random choice, no per-channel target); '
                        'an ID starting with "-" must be given as --ids=-ID')
    c = sub.add_parser("compact")
    c.add_argument("--force", action="store_true", help="also compact songs whose notation was not read")
    c.add_argument('--only', nargs='+', help='compact only these channels')
    sub.add_parser("status")
    ls = sub.add_parser("list", help="refresh channel listings (truth/harvest/lists) and show how many videos are eligible")
    ls.add_argument("--only", nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "fetch":
        if args.per_channel < 0 or (args.limit is not None and args.limit < 1):
            parser.error('per-channel must be nonnegative; limit must be positive')
        if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
            parser.error('min-free-gb must be finite and nonnegative')
        if args.max_height < 144:
            parser.error('max-height must be at least 144')
        with writer_lock():
            fetch(args.per_channel, args.only, args.seed, device=args.device,
                  min_free_gb=args.min_free_gb, limit=args.limit, additional=args.additional,
                  max_height=args.max_height, ids=args.ids)
    elif args.command == "list":
        done = manifest()
        for channel in json.loads(CHANNELS.read_text())["channels"]:
            if channel["name"] in args.only:
                rows = listing(channel, refresh=True)
                eligible = [c for c in candidates(channel) if c[0] not in done]
                print(f"{channel['name']}: {len(rows)} listed, {len(eligible)} eligible and not yet fetched", flush=True)
    elif args.command == "compact":
        with writer_lock():
            compact(args.force, args.only)
    else:
        status()
