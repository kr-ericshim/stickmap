"""Fetch drum-notation videos from several channels for ground truth.

Each chosen video is transcribed into truth/harvest/songs/<slug> with the normal
pipeline, which keeps the video, the drum stem and the detector caches. The
notation is read afterwards (tools/truth_video); `compact` then drops what only
the reading needed, so thousands of songs fit on the disk.

    uv run python tools/harvest/harvest.py fetch --per-channel 5
    uv run python tools/harvest/harvest.py compact
    uv run python tools/harvest/harvest.py status

Everything stays on this machine (truth/ is not in git): the videos and their
notation belong to their channels.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import time
import fcntl
from pathlib import Path
from manifest_io import read as read_manifest, append as append_manifest

ROOT = Path(__file__).resolve().parents[2]
CHANNELS = Path(__file__).with_name("channels.json")
OUT = ROOT / "truth" / "harvest"
SONGS = OUT / "songs"
MANIFEST = OUT / "manifest.jsonl"
LISTS = OUT / "lists"
VIDEO_FORMAT = "bv*[height<=720][vcodec^=avc1]/bv*[height<=720]"
MIN_FREE_GB = 6.0  # stop fetching below this much free disk
DURATION = (75, 480)  # seconds: skips shorts and long streams


def free_gb() -> float:
    return shutil.disk_usage(ROOT).free / 1e9


def listing(channel: dict, refresh: bool = False) -> list[tuple[str, int, str]]:
    """The channel's uploads, newest first: (video id, seconds, title)."""
    LISTS.mkdir(parents=True, exist_ok=True)
    path = LISTS / f"{channel['name']}.txt"
    if refresh or not path.exists():
        out = subprocess.run(
            ["uv", "run", "yt-dlp", "--flat-playlist", "--print", "%(id)s|%(duration)s|%(title)s",
             f"https://www.youtube.com/channel/{channel['id']}/videos"],
            cwd=ROOT, capture_output=True, text=True, check=True,
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
        if not channel.get("min", DURATION[0]) <= seconds <= channel.get("max", DURATION[1]):
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


def fetch(per_channel: int, only: list[str] | None, seed: int) -> None:
    config = json.loads(CHANNELS.read_text())["channels"]
    done = manifest()
    SONGS.mkdir(parents=True, exist_ok=True)
    for channel in config:
        if only and channel["name"] not in only:
            continue
        have = sum(1 for row in done.values() if row.get("channel") == channel["name"] and row.get("status") != "failed")
        pool = [c for c in candidates(channel) if c[0] not in done]
        random.Random(f"{seed}-{channel['name']}").shuffle(pool)
        for vid, seconds, title in pool[: max(0, per_channel - have)]:
            if free_gb() < MIN_FREE_GB:
                print(f"stopping: {free_gb():.1f} GB free")
                return
            print(f"[{channel['name']}] {vid} {title[:70]}", flush=True)
            url = f"https://www.youtube.com/watch?v={vid}"
            for attempt in range(3):  # YouTube's CDN refuses now and then (HTTP 403)
                run = subprocess.run(["uv", "run", "groovemap", "transcribe", url, "--out", str(SONGS), "--no-pdf"],
                                     cwd=ROOT, capture_output=True, text=True)
                if run.returncode == 0:
                    break
                time.sleep(20 * (attempt + 1))
            slug = next((p.name for p in SONGS.iterdir() if p.name.endswith(vid)), None)
            video = SONGS / slug / "video.mp4" if slug else None
            for attempt in range(3):  # the audio can arrive while the video download failed
                if not video or video.exists():
                    break
                subprocess.run(["uv", "run", "yt-dlp", "-q", "-f", VIDEO_FORMAT, "--merge-output-format", "mp4",
                                "-o", str(video), url], cwd=ROOT, capture_output=True, text=True)
                if not video.exists():
                    time.sleep(20 * (attempt + 1))
            ok = run.returncode == 0 and video is not None and video.exists()
            record(video=vid, channel=channel["name"], layout=channel["layout"], title=title, seconds=seconds,
                   slug=slug, status="fetched" if ok else "failed",
                   error=None if ok else (run.stderr or run.stdout)[-400:])


# Kept after reading: what scoring and training need. Everything else goes.
KEEP = {"meta.json", "options.json", "beats.json", "score.json", "truth", ".truth.lock"}
KEEP_SUFFIX = (".npy",)


def compact(force: bool) -> None:
    """Drop the video and full mix of every song whose notation has been read; shrink the stem."""
    rows = manifest()
    for row in rows.values():
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
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(stem), "-c:a", "libopus",
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
    c = sub.add_parser("compact")
    c.add_argument("--force", action="store_true", help="also compact songs whose notation was not read")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch(args.per_channel, args.only, args.seed)
    elif args.command == "compact":
        compact(args.force)
    else:
        status()
