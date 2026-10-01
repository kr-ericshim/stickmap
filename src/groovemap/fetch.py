"""Stage 1: get the song as a 44.1 kHz stereo WAV (YouTube via yt-dlp, or a local file)."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

SAMPLE_RATE = 44100


class FetchError(RuntimeError):
    pass


@dataclass
class SongMeta:
    kind: str  # "youtube" | "file"
    source: str  # URL or original file path
    title: str
    uploader: str = ""
    duration: float = 0.0
    video_id: str = ""
    fingerprint: str = ""  # content hash of a local source file

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SongMeta":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def slugify(text: str, max_len: int = 60) -> str:
    """Filesystem-safe slug that keeps non-ASCII letters (Korean/Japanese titles)."""
    slug = re.sub(r"[^\w]+", "-", text.strip().lower(), flags=re.UNICODE).strip("-_")
    return slug[:max_len].rstrip("-_") or "song"


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if not path:
        raise FetchError(f"`{tool}` not found on PATH (brew install {tool})")
    return path


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-5:])
        raise FetchError(f"{Path(cmd[0]).name} failed:\n{tail}")
    return proc.stdout


def search_youtube(query: str, limit: int = 5) -> list[SongMeta]:
    out = _run([_require("yt-dlp"), f"ytsearch{limit}:{query}", "--flat-playlist", "-J"])
    return [_meta_from_info(e) for e in json.loads(out).get("entries", []) if e.get("id")]


def youtube_info(url: str) -> SongMeta:
    return _meta_from_info(json.loads(_run([_require("yt-dlp"), "--no-playlist", "-J", url])))


def _meta_from_info(info: dict) -> SongMeta:
    video_id = info.get("id", "")
    return SongMeta(
        kind="youtube",
        source=info.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}",
        title=info.get("title") or video_id,
        uploader=info.get("channel") or info.get("uploader") or "",
        duration=float(info.get("duration") or 0),
        video_id=video_id,
    )


def download_audio(url: str, wav_path: Path) -> None:
    _require("ffmpeg")
    _run(
        [
            _require("yt-dlp"),
            "--no-playlist",
            "-f",
            "bestaudio/best",
            "-x",
            "--audio-format",
            "wav",
            "--postprocessor-args",
            f"ffmpeg:-ar {SAMPLE_RATE} -ac 2",
            "-o",
            str(wav_path.with_suffix(".%(ext)s")),
            url,
        ]
    )
    if not wav_path.exists():
        raise FetchError(f"yt-dlp finished but {wav_path.name} was not produced")


# H.264 + AAC plays in every browser's <video>; 720p keeps a song to tens of MB.
VIDEO_FORMAT = "bv*[height<=720][vcodec^=avc1]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]"


def download_video(url: str, mp4_path: Path) -> None:
    _require("ffmpeg")
    _run(
        [
            _require("yt-dlp"),
            "--no-playlist",
            "-f",
            VIDEO_FORMAT,
            "--merge-output-format",
            "mp4",
            "-o",
            str(mp4_path.with_suffix(".%(ext)s")),
            url,
        ]
    )
    if not mp4_path.exists():
        raise FetchError(f"yt-dlp finished but {mp4_path.name} was not produced")


def decode_file(src: Path, wav_path: Path) -> SongMeta:
    _run(
        [
            _require("ffmpeg"),
            "-y",
            "-v",
            "error",
            "-i",
            str(src),
            "-vn",
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "2",
            str(wav_path),
        ]
    )
    return SongMeta(kind="file", source=str(src.resolve()), title=src.stem)
