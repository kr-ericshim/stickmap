"""Non-interactive song preparation, saved score options, and library summaries."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import fetch, fingerprint
from .fetch import FetchError, SongMeta

if TYPE_CHECKING:
    from .pipeline import Options

DEFAULT_OPTIONS = {
    "grid": 16,
    "triplets": "auto",
    "meter": None,
    "time_signature": None,
    "meter_changes": [],  # [{"time": seconds, "time_signature": "6/8"}]: bars from that point use that signature
    "odd_bars": "none",
    "beat_shift": 0,
    "tempo_scale": 1,
    "voices": 1,
    "sensitivity": 1,
    "ride_runs": True,
    "cymbal_veto": True,
    "sound_names": {},  # sound group id -> instrument or "ignore"; replayed from soundmap.json (see kit.py)
}

# Regenerable model outputs: what "clear cache" removes.
CACHE_PATTERNS = (
    "drums.*.wav", "nodrums.*.wav", "nodrums.*.m4a", "beats.json", "activations.*.npy", "adtof_input.*.npy", "snare.*.npy", "attacks.*.npy", "cache.json", "preview.wav", "*.pdf",
)


# Words in YouTube titles that describe the video, not the song.
_NOISE_WORDS = (
    r"drum\s*(?:cover|cam|playthrough|play[-\s]?along|chart|score|transcription|tutorial)s?",
    r"(?:full\s+)?playthrough", r"official\s+(?:music\s+)?video", r"official\s+audio",
    r"music\s+video", r"lyric\s+video", r"m/?v", r"드럼\s*(?:커버|연주|악보|채보)",
)
# Too common in real song names ("Cover Me", "Live Wire") to drop outside brackets.
_BRACKET_WORDS = (r"cover", r"live", r"커버", r"drums?", r"드럼", r"shorts")
_NOISE = re.compile(rf"(?<![\w])(?:{'|'.join(_NOISE_WORDS)})(?![\w])", re.IGNORECASE)
_BRACKET_NOISE = re.compile(rf"(?<![\w])(?:{'|'.join(_BRACKET_WORDS)})(?![\w])", re.IGNORECASE)
_BRACKETS = re.compile(r"\(([^()]*)\)|\[([^\[\]]*)\]|【([^【】]*)】|「([^「」]*)」|〔([^〔〕]*)〕")
_HASHTAG = re.compile(r"(?<![\w&])#[^\s#|]+")
_SEPARATORS = "|-–—:/·•~,"
_EDGE = re.compile(rf"^[\s{re.escape(_SEPARATORS)}]+|[\s{re.escape(_SEPARATORS)}]+$")


def _is_noise(text: str) -> bool:
    """Bracketed text made only of noise words and separators, e.g. "Drum Cover" or "MV / 드럼커버"."""
    words = _BRACKET_NOISE.sub("", _NOISE.sub("", _HASHTAG.sub("", text)))
    return not re.sub(rf"[\s{re.escape(_SEPARATORS)}&+]", "", words)


def clean_title(raw: str, channel: str = "") -> str:
    """Display title for a YouTube title: drop hashtags, cover/MV tags and a trailing "| channel".

    The raw title stays in meta.json; this only shapes what is shown. Never returns an empty string.
    """
    text = _HASHTAG.sub(" ", raw)
    # Brackets holding only noise go; brackets with real words ("(feat. X)", "[Live at Y]") stay.
    text = _BRACKETS.sub(lambda m: " " if _is_noise(next(g for g in m.groups() if g is not None)) else m.group(0), text)
    channel = channel.strip()
    if channel:
        tail = re.compile(rf"\s*(?:[|\-–—:/]|\bby\b)\s*{re.escape(channel)}\s*$", re.IGNORECASE)
        text = tail.sub("", text)
    text = _NOISE.sub(" ", text)
    # Separators left with nothing between them ("Song |  | x" -> "Song | x").
    text = re.sub(r"\(\s*\)|\[\s*\]|【\s*】", " ", text)
    text = re.sub(rf"(\s*[{re.escape(_SEPARATORS)}]\s*)(?:[{re.escape(_SEPARATORS)}]\s*)+", r"\1", text)
    text = _EDGE.sub("", re.sub(r"\s+", " ", text)).strip()
    text = re.sub(r"\s+([,:])", r"\1", text)
    return text or raw.strip() or raw


def display_title(meta: SongMeta, meta_path: Path | None = None) -> str:
    """A title the user set (meta.json "title_override") wins; otherwise the cleaned title."""
    if meta_path is not None:
        try:
            override = json.loads(meta_path.read_text(encoding="utf-8")).get("title_override")
        except (OSError, ValueError, AttributeError):
            override = None
        if isinstance(override, str) and override.strip():
            return override.strip()
    if meta.kind != "youtube":
        return meta.title
    return clean_title(meta.title, meta.uploader)


def song_path(out: Path, slug: str) -> Path:
    """Resolve a slug to a direct child of out, including after symlink resolution."""
    if not slug or slug in (".", "..") or any(c in slug for c in ("/", "\\", "\0")):
        raise ValueError("invalid song slug")
    root = out.resolve()
    path = (root / slug).resolve()
    if Path(slug).is_absolute() or path.parent != root:
        raise ValueError("invalid song slug")
    return path


def _publish(song_dir: Path, name: str, produce: Callable[[Path], object]) -> None:
    """Build a file under a temporary name; it only gets its real name once complete."""
    part = song_dir / f".part.{name}"
    part.unlink(missing_ok=True)
    try:
        produce(part)
        if not part.is_file():
            raise FetchError(f"{name} was not produced")
        part.replace(song_dir / name)
    finally:
        part.unlink(missing_ok=True)


def backing_track_cached(song_dir: Path) -> Path | None:
    """The encoded backing track when it is already up to date, without encoding anything."""
    audio, drums = song_dir / "audio.wav", song_dir / "drums.htdemucs.wav"
    target = song_dir / "nodrums.htdemucs.m4a"
    try:
        newest = max(audio.stat().st_mtime_ns, drums.stat().st_mtime_ns)
        return target if target.stat().st_mtime_ns >= newest else None
    except OSError:
        return None


def backing_track(song_dir: Path) -> Path:
    """Cache an AAC backing track; migrate a valid old WAV only after publishing it."""
    audio, drums = song_dir / "audio.wav", song_dir / "drums.htdemucs.wav"
    target = song_dir / "nodrums.htdemucs.m4a"
    legacy = song_dir / "nodrums.htdemucs.wav"
    newest = max(audio.stat().st_mtime_ns, drums.stat().st_mtime_ns)
    if target.is_file() and target.stat().st_mtime_ns >= newest:
        legacy.unlink(missing_ok=True)
        return target
    valid_legacy = legacy.is_file() and legacy.stat().st_mtime_ns >= newest

    def encode(part: Path) -> None:
        data = None
        if valid_legacy:
            source = ["-i", str(legacy)]
        else:
            samples, sr = _backing_samples(audio, drums)
            source = ["-f", "f32le", "-ar", str(sr), "-ac", str(samples.shape[1]), "-i", "pipe:0"]
            data = samples.astype("<f4", copy=False).tobytes()
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *source,
             "-vn", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", "-f", "ipod", str(part)],
            input=data, capture_output=True, check=True,
        )

    try:
        _publish(song_dir, target.name, encode)
    except (OSError, subprocess.CalledProcessError) as error:
        if valid_legacy:
            return legacy
        raise FetchError("MR compression failed: check that ffmpeg supports the AAC encoder") from error
    legacy.unlink(missing_ok=True)
    return target


def _backing_samples(audio: Path, drums: Path) -> tuple[Any, int]:
    """Subtract in PCM before compression, preserving the mix's length and channels."""

    import numpy as np
    import soundfile as sf

    mix, sr = sf.read(audio, dtype="float32", always_2d=True)
    stem, stem_sr = sf.read(drums, dtype="float32", always_2d=True)
    if stem_sr != sr:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(sr, stem_sr)
        stem = resample_poly(stem, sr // divisor, stem_sr // divisor, axis=0)
    if mix.shape[1] == 1:
        stem = stem.mean(axis=1, keepdims=True)
    elif stem.shape[1] == 1:
        stem = np.repeat(stem, mix.shape[1], axis=1)
    elif stem.shape[1] != mix.shape[1]:
        raise ValueError("mix and drum stem have incompatible channel counts")

    # A short stem leaves the mix's tail intact; a long one stops at the mix's end.
    frames = min(len(mix), len(stem))
    mix[:frames] -= stem[:frames]
    peak = np.abs(mix).max(initial=0)
    if peak > 0.999:
        mix *= 0.999 / peak
    return mix, sr


def file_fingerprint(path: Path) -> str:
    """A short digest of a file's bytes: whether two recordings, or two cache versions, are the same."""
    digest = hashlib.sha1()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def _saved_meta(song_dir: Path) -> SongMeta | None:
    try:
        return _load_meta(song_dir / "meta.json")
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def prepare_song(source: str, out: Path, *, meta: SongMeta | None = None) -> tuple[Path, SongMeta]:
    """Resolve a URL or file into audio.wav (+ video.mp4) + meta.json; never perform a search."""
    out.mkdir(parents=True, exist_ok=True)
    if fetch.is_url(source):
        meta = meta or fetch.youtube_info(source)
        existing = sorted(
            d for d in out.iterdir()
            if d.is_dir() and d.name.endswith(f"-{meta.video_id}") and (d / "audio.wav").is_file()
        )
        slug = existing[0].name if existing else f"{fetch.slugify(meta.title)}-{meta.video_id}"
        song_dir = song_path(out, slug)
        song_dir.mkdir(exist_ok=True)
        video = song_dir / "video.mp4"
        if not video.is_file():
            try:
                _publish(song_dir, "video.mp4", lambda part: fetch.download_video(meta.source, part))
            except FetchError:
                pass  # no browser-playable format: the song still works with audio only
        if not (song_dir / "audio.wav").is_file():
            if video.is_file():  # same file, so the score and the picture share a timeline
                _publish(song_dir, "audio.wav", lambda part: fetch.decode_file(video, part))
            else:
                _publish(song_dir, "audio.wav", lambda part: fetch.download_audio(meta.source, part))
    else:
        local = Path(source).expanduser()
        if not local.is_file():
            raise FetchError(f"audio file not found: {source}")
        fingerprint = file_fingerprint(local)
        slug = fetch.slugify(meta.title if meta else local.stem)
        saved = _saved_meta(song_path(out, slug))
        if saved is not None and saved.fingerprint not in ("", fingerprint):
            slug = f"{slug}-{fingerprint[:6]}"  # same name, different recording
            saved = _saved_meta(song_path(out, slug))
        song_dir = song_path(out, slug)
        song_dir.mkdir(exist_ok=True)
        if not (song_dir / "audio.wav").is_file():
            _publish(song_dir, "audio.wav", lambda part: fetch.decode_file(local, part))
        meta = meta or saved or SongMeta(kind="file", source=str(local.resolve()), title=local.stem)
        meta.fingerprint = fingerprint

    _save_meta(meta, song_dir / "meta.json")
    return song_dir, meta


def _save_meta(meta: SongMeta, path: Path) -> None:
    """Write meta.json, keeping fields SongMeta does not hold (a user's "title_override")."""
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    extra = {key: value for key, value in old.items() if key not in SongMeta.__dataclass_fields__} if isinstance(old, dict) else {}
    meta.save(path)
    if extra:
        data = json.loads(path.read_text(encoding="utf-8")) | extra
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def cache_files(song_dir: Path) -> list[Path]:
    return [path for pattern in CACHE_PATTERNS for path in song_dir.glob(pattern) if path.is_file()]


def clear_cache(song_dir: Path) -> None:
    for path in cache_files(song_dir):
        path.unlink()


def delete_song(song_dir: Path) -> None:
    shutil.rmtree(song_dir)


def load_options(song_dir: Path) -> dict[str, Any]:
    path = song_dir / "options.json"
    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if saved.pop("follow_downbeats", False):  # written before odd_bars existed
        saved.setdefault("odd_bars", "detected")
    return {key: saved.get(key, deepcopy(default)) for key, default in DEFAULT_OPTIONS.items()}


def save_options(song_dir: Path, options: Options | Mapping[str, Any]) -> None:
    saved: dict[str, Any] = {
        key: options[key] if isinstance(options, Mapping) else getattr(options, key)
        for key in DEFAULT_OPTIONS
    }
    # Saved after every score, so it also records the code that made the score.
    fingerprints = fingerprint.snapshot(check=True)
    saved["code_version"] = fingerprints["score"] if fingerprints else None
    path = song_dir / "options.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    temporary.replace(path)


def code_version(song_dir: Path) -> str | None:
    """The fingerprint of the code the current score was made with; None if unknown."""
    try:
        return json.loads((song_dir / "options.json").read_text(encoding="utf-8")).get("code_version")
    except (OSError, ValueError, AttributeError):
        return None


def cache_stamps(song_dir: Path) -> dict[str, str | None]:
    """Cache file name -> fingerprint of the code that made it (see fingerprint.cache_stamp)."""
    try:
        stamps = json.loads((song_dir / "cache.json").read_text(encoding="utf-8"))
        return stamps if isinstance(stamps, dict) else {}
    except (OSError, ValueError):
        return {}


def stamp_cache(song_dir: Path, name: str, stamp: str | None) -> None:
    stamps = cache_stamps(song_dir) | {name: stamp}
    path = song_dir / "cache.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(stamps, indent=1), encoding="utf-8")
    temporary.replace(path)


def stale_caches(song_dir: Path) -> list[str]:
    """Cached model outputs made by other code than this process runs, or by unknown code."""
    fingerprints = fingerprint.snapshot()
    stamps = cache_stamps(song_dir)
    return [
        path.name for path in cache_files(song_dir)
        if (expected := fingerprint.cache_stamp(path.name, fingerprints)) is not None and stamps.get(path.name) != expected
    ]


def is_outdated(song_dir: Path) -> bool:
    """Re-transcribing would change the score: its code, or the code behind its caches, changed."""
    return code_version(song_dir) != fingerprint.snapshot()["score"] or bool(stale_caches(song_dir))


def _load_meta(path: Path) -> SongMeta:
    """SongMeta from meta.json, ignoring fields SongMeta does not know (such as "title_override")."""
    data = json.loads(path.read_text(encoding="utf-8"))
    known = SongMeta.__dataclass_fields__
    return SongMeta(**{key: value for key, value in data.items() if key in known})


def summarize(song_dir: Path) -> dict[str, Any]:
    meta_path = song_dir / "meta.json"
    meta = _load_meta(meta_path)
    title = display_title(meta, meta_path)
    score_path = song_dir / "score.json"
    score = json.loads(score_path.read_text(encoding="utf-8")) if score_path.is_file() else None
    return {
        "slug": song_dir.name,
        "title": title,
        # The YouTube title as fetched, when the shown one differs.
        **({"raw_title": meta.title} if title != meta.title else {}),
        "uploader": meta.uploader,
        "duration": meta.duration,
        "kind": meta.kind,
        "video_id": meta.video_id,
        "transcribed": score is not None,
        # Made by older code: re-transcribing picks up detector and notation changes.
        "outdated": score is not None and is_outdated(song_dir),
        "bars": len(score["bars"]) if score is not None else None,
        "tempo": score["tempo"] if score is not None else None,
        "updated_at": (score_path if score is not None else meta_path).stat().st_mtime,
        "size_bytes": sum(path.stat().st_size for path in song_dir.iterdir() if path.is_file()),
        "cache_bytes": sum(path.stat().st_size for path in cache_files(song_dir)),
    }


def list_songs(out: Path) -> list[dict[str, Any]]:
    songs = []
    for meta_path in out.glob("*/meta.json"):
        try:
            directory = song_path(out, meta_path.parent.name)
            if (directory / "audio.wav").is_file():
                songs.append(summarize(directory))
        except (ValueError, OSError, KeyError, TypeError):
            continue  # Ignore incomplete or malformed library entries.
    return sorted(songs, key=lambda song: song["updated_at"], reverse=True)
