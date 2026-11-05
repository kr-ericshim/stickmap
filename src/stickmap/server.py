"""Local HTTP API with one FIFO transcription worker."""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import tempfile
import threading
import statistics
import time
import uuid
from _thread import LockType
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager, nullcontext
from copy import deepcopy
from pathlib import Path
from queue import Queue
from typing import Annotated, Any, Literal

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from starlette.concurrency import run_in_threadpool

from . import fetch, library
from .fetch import SongMeta

STAGES = ("fetch", "separate", "beats", "onsets", "score")


class OptionsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    grid: Literal[8, 16] = 16
    triplets: Literal["auto", "off"] = "auto"
    meter: Annotated[int, Field(ge=1, le=12)] | None = None
    time_signature: str | None = None
    meter_changes: list[dict[str, Any]] = Field(default_factory=list)
    odd_bars: Literal["none", "auto", "detected"] = "none"
    beat_shift: Annotated[float, Field(ge=-11, le=11, multiple_of=0.25)] = 0
    tempo_scale: Literal[0.5, 1, 2] = 1
    voices: Literal[1, 2] = 1
    sensitivity: Annotated[float, Field(ge=0.5, le=5)] = 1
    ride_runs: bool = True
    cymbal_veto: bool = True
    tom_rescue: bool = True
    sound_names: dict[str, str] = Field(default_factory=dict)

    @field_validator("time_signature")
    @classmethod
    def validate_signature(cls, value: str | None) -> str | None:
        if value is not None:
            from .metre import parse_signatures

            parse_signatures(value)
        return value

    @model_validator(mode="after")
    def validate_meter_changes(self) -> "OptionsPatch":
        if self.meter_changes:
            from .metre import parse_meter_changes

            if self.time_signature is None:
                raise ValueError("meter_changes need a time_signature")
            parse_meter_changes(self.meter_changes, self.time_signature)
        return self

    @field_validator("sound_names")
    @classmethod
    def validate_sound_names(cls, names: dict[str, str]) -> dict[str, str]:
        from .kit import NAMES

        if any(not group for group in names) or any(name not in NAMES for name in names.values()):
            raise ValueError(f"sound names map group ids to one of {', '.join(NAMES)}")
        return names

    @field_validator("grid", "voices", mode="before")
    @classmethod
    def require_integer(cls, value: Any) -> Any:
        if type(value) is not int:
            raise ValueError("must be an integer")
        return value


class RetranscribeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["outdated", "all"] = "outdated"


class SoundNamesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: dict[str, str]

    @field_validator("names")
    @classmethod
    def validate_names(cls, names: dict[str, str]) -> dict[str, str]:
        return OptionsPatch.validate_sound_names(names)


class BeatGrid(BaseModel):
    beats: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]]
    downbeats: list[Annotated[float, Field(ge=0, allow_inf_nan=False)]]

    @field_validator("beats", "downbeats")
    @classmethod
    def ordered_times(cls, times: list[float]) -> list[float]:
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("beat times must be strictly increasing")
        return times


# Everything the sound map is learned from and replays against; it never starts a missing model.
REFINEMENT_CACHES = (
    "drums.htdemucs.wav", "beats.json", "activations.drums-htdemucs.npy",
    "activations.mix.npy", "snare.htdemucs.npy", "attacks.htdemucs.npy",
)


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    source: str = Field(min_length=1)
    options: OptionsPatch = Field(default_factory=OptionsPatch)

    @field_validator("source")
    @classmethod
    def validate_source(cls, source: str) -> str:
        if not fetch.is_url(source) and not Path(source).is_absolute():
            raise ValueError("source must be a YouTube URL or an absolute local file path")
        return source


class ShutdownRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    force: bool = False


def stop_process(force: bool) -> None:
    """Stop this server the way Ctrl+C does; a forced stop does not wait for a running transcription."""
    threading.Timer(.3, os.kill, args=(os.getpid(), signal.SIGINT)).start()
    if force:
        watchdog = threading.Timer(8, os._exit, args=(0,))
        watchdog.daemon = True
        watchdog.start()


# Names this server answers to. Any web page can make the browser send requests to 127.0.0.1, and a
# page whose domain re-resolves to 127.0.0.1 (DNS rebinding) even reads the answers, so requests for
# other host names are refused, and so are changes sent from another site's page.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
ALLOWED_HOSTS = LOOPBACK_HOSTS


def _hostname(netloc: str) -> str:
    """'127.0.0.1:8765' -> '127.0.0.1', '[::1]:8765' -> '::1'."""
    if netloc.startswith("["):
        return netloc[1:].partition("]")[0].lower()
    return netloc.rpartition(":")[0].lower() if netloc.count(":") == 1 else netloc.lower()


def request_refusal(method: str, host: str, origin: str | None, hosts: frozenset[str] | None) -> str | None:
    """Why a request must be refused, or None. `hosts` None skips the host check (bound to every interface)."""
    if hosts is not None and _hostname(host) not in hosts:
        return "unknown host"
    if method in ("GET", "HEAD", "OPTIONS") or origin is None:
        return None
    site = origin.partition("://")[2]
    if site == host or _hostname(site) in LOOPBACK_HOSTS:  # same page, or the dev server on another port
        return None
    return "cross-site request"


BUSY = "지금 다른 곡을 채보하는 중입니다. 끝난 뒤 다시 시도하세요."

# (substrings in lower-cased error text, message); the first match wins, so specific causes come first.
ERROR_MESSAGES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("available in your country", "geo restrict", "geo-restrict", "geo_restrict", "blocked it in your country"),
     "이 지역에서 볼 수 없는 영상입니다"),
    (("confirm your age", "age-restricted", "age restricted", "inappropriate for some users", "sign in to confirm",
      "login required", "requires authentication", "members-only", "join this channel", "use --cookies"),
     "로그인이나 연령 확인이 필요한 영상이라 받을 수 없습니다"),
    (("private video", "video unavailable", "this video is unavailable", "has been removed", "been terminated",
      "video is not available", "http error 404", "is no longer available"),
     "비공개이거나 삭제된 영상입니다"),
    (("unable to download webpage", "name or service not known", "nodename nor servname", "name resolution",
      "failed to resolve", "getaddrinfo", "timed out", "timeout", "connection refused", "connection reset",
      "connection aborted", "network is unreachable", "no route to host", "urlopen error", "remote end closed",
      "ssl:", "max retries exceeded"),
     "네트워크에 연결할 수 없습니다"),
    (("ffmpeg", "ffprobe", "invalid data found", "error opening input", "could not decode", "decoding",
      "libsndfile", "format not recognised", "not a wav", "unknown format"),
     "오디오를 읽지 못했습니다"),
)


def humanize_error(error: BaseException) -> tuple[str, str]:
    """A short Korean message for the sidebar, and the raw text for a tooltip."""
    detail = " ".join(str(error).split())[:2000] or type(error).__name__
    lowered = detail.lower()
    for needles, message in ERROR_MESSAGES:
        if any(needle in lowered for needle in needles):
            return message, detail
    if isinstance(error, fetch.FetchError) and re.search(r"\byt-dlp\b", lowered):
        return "영상을 받지 못했습니다", detail
    return "채보 중 오류가 났습니다", detail


class JobCancelled(Exception):
    pass


def _audio_seconds(path: Path) -> float | None:
    try:
        import soundfile

        return float(soundfile.info(str(path)).duration) or None
    except Exception:
        return None


def _validation_message(errors: list[dict[str, Any]]) -> str:
    return "; ".join(
        f"{'.'.join(map(str, error['loc'])) or 'options'}: {error['msg']}" for error in errors
    )


class JobRunner:
    def __init__(self, songs_dir: Path, execution_lock: LockType) -> None:
        self.songs_dir = songs_dir
        self.execution_lock = execution_lock
        self.lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.queue: Queue[tuple[str, OptionsPatch, Path | None, SongMeta | None, Path | None] | None] = Queue()
        self.thread: threading.Thread | None = None
        self.cancelled: set[str] = set()  # running jobs asked to stop at the next stage boundary
        self.rates: list[float] = []  # processing seconds per audio second of finished jobs

    def start(self) -> None:
        self.thread = threading.Thread(target=self._work, name="stickmap-worker", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.queue.put(None)
        if self.thread is not None:
            self.thread.join()

    def enqueue(
        self, source: str, options: OptionsPatch, temporary: Path | None = None, meta: SongMeta | None = None,
        song_dir: Path | None = None, title: str | None = None,
    ) -> dict[str, Any]:
        """Queue a transcription. `song_dir` re-runs a song that is already in the library."""
        return self._add(source, options, temporary, meta, song_dir, title)  # type: ignore[return-value]

    def enqueue_unless_pending(self, song_dir: Path, title: str, source: str) -> dict[str, Any] | None:
        """Re-run a library song unless a job for it is already queued or running."""
        return self._add(source, OptionsPatch(), None, None, song_dir, title, unless_pending=True)

    def _add(
        self, source: str, options: OptionsPatch, temporary: Path | None, meta: SongMeta | None,
        song_dir: Path | None, title: str | None, unless_pending: bool = False,
    ) -> dict[str, Any] | None:
        job = {
            "id": uuid.uuid4().hex,
            "status": "queued",
            "source": source,
            "title": title,
            "slug": song_dir.name if song_dir is not None else None,
            "stages": [{"key": key, "status": "pending", "seconds": None} for key in STAGES],
            "error": None,
            "error_detail": None,
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "audio_seconds": _audio_seconds(song_dir / "audio.wav") if song_dir is not None else (
                meta.duration if meta is not None and meta.duration else None
            ),
        }
        with self.lock:
            if unless_pending and any(
                other["slug"] == job["slug"] and other["status"] in ("queued", "running") for other in self.jobs.values()
            ):
                return None
            self.jobs[job["id"]] = job
            self.queue.put((job["id"], options, temporary, meta, song_dir))
            return self._estimated()[job["id"]]

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(404, "job not found")
            return self._estimated()[job_id]

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(reversed(self._estimated().values()))

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Drop a queued job now; ask a running one to stop before its next stage."""
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "job not found")
            if job["status"] == "queued":
                job.update(status="cancelled", finished_at=time.time())
            elif job["status"] == "running":
                self.cancelled.add(job_id)
            elif job["status"] != "cancelled":
                raise HTTPException(409, "이미 끝난 작업입니다")
            return self._estimated()[job_id]

    def _check(self, job_id: str) -> None:
        with self.lock:
            if job_id in self.cancelled:
                raise JobCancelled

    def _estimated(self) -> dict[str, dict[str, Any]]:
        """Copies of the jobs with eta_seconds and queue_position; call with self.lock held."""
        rate = statistics.median(self.rates) if self.rates else None
        now = time.time()
        jobs = {key: deepcopy(job) | {"eta_seconds": None, "queue_position": None} for key, job in self.jobs.items()}
        ahead: float | None = 0.0  # seconds of work before the next queued job starts
        position = 0
        for job in jobs.values():  # insertion order is queue order
            if job["status"] not in ("queued", "running"):
                continue
            own = rate * job["audio_seconds"] if rate is not None and job["audio_seconds"] else None
            if job["status"] == "running":
                if own is not None and job["started_at"] is not None:
                    left = own - (now - job["started_at"])
                    job["eta_seconds"] = round(left) if left > 0 else None
                    own = max(left, 0)
            else:
                position += 1
                job["queue_position"] = position
                if ahead is not None and own is not None:
                    job["eta_seconds"] = round(ahead + own)
            ahead = ahead + own if ahead is not None and own is not None else None
        return jobs

    def _work(self) -> None:
        while (item := self.queue.get()) is not None:
            job_id, patch, temporary, meta, song_dir = item

            def progress(key: str, status: str, seconds: float | None = None) -> None:
                if status in ("running", "cached"):
                    self._check(job_id)  # stop between stages, before the next one writes anything
                with self.lock:
                    stage = self.jobs[job_id]["stages"][STAGES.index(key)]
                    stage.update(status=status, seconds=seconds)

            created: Path | None = None  # a song directory this job made, removed again if it is cancelled
            try:
                with self.execution_lock:
                    with self.lock:
                        if job_id not in self.jobs or self.jobs[job_id]["status"] == "cancelled":
                            continue  # cancelled while queued (and maybe already trimmed); finally still cleans up
                        self.jobs[job_id].update(status="running", started_at=time.time())
                        source = self.jobs[job_id]["source"]
                    if song_dir is not None:
                        meta = SongMeta.load(song_dir / "meta.json")
                        with self.lock:
                            self.jobs[job_id].update(title=library.display_title(meta, song_dir / "meta.json"), slug=song_dir.name)
                        progress("fetch", "cached")
                    else:
                        progress("fetch", "running")
                        started = time.monotonic()
                        if fetch.is_url(source):
                            meta = meta or fetch.youtube_info(source)
                        with self.lock:
                            self.jobs[job_id]["title"] = library.display_title(meta) if meta else Path(source).stem
                        before = {d.name for d in self.songs_dir.iterdir()} if self.songs_dir.is_dir() else set()
                        song_dir, meta = library.prepare_song(source, self.songs_dir, meta=meta)
                        if song_dir.name not in before:
                            created = song_dir
                        with self.lock:
                            self.jobs[job_id].update(title=library.display_title(meta, song_dir / "meta.json"), slug=song_dir.name)
                        progress("fetch", "done", time.monotonic() - started)
                    with self.lock:
                        self.jobs[job_id]["audio_seconds"] = _audio_seconds(song_dir / "audio.wav") or (
                            meta.duration if meta is not None and meta.duration else None
                        )
                    self._check(job_id)

                    from . import pipeline  # torch is loaded only when there is work to do.

                    options = OptionsPatch.model_validate(
                        library.load_options(song_dir) | patch.model_dump(exclude_unset=True)
                    )
                    opts = pipeline.Options(**options.model_dump(), pdf=False, preview=False, refresh_stale=True)
                    pipeline.run(song_dir, meta, opts, progress=progress)
                    self._check(job_id)
                    library.save_options(song_dir, opts)
                    with self.lock:
                        job = self.jobs[job_id]
                        job.update(status="done", finished_at=time.time())
                        if job["audio_seconds"] and job["started_at"] is not None:
                            self.rates.append((job["finished_at"] - job["started_at"]) / job["audio_seconds"])
            except JobCancelled:
                if created is not None:
                    shutil.rmtree(created, ignore_errors=True)  # only a song this job started; never older results
                with self.lock:
                    self.jobs[job_id].update(status="cancelled", finished_at=time.time())
            except Exception as error:
                message, detail = humanize_error(error)
                with self.lock:
                    self.jobs[job_id].update(status="error", error=message, error_detail=detail, finished_at=time.time())
            finally:
                if temporary is not None:
                    shutil.rmtree(temporary.parent, ignore_errors=True)
                with self.lock:
                    self.cancelled.discard(job_id)
                    finished = [key for key, job in self.jobs.items() if job["status"] in ("done", "error", "cancelled")]
                    for key in finished[:-20]:
                        del self.jobs[key]
                self.queue.task_done()
        self.queue.task_done()


def create_app(
    songs_dir: Path,
    static_dir: Path | None = None,
    shutdown: Callable[[bool], None] = stop_process,
    hosts: frozenset[str] | None | Literal["default"] = "default",
) -> FastAPI:
    allowed = ALLOWED_HOSTS if hosts == "default" else hosts
    songs_dir = songs_dir.resolve()
    execution_lock = threading.Lock()
    runner = JobRunner(songs_dir, execution_lock)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        songs_dir.mkdir(parents=True, exist_ok=True)
        runner.start()
        try:
            yield
        finally:
            await run_in_threadpool(runner.stop)

    app = FastAPI(title="stickmap", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Range"],
    )

    @app.middleware("http")
    async def local_only(request: Request, call_next: Callable[[Request], Any]) -> Response:
        refusal = request_refusal(
            request.method, request.headers.get("host", ""), request.headers.get("origin"), allowed
        )
        if refusal is not None:
            return JSONResponse({"detail": refusal}, status_code=403)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Any, error: RequestValidationError) -> JSONResponse:
        return JSONResponse({"detail": _validation_message(error.errors())}, status_code=422)

    def directory(slug: str) -> Path:
        try:
            path = library.song_path(songs_dir, slug)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        if not (path / "meta.json").is_file() or not (path / "audio.wav").is_file():
            raise HTTPException(404, "song not found")
        return path

    def song(path: Path) -> dict[str, Any]:
        score = path / "score.json"
        return library.summarize(path) | {
            "options": library.load_options(path),
            "score": json.loads(score.read_text(encoding="utf-8")) if score.is_file() else None,
            "has_drums": any(p.is_file() for p in path.glob("drums.*.wav")),
            "has_mr": (path / "drums.htdemucs.wav").is_file(),
            "has_video": (path / "video.mp4").is_file(),
        }

    @contextmanager
    def idle() -> Iterator[None]:
        """Hold the pipeline lock, refusing instead of waiting behind a running job."""
        if not execution_lock.acquire(blocking=False):
            raise HTTPException(409, BUSY)
        try:
            yield
        finally:
            execution_lock.release()

    def require_file(path: Path) -> Path:
        if not path.is_file():
            raise HTTPException(404, "file not found")
        return path

    @app.get("/api/songs")
    def list_songs() -> list[dict[str, Any]]:
        return library.list_songs(songs_dir)

    @app.get("/api/songs/{slug}")
    def get_song(slug: str) -> dict[str, Any]:
        return song(directory(slug))

    @app.post("/api/songs/{slug}/score")
    def update_score(slug: str, patch: Annotated[dict[str, Any], Body()]) -> dict[str, Any]:
        path = directory(slug)
        with idle():
            # Validate the merged options, not the patch alone: a patch may carry
            # only meter changes for the signature already saved with the song.
            try:
                options = OptionsPatch.model_validate_json(json.dumps(library.load_options(path) | patch))
            except ValidationError as error:
                raise RequestValidationError(error.errors()) from error
            # Signature corrections and sound names must never start a missing slow model.
            cached = REFINEMENT_CACHES if options.time_signature is not None or options.sound_names else REFINEMENT_CACHES[:3]
            require_caches(path, cached)

            from . import pipeline

            opts = pipeline.Options(**options.model_dump(), pdf=False, preview=False)
            try:
                pipeline.run(path, SongMeta.load(path / "meta.json"), opts)
                library.save_options(path, opts)
            except Exception as error:
                raise HTTPException(500, " ".join(str(error).split())[:300] or "score export failed") from error
            return song(path)

    def require_caches(path: Path, names: tuple[str, ...]) -> None:
        if not all((path / name).is_file() for name in names):
            raise HTTPException(409, "캐시가 지워져 있습니다. ‘다시 채보’를 눌러 분석을 다시 돌리세요")

    def sound_groups(path: Path, sound_map: Any, names: dict[str, str]) -> dict[str, Any]:
        groups = [
            {key: value for key, value in vars(group).items() if key != "centre"}
            | {"examples": [vars(example) for example in group.examples], "name": names.get(group.id)}
            for group in sound_map.groups
        ]
        return {"groups": groups, "has_video": (path / "video.mp4").is_file()}

    @app.get("/api/songs/{slug}/sounds")
    def sounds(slug: str) -> dict[str, Any]:
        """The song's sound groups, learned on first request and kept while its caches are unchanged."""
        path = directory(slug)
        with idle():
            options = OptionsPatch.model_validate(library.load_options(path))
            require_caches(path, REFINEMENT_CACHES)

            from . import kit, pipeline

            opts = pipeline.Options(**options.model_dump(), pdf=False, preview=False)
            paths = pipeline.cache_paths(path, opts)
            hits, _, attacks = pipeline.detect_hits(path, opts)
            sound_map = kit.load_or_learn(path, paths["drums"], paths["activations"], paths["attacks"], hits, attacks)
        return sound_groups(path, sound_map, options.sound_names)

    @app.post("/api/songs/{slug}/sounds")
    def name_sounds(slug: str, body: SoundNamesRequest) -> dict[str, Any]:
        """Replace the song's sound names ({} clears them) and re-quantize with them."""
        path = directory(slug)
        from .kit import SoundMap

        sound_map = SoundMap.load(path)
        if sound_map is None:
            raise HTTPException(409, "sound groups have not been learned yet: GET the song's sounds first")
        unknown = sorted(set(body.names) - {group.id for group in sound_map.groups})
        if unknown:
            raise HTTPException(422, f"unknown sound group: {', '.join(unknown)}")
        return update_score(slug, {"sound_names": body.names})

    @app.post("/api/songs/{slug}/transcribe")
    def transcribe(slug: str, patch: OptionsPatch | None = None) -> dict[str, Any]:
        path = directory(slug)
        meta = SongMeta.load(path / "meta.json")
        return runner.enqueue(meta.source, patch or OptionsPatch(), song_dir=path, title=library.display_title(meta, path / "meta.json"))

    @app.post("/api/songs/retranscribe")
    def retranscribe(body: RetranscribeRequest | None = None) -> list[dict[str, Any]]:
        """Queue every song (or every outdated one) again with its saved options."""
        scope = (body or RetranscribeRequest()).scope
        jobs = []
        for summary in library.list_songs(songs_dir):
            if scope == "outdated" and not summary["outdated"]:
                continue
            try:
                path = directory(summary["slug"])
                source = SongMeta.load(path / "meta.json").source
            except (HTTPException, OSError, ValueError, TypeError, KeyError):
                continue  # deleted or rewritten meanwhile; the rest of the library still gets queued
            job = runner.enqueue_unless_pending(path, summary["title"], source)
            if job is not None:
                jobs.append(job)
        return jobs

    @app.get("/api/songs/{slug}/musicxml")
    def musicxml(slug: str) -> Response:
        path = directory(slug)
        text = require_file(path / f"{path.name}.musicxml").read_text(encoding="utf-8")
        return Response(text, media_type="application/vnd.recordare.musicxml+xml")

    @app.get("/api/songs/{slug}/beats")
    def beats(slug: str) -> dict[str, Any]:
        """Read the tracker clock for practice clicks; never start a missing model."""
        path = require_file(directory(slug) / "beats.json")
        # A plain read: it must keep working while another song is being transcribed.
        try:
            return BeatGrid.model_validate_json(path.read_text(encoding="utf-8")).model_dump()
        except (OSError, ValueError) as error:
            raise HTTPException(409, "박자 캐시를 읽지 못했습니다. ‘다시 채보’를 눌러 분석을 다시 돌리세요") from error

    @app.get("/api/songs/{slug}/audio")
    def audio(slug: str, stem: Literal["mix", "drums", "nodrums"] = "mix") -> FileResponse:
        path = directory(slug)
        if stem == "mix":
            wav = path / "audio.wav"
        elif stem == "nodrums" and (cached := library.backing_track_cached(path)) is not None:
            wav = cached
        elif stem == "nodrums":
            with idle():
                require_file(path / "drums.htdemucs.wav")
                try:
                    wav = library.backing_track(path)
                except fetch.FetchError as error:
                    raise HTTPException(503, str(error)) from error
        else:
            stems = [p for p in path.glob("drums.*.wav") if p.is_file()]
            if not stems:
                raise HTTPException(404, "drum stem not found")
            wav = max(stems, key=lambda p: p.stat().st_mtime)
        return FileResponse(require_file(wav), media_type="audio/mp4" if wav.suffix == ".m4a" else "audio/wav")

    @app.get("/api/songs/{slug}/video")
    def video(slug: str) -> FileResponse:
        return FileResponse(require_file(directory(slug) / "video.mp4"), media_type="video/mp4")

    @app.delete("/api/songs/{slug}/cache")
    def clear_cache(slug: str) -> dict[str, Any]:
        path = directory(slug)
        with idle():
            library.clear_cache(path)
        return song(path)

    @app.delete("/api/songs/{slug}", status_code=204)
    def delete_song(slug: str) -> Response:
        path = directory(slug)
        with idle():
            library.delete_song(path)
        return Response(status_code=204)

    @app.get("/api/songs/{slug}/download/{kind}")
    def download(slug: str, kind: Literal["musicxml", "midi", "pdf"]) -> FileResponse:
        path = directory(slug)
        suffix = "mid" if kind == "midi" else kind
        target = path / f"{path.name}.{suffix}"
        if kind == "pdf":
            from . import render

            xml = require_file(path / f"{path.name}.musicxml")
            fresh = target.is_file() and target.stat().st_mtime >= xml.stat().st_mtime
            with nullcontext() if fresh else idle():
                if not fresh:
                    try:
                        rendered = render.render_pdf(xml, target)
                    except OSError as error:
                        raise HTTPException(503, "PDF engraving unavailable: MuseScore could not be started") from error
                    if not rendered:
                        raise HTTPException(503, "PDF engraving unavailable: install MuseScore or check its engraving output")
        return FileResponse(require_file(target), filename=target.name)

    @app.get("/api/search")
    def search(q: str = "") -> list[dict[str, Any]]:
        if not q.strip():
            raise HTTPException(400, "search query must not be empty")
        try:
            results = fetch.search_youtube(q.strip(), 8)
        except fetch.FetchError as error:
            raise HTTPException(502, str(error)) from error
        return [
            {"video_id": meta.video_id, "title": meta.title, "uploader": meta.uploader,
             "duration": meta.duration, "url": meta.source}
            for meta in results[:8]
        ]

    @app.post("/api/jobs")
    def enqueue(body: JobRequest) -> dict[str, Any]:
        return runner.enqueue(body.source, body.options)

    @app.post("/api/upload")
    def upload(
        file: Annotated[UploadFile, File()], options: Annotated[str | None, Form()] = None
    ) -> dict[str, Any]:
        try:
            patch = OptionsPatch.model_validate_json(options) if options is not None else OptionsPatch()
        except ValidationError as error:
            raise HTTPException(422, _validation_message(error.errors())) from error
        filename = Path((file.filename or "audio.wav").replace("\\", "/")).name
        if filename in ("", ".", ".."):
            raise HTTPException(400, "invalid upload filename")
        temporary = Path(tempfile.mkdtemp(prefix="stickmap-upload-")) / filename
        try:
            with temporary.open("wb") as destination:
                shutil.copyfileobj(file.file, destination)
            meta = SongMeta(kind="file", source=str(temporary), title=temporary.stem)
            return runner.enqueue(str(temporary), patch, temporary, meta)
        except Exception:
            shutil.rmtree(temporary.parent, ignore_errors=True)
            raise

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return runner.list()

    @app.get("/api/jobs/{id}")
    def get_job(id: str) -> dict[str, Any]:
        return runner.get(id)

    @app.delete("/api/jobs/{id}")
    def cancel_job(id: str) -> dict[str, Any]:
        return runner.cancel(id)

    @app.post("/api/shutdown")
    def shutdown_server(body: ShutdownRequest) -> dict[str, Any]:
        # The JSON body forces a CORS preflight (FastAPI ignores non-JSON bodies), so other sites cannot stop the server.
        active = [job for job in runner.list() if job["status"] in ("queued", "running")]
        if active and not body.force:
            raise HTTPException(409, f"채보 작업 {len(active)}개가 진행 중입니다")
        shutdown(body.force)
        return {"stopping": True, "interrupted": len(active)}

    if static_dir is not None and static_dir.is_dir():
        static_root = static_dir.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def static(path: str) -> FileResponse:
            if path == "api" or path.startswith("api/"):
                raise HTTPException(404, "API endpoint not found")
            target = (static_root / path).resolve()
            if not target.is_relative_to(static_root):
                raise HTTPException(404, "file not found")
            return FileResponse(require_file(target if target.is_file() else static_root / "index.html"))

    return app
