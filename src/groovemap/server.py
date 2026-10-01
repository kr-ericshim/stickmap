"""Local HTTP API with one FIFO transcription worker."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
import uuid
from _thread import LockType
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from pathlib import Path
from queue import Queue
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
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
    odd_bars: Literal["none", "auto", "detected"] = "none"
    beat_shift: Annotated[float, Field(ge=-11, le=11, multiple_of=0.25)] = 0
    tempo_scale: Literal[0.5, 1, 2] = 1
    voices: Literal[1, 2] = 1
    sensitivity: Annotated[float, Field(ge=0.5, le=5)] = 1
    sound_names: dict[str, str] = Field(default_factory=dict)

    @field_validator("time_signature")
    @classmethod
    def validate_signature(cls, value: str | None) -> str | None:
        if value is not None:
            from .metre import Signature

            Signature.parse(value)
        return value

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

    def start(self) -> None:
        self.thread = threading.Thread(target=self._work, name="groovemap-worker", daemon=True)
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
            "created_at": time.time(),
        }
        with self.lock:
            if unless_pending and any(
                other["slug"] == job["slug"] and other["status"] in ("queued", "running") for other in self.jobs.values()
            ):
                return None
            self.jobs[job["id"]] = job
            self.queue.put((job["id"], options, temporary, meta, song_dir))
            return deepcopy(job)

    def get(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise HTTPException(404, "job not found")
            return deepcopy(self.jobs[job_id])

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return deepcopy(list(reversed(self.jobs.values())))

    def _work(self) -> None:
        while (item := self.queue.get()) is not None:
            job_id, patch, temporary, meta, song_dir = item

            def progress(key: str, status: str, seconds: float | None = None) -> None:
                with self.lock:
                    stage = self.jobs[job_id]["stages"][STAGES.index(key)]
                    stage.update(status=status, seconds=seconds)

            try:
                with self.execution_lock:
                    with self.lock:
                        self.jobs[job_id]["status"] = "running"
                        source = self.jobs[job_id]["source"]
                    if song_dir is not None:
                        meta = SongMeta.load(song_dir / "meta.json")
                        with self.lock:
                            self.jobs[job_id].update(title=meta.title, slug=song_dir.name)
                        progress("fetch", "cached")
                    else:
                        progress("fetch", "running")
                        started = time.monotonic()
                        if fetch.is_url(source):
                            meta = meta or fetch.youtube_info(source)
                        with self.lock:
                            self.jobs[job_id]["title"] = meta.title if meta else Path(source).stem
                        song_dir, meta = library.prepare_song(source, self.songs_dir, meta=meta)
                        with self.lock:
                            self.jobs[job_id].update(title=meta.title, slug=song_dir.name)
                        progress("fetch", "done", time.monotonic() - started)

                    from . import pipeline  # torch is loaded only when there is work to do.

                    options = OptionsPatch.model_validate(
                        library.load_options(song_dir) | patch.model_dump(exclude_unset=True)
                    )
                    opts = pipeline.Options(**options.model_dump(), pdf=False, preview=False, refresh_stale=True)
                    pipeline.run(song_dir, meta, opts, progress=progress)
                    library.save_options(song_dir, opts)
                    with self.lock:
                        self.jobs[job_id]["status"] = "done"
            except Exception as error:
                with self.lock:
                    self.jobs[job_id].update(
                        status="error", error=" ".join(str(error).split())[:300] or type(error).__name__
                    )
            finally:
                if temporary is not None:
                    shutil.rmtree(temporary.parent, ignore_errors=True)
                with self.lock:
                    finished = [key for key, job in self.jobs.items() if job["status"] in ("done", "error")]
                    for key in finished[:-20]:
                        del self.jobs[key]
                self.queue.task_done()
        self.queue.task_done()


def create_app(songs_dir: Path, static_dir: Path | None = None) -> FastAPI:
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

    app = FastAPI(title="groovemap", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Range"],
    )

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
            "has_video": (path / "video.mp4").is_file(),
        }

    @contextmanager
    def idle() -> Iterator[None]:
        """Hold the pipeline lock, refusing instead of waiting behind a running job."""
        if not execution_lock.acquire(blocking=False):
            raise HTTPException(409, "채보 작업이 끝난 뒤에 다시 시도하세요")
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
    def update_score(slug: str, patch: OptionsPatch) -> dict[str, Any]:
        path = directory(slug)
        with execution_lock:
            options = OptionsPatch.model_validate(library.load_options(path) | patch.model_dump(exclude_unset=True))
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
        with execution_lock:
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
        return update_score(slug, OptionsPatch(sound_names=body.names))

    @app.post("/api/songs/{slug}/transcribe")
    def transcribe(slug: str, patch: OptionsPatch | None = None) -> dict[str, Any]:
        path = directory(slug)
        meta = SongMeta.load(path / "meta.json")
        return runner.enqueue(meta.source, patch or OptionsPatch(), song_dir=path, title=meta.title)

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

    @app.get("/api/songs/{slug}/audio")
    def audio(slug: str, stem: Literal["mix", "drums"] = "mix") -> FileResponse:
        path = directory(slug)
        if stem == "mix":
            wav = path / "audio.wav"
        else:
            stems = [p for p in path.glob("drums.*.wav") if p.is_file()]
            if not stems:
                raise HTTPException(404, "drum stem not found")
            wav = max(stems, key=lambda p: p.stat().st_mtime)
        return FileResponse(require_file(wav), media_type="audio/wav")

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

            with execution_lock:
                xml = require_file(path / f"{path.name}.musicxml")
                if not target.is_file() or target.stat().st_mtime < xml.stat().st_mtime:
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
        temporary = Path(tempfile.mkdtemp(prefix="groovemap-upload-")) / filename
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
