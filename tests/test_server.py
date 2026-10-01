import json
import os
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from groovemap import fetch, fingerprint, library, pipeline, render
from groovemap.fetch import SongMeta
from groovemap.score import Bar, Score
from groovemap.server import STAGES, create_app


def write_wav(path):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(44100)
        wav.writeframes(b"\0" * 400)


def write_score(path, opts=None):
    score = Score("Test groove", [Bar(0, 4, [0, .5, 1, 1.5, 2], [4] * 4, 120)],
                  grid=opts.grid if opts else 16)
    score.save(path / "score.json")
    (path / f"{path.name}.musicxml").write_text("<score-partwise/>")
    (path / f"{path.name}.mid").write_bytes(b"MThd")
    return score


def cached_outputs(path):
    write_wav(path / "drums.htdemucs.wav")
    (path / "beats.json").write_text('{"beats": [0, 0.5], "downbeats": [0]}')
    np.save(path / "activations.drums-htdemucs.npy", np.zeros((2, 5)))
    np.save(path / "activations.mix.npy", np.zeros((2, 5)))
    np.save(path / "snare.htdemucs.npy", np.zeros((0, 2)))
    np.save(path / "attacks.htdemucs.npy", np.zeros(0))
    stamp_current(path)


def stamp_current(path):
    """Mark the cached model outputs as made by the code under test."""
    for cached in library.cache_files(path):
        stamp = fingerprint.cache_stamp(cached.name, fingerprint.snapshot())
        if stamp is not None:
            library.stamp_cache(path, cached.name, stamp)


def make_song(out, slug="groove", *, transcribed=True):
    path = out / slug
    path.mkdir(parents=True)
    write_wav(path / "audio.wav")
    SongMeta("file", "/original/groove.wav", "Test groove", duration=2).save(path / "meta.json")
    if transcribed:
        cached_outputs(path)
        write_score(path)
    return path


@pytest.fixture
def fakes(monkeypatch):
    calls = []

    def decode(source, destination):
        assert source.is_file()
        write_wav(destination)
        return SongMeta("file", str(source), source.stem)

    def no_video(source, destination):
        raise fetch.FetchError("no video format")

    def info(source):
        return SongMeta("youtube", source, "YouTube groove", "Drummer", 12, "video123")

    def run(path, meta, opts, log=print, progress=None):
        calls.append((path, meta, opts))
        outputs = ("drums.htdemucs.wav", "beats.json", "activations.drums-htdemucs.npy")
        for key, output in zip(STAGES[1:4], outputs):
            cached = (path / output).exists()
            if progress:
                progress(key, "cached" if cached else "running", None)
            if not cached:
                (path / output).write_bytes(b"fake model output")
                if progress:
                    progress(key, "done", .01)
        stamp_current(path)
        if progress:
            progress("score", "running", None)
        score = write_score(path, opts)
        if progress:
            progress("score", "done", .02)
        return pipeline.Result(score)

    monkeypatch.setattr(fetch, "decode_file", decode)
    monkeypatch.setattr(fetch, "youtube_info", info)
    monkeypatch.setattr(fetch, "download_audio", lambda source, destination: write_wav(destination))
    monkeypatch.setattr(fetch, "download_video", no_video)
    monkeypatch.setattr(fetch, "search_youtube", lambda q, limit: [info("https://youtube.com/watch?v=video123")])
    monkeypatch.setattr(pipeline, "run", run)
    return calls


def finished(client, job):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = client.get(f"/api/jobs/{job['id']}").json()
        if current["status"] in ("done", "error"):
            return current
        time.sleep(.01)
    pytest.fail("job did not finish")


def test_list_and_get_song(tmp_path):
    old = make_song(tmp_path, "old")
    new = make_song(tmp_path, "new")
    os.utime(old / "score.json", (100, 100))
    os.utime(new / "score.json", (200, 200))
    with TestClient(create_app(tmp_path)) as client:
        songs = client.get("/api/songs").json()
        assert [s["slug"] for s in songs] == ["new", "old"]
        assert songs[0] == {
            "slug": "new", "title": "Test groove", "uploader": "", "duration": 2,
            "kind": "file", "video_id": "", "transcribed": True, "outdated": True,
            "bars": 1, "tempo": 120, "updated_at": 200,
            "size_bytes": songs[0]["size_bytes"], "cache_bytes": songs[0]["cache_bytes"],
        }
        assert 0 < songs[0]["cache_bytes"] < songs[0]["size_bytes"]
        song = client.get("/api/songs/new").json()
        assert song["options"] == library.DEFAULT_OPTIONS
        assert song["score"]["bars"][0]["beats"] == 4
        assert song["has_drums"] is True


def test_untranscribed_song_and_incomplete_directory(tmp_path):
    make_song(tmp_path, "pending", transcribed=False)
    broken = tmp_path / "broken"
    broken.mkdir()
    SongMeta("file", "/missing", "Incomplete").save(broken / "meta.json")
    with TestClient(create_app(tmp_path)) as client:
        assert len(client.get("/api/songs").json()) == 1
        song = client.get("/api/songs/pending").json()
        assert song["score"] is None and song["bars"] is None and song["tempo"] is None
        assert song["has_drums"] is False


@pytest.mark.parametrize("url", ["/api/songs/unknown", "/api/jobs/unknown", "/api/songs/unknown/audio"])
def test_unknown_returns_404(tmp_path, url):
    with TestClient(create_app(tmp_path)) as client:
        response = client.get(url)
        assert response.status_code == 404
        assert isinstance(response.json()["detail"], str)


@pytest.mark.parametrize("slug", ["..", ".", "../escape", "/absolute", "a/b", "a\\b", ""])
def test_slug_validation(tmp_path, slug):
    with pytest.raises(ValueError):
        library.song_path(tmp_path, slug)


def test_path_traversal_and_symlink_rejected(tmp_path):
    songs = tmp_path / "songs"
    songs.mkdir()
    outside = make_song(tmp_path, "outside")
    (songs / "escape").symlink_to(outside, target_is_directory=True)
    with TestClient(create_app(songs)) as client:
        for url in ("/api/songs/%2e%2e", "/api/songs/..%2Foutside", "/api/songs/escape"):
            assert client.get(url).status_code in (400, 404)
        assert client.get("/api/songs").json() == []


def test_requantize_merges_and_persists_options(tmp_path, fakes):
    path = make_song(tmp_path)
    library.save_options(path, library.DEFAULT_OPTIONS | {"triplets": "off", "meter": 3})
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/songs/groove/score", json={"grid": 8, "voices": 2, "meter": None})
        assert response.status_code == 200
        options = library.DEFAULT_OPTIONS | {"grid": 8, "voices": 2, "triplets": "off"}
        assert response.json()["options"] == options
        assert response.json()["score"]["grid"] == 8
        assert json.loads((path / "options.json").read_text()) == options | {"code_version": fingerprint.snapshot()["score"]}
        assert client.get("/api/songs/groove").json()["options"] == options
    assert fakes[0][2].pdf is False and fakes[0][2].preview is False


def test_requantize_requires_all_model_outputs(tmp_path, fakes):
    path = make_song(tmp_path)
    (path / "activations.drums-htdemucs.npy").unlink()
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/songs/groove/score", json={}).status_code == 409
    assert not fakes


@pytest.mark.parametrize("options", [
    {"grid": 32}, {"triplets": "on"}, {"voices": 3}, {"meter": 0}, {"meter": 13},
    {"meter": 4.5}, {"grid": None}, {"odd_bars": "maybe"}, {"tempo_scale": 3}, {"beat_shift": 12}, {"beat_shift": 0.125}, {"force": True},
    {"voices": True}, {"grid": 8.0},
])
def test_invalid_options_return_422(tmp_path, options, fakes):
    make_song(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        for response in (
            client.post("/api/songs/groove/score", json=options),
            client.post("/api/jobs", json={"source": "/tmp/audio.wav", "options": options}),
            client.post("/api/upload", files={"file": ("groove.wav", b"audio")},
                        data={"options": json.dumps(options)}),
        ):
            assert response.status_code == 422
            assert isinstance(response.json()["detail"], str)
    assert not fakes


def test_job_lifecycle_all_stages_in_order(tmp_path, monkeypatch, fakes):
    entered, release = threading.Event(), threading.Event()

    def download(source, destination):
        entered.set()
        assert release.wait(5)
        write_wav(destination)

    monkeypatch.setattr(fetch, "download_audio", download)
    with TestClient(create_app(tmp_path)) as client:
        job = client.post("/api/jobs", json={"source": "https://youtube.com/watch?v=video123",
                                             "options": {"grid": 8}}).json()
        try:
            assert job["status"] == "queued"
            assert [s["key"] for s in job["stages"]] == list(STAGES)
            assert all(s["status"] == "pending" for s in job["stages"])
            assert entered.wait(5)
            running = client.get(f"/api/jobs/{job['id']}").json()
            assert running["status"] == "running" and running["stages"][0]["status"] == "running"
            assert running["title"] == "YouTube groove"
        finally:
            release.set()
        done = finished(client, job)
        assert done["status"] == "done" and done["error"] is None
        assert done["title"] == "YouTube groove" and done["slug"] == "youtube-groove-video123"
        assert [s["key"] for s in done["stages"]] == list(STAGES)
        assert all(s["status"] == "done" and s["seconds"] >= 0 for s in done["stages"])
        assert client.get("/api/jobs").json() == [done]
        assert client.get(f"/api/songs/{done['slug']}").json()["options"]["grid"] == 8


def test_existing_song_job_reuses_saved_options_and_cached_stages(tmp_path, fakes):
    songs = tmp_path / "songs"
    path = make_song(songs)
    source = tmp_path / "groove.wav"
    source.write_bytes(b"upload")
    library.save_options(path, library.DEFAULT_OPTIONS | {"voices": 2})
    with TestClient(create_app(songs)) as client:
        job = client.post("/api/jobs", json={"source": str(source), "options": {"grid": 8}}).json()
        done = finished(client, job)
        assert done["status"] == "done"
        assert [s["status"] for s in done["stages"]] == ["done", "cached", "cached", "cached", "done"]
        assert client.get("/api/songs/groove").json()["options"] == library.DEFAULT_OPTIONS | {"voices": 2, "grid": 8}


@pytest.mark.parametrize("failure", ["fetch", "beats"])
def test_job_error_preserves_running_stage(tmp_path, monkeypatch, fakes, failure):
    if failure == "fetch":
        def fail(*args):
            raise fetch.FetchError("download failed")
        monkeypatch.setattr(fetch, "download_audio", fail)
    else:
        def fail(path, meta, opts, log=print, progress=None):
            progress("separate", "cached", None)
            progress("beats", "running", None)
            raise RuntimeError("beat tracking failed")
        monkeypatch.setattr(pipeline, "run", fail)
    with TestClient(create_app(tmp_path)) as client:
        job = client.post("/api/jobs", json={"source": "https://youtube.com/watch?v=video123"}).json()
        result = finished(client, job)
        assert result["status"] == "error" and "failed" in result["error"]
        assert result["stages"][STAGES.index(failure)]["status"] == "running"
        assert result["stages"][-1]["status"] == "pending"
        assert not any(tmp_path.glob("*/options.json"))
        if failure == "fetch":
            assert not any(tmp_path.glob("*/meta.json"))


def test_upload_job_preserves_filename_and_cleans_temp(tmp_path, fakes):
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/upload", files={"file": ("My Drum Take.mp3", b"audio", "audio/mpeg")},
                               data={"options": '{"voices": 2}'})
        assert response.status_code == 200
        job = response.json()
        result = finished(client, job)
        assert result["status"] == "done"
        assert result["slug"] == "my-drum-take" and result["title"] == "My Drum Take"
        assert client.get("/api/songs/my-drum-take").json()["options"]["voices"] == 2
    assert not Path(job["source"]).exists()
    assert not Path(job["source"]).parent.exists()


def test_upload_cleanup_on_error(tmp_path, monkeypatch, fakes):
    def fail(*args):
        raise fetch.FetchError("bad audio")
    monkeypatch.setattr(fetch, "decode_file", fail)
    with TestClient(create_app(tmp_path)) as client:
        job = client.post("/api/upload", files={"file": ("take.wav", b"invalid")}).json()
        assert finished(client, job)["status"] == "error"
    assert not Path(job["source"]).parent.exists()


def test_jobs_are_fifo_and_keep_last_20_finished(tmp_path, fakes):
    sources = []
    for number in range(22):
        source = tmp_path / f"take-{number}.wav"
        source.write_bytes(b"audio")
        sources.append(str(source))
    with TestClient(create_app(tmp_path / "songs")) as client:
        jobs = [client.post("/api/jobs", json={"source": source}).json() for source in sources]
        assert finished(client, jobs[-1])["status"] == "done"
        deadline = time.monotonic() + 5
        while len(retained := client.get("/api/jobs").json()) != 20:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert [job["id"] for job in retained] == [job["id"] for job in reversed(jobs[2:])]
        assert client.get(f"/api/jobs/{jobs[0]['id']}").status_code == 404
    assert [call[1].source for call in fakes] == sources


def test_worker_and_requantize_do_not_overlap(tmp_path, monkeypatch, fakes):
    songs = tmp_path / "songs"
    make_song(songs)
    source = tmp_path / "groove.wav"
    source.write_bytes(b"audio")
    entered, release, second = threading.Event(), threading.Event(), threading.Event()
    fake_run = pipeline.run
    count = 0

    def gated_run(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            entered.set()
            assert release.wait(5)
        else:
            second.set()
        return fake_run(*args, **kwargs)

    monkeypatch.setattr(pipeline, "run", gated_run)
    with TestClient(create_app(songs)) as client, ThreadPoolExecutor() as pool:
        job = client.post("/api/jobs", json={"source": str(source)}).json()
        try:
            assert entered.wait(5)
            future = pool.submit(client.post, "/api/songs/groove/score", json={"grid": 8})
            assert not second.wait(.1)
        finally:
            release.set()
        assert future.result(timeout=5).status_code == 200
        assert finished(client, job)["status"] == "done"
        assert second.is_set()
        assert client.get("/api/songs/groove").json()["options"]["grid"] == 8


def test_audio_range_and_newest_drum_stem(tmp_path):
    path = make_song(tmp_path)
    newest = path / "drums.htdemucs_ft.wav"
    newest.write_bytes(b"newest drum stem")
    os.utime(path / "drums.htdemucs.wav", (100, 100))
    os.utime(newest, (200, 200))
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/api/songs/groove/audio", headers={"Range": "bytes=0-15"})
        assert response.status_code == 206
        assert response.content == (path / "audio.wav").read_bytes()[:16]
        assert response.headers["content-range"].startswith("bytes 0-15/")
        assert client.get("/api/songs/groove/audio?stem=drums").content == newest.read_bytes()
        newest.unlink()
        (path / "drums.htdemucs.wav").unlink()
        assert client.get("/api/songs/groove/audio?stem=drums").status_code == 404


def test_musicxml_and_attachment_downloads(tmp_path):
    path = make_song(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/songs/groove/musicxml").text == "<score-partwise/>"
        for kind, suffix in (("musicxml", "musicxml"), ("midi", "mid")):
            response = client.get(f"/api/songs/groove/download/{kind}")
            assert response.content == (path / f"groove.{suffix}").read_bytes()
            assert f'attachment; filename="groove.{suffix}"' == response.headers["content-disposition"]


def test_pdf_on_demand_refresh_and_unavailable(tmp_path, monkeypatch):
    path = make_song(tmp_path)
    calls = []

    def engrave(xml, pdf):
        calls.append(xml)
        pdf.write_bytes(b"%PDF-fake")
        return True

    monkeypatch.setattr(render, "render_pdf", engrave)
    with TestClient(create_app(tmp_path)) as client:
        url = "/api/songs/groove/download/pdf"
        assert client.get(url).content == b"%PDF-fake"
        assert client.get(url).status_code == 200 and len(calls) == 1
        os.utime(path / "groove.pdf", (1, 1))
        assert client.get(url).status_code == 200 and len(calls) == 2
        (path / "groove.pdf").unlink()
        monkeypatch.setattr(render, "render_pdf", lambda *args: False)
        response = client.get(url)
        assert response.status_code == 503 and "MuseScore" in response.json()["detail"]


def test_search_empty_success_and_failure(tmp_path, monkeypatch, fakes):
    with TestClient(create_app(tmp_path)) as client:
        for params in ({}, {"q": ""}, {"q": "  "}):
            assert client.get("/api/search", params=params).status_code == 400
        calls = []

        def search(q, limit):
            calls.append((q, limit))
            return [SongMeta("youtube", "https://youtube.com/watch?v=id", "Groove", "Drummer", 12, "id")]

        monkeypatch.setattr(fetch, "search_youtube", search)
        assert client.get("/api/search?q=drums").json() == [
            {"video_id": "id", "title": "Groove", "uploader": "Drummer", "duration": 12,
             "url": "https://youtube.com/watch?v=id"}
        ]
        assert calls == [("drums", 8)]

        def fail(*args):
            raise fetch.FetchError("yt-dlp failed")

        monkeypatch.setattr(fetch, "search_youtube", fail)
        assert client.get("/api/search?q=drums").status_code == 502


def test_spa_fallback_never_shadows_api(tmp_path):
    static = tmp_path / "dist"
    static.mkdir()
    (static / "index.html").write_text("<html>SPA</html>")
    (static / "app.js").write_text("console.log('app')")
    with TestClient(create_app(tmp_path / "songs", static)) as client:
        for url in ("/", "/library/groove"):
            assert client.get(url).text == "<html>SPA</html>"
        assert client.get("/app.js").text == "console.log('app')"
        assert client.get("/api/songs").json() == []
        for url in ("/api", "/api/no-such-route", "/api/songs/missing/audio"):
            response = client.get(url)
            assert response.status_code == 404 and "detail" in response.json()


def test_dev_cors(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        response = client.options("/api/jobs", headers={
            "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_prepare_song_does_not_save_meta_without_audio(tmp_path, monkeypatch):
    source = tmp_path / "original.wav"
    source.write_bytes(b"audio")
    monkeypatch.setattr(fetch, "decode_file", lambda *args: SongMeta("file", str(source), "Original"))
    with pytest.raises(fetch.FetchError, match="not produced"):
        library.prepare_song(str(source), tmp_path / "songs")
    assert not list((tmp_path / "songs").glob("*/meta.json"))


def test_pipeline_progress_real_run_and_unchanged_logs(tmp_path, monkeypatch):
    path = make_song(tmp_path)
    meta = SongMeta.load(path / "meta.json")
    monkeypatch.setattr(pipeline.models, "pick_device", lambda requested: "cpu")
    monkeypatch.setattr(pipeline.models, "pick_hits", lambda *args, **kw: [])
    monkeypatch.setattr(pipeline.models, "separate_drums", lambda audio, drums, **kw: write_wav(drums))
    monkeypatch.setattr(pipeline.models, "track_beats", lambda *args, **kw: ([0, .5, 1, 1.5, 2], [0, 2]))
    monkeypatch.setattr(pipeline.models, "drum_activations", lambda *args, **kw: np.zeros((2, 5)))
    events, logs = [], []
    pipeline.run(path, meta, pipeline.Options(pdf=False), log=logs.append,
                 progress=lambda *event: events.append(event))
    assert [(e[0], e[1]) for e in events] == [
        ("separate", "cached"), ("beats", "cached"), ("onsets", "cached"),
        ("score", "running"), ("score", "done"),
    ]
    assert logs == ["drum separation (htdemucs, cpu): cached", "beat tracking: cached",
                    "drum transcription (on drums): cached"]
    events.clear()
    pipeline.run(path, meta, pipeline.Options(pdf=False, force=True), log=lambda line: None,
                 progress=lambda *event: events.append(event))
    assert [(e[0], e[1]) for e in events] == [(key, status) for key in STAGES[1:] for status in ("running", "done")]
    assert all(e[2] >= 0 for e in events if e[1] == "done")


def test_clear_cache_keeps_audio_and_score(tmp_path, fakes):
    path = make_song(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        song = client.delete("/api/songs/groove/cache").json()
        assert song["cache_bytes"] == 0 and song["has_drums"] is False and song["transcribed"] is True
        assert (path / "audio.wav").is_file() and (path / "score.json").is_file()
        assert not list(path.glob("drums.*.wav")) and not (path / "beats.json").exists()
        assert client.post("/api/songs/groove/score", json={}).status_code == 409


def test_delete_song_removes_directory(tmp_path):
    path = make_song(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        assert client.delete("/api/songs/groove").status_code == 204
        assert not path.exists()
        assert client.get("/api/songs").json() == []
        assert client.delete("/api/songs/groove").status_code == 404
        assert client.delete("/api/songs/..%2Fescape").status_code in (400, 404)


def test_video_is_served_with_range_and_reported(tmp_path):
    path = make_song(tmp_path)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/songs/groove").json()["has_video"] is False
        assert client.get("/api/songs/groove/video").status_code == 404
        (path / "video.mp4").write_bytes(bytes(range(100)))
        assert client.get("/api/songs/groove").json()["has_video"] is True
        part = client.get("/api/songs/groove/video", headers={"Range": "bytes=10-19"})
        assert part.status_code == 206 and part.content == bytes(range(10, 20))


def test_youtube_song_keeps_video_and_takes_audio_from_it(tmp_path, monkeypatch, fakes):
    decoded = []
    monkeypatch.setattr(fetch, "download_video", lambda source, destination: destination.write_bytes(b"mp4"))
    monkeypatch.setattr(fetch, "decode_file", lambda source, destination: (decoded.append(source.name), write_wav(destination)))
    monkeypatch.setattr(fetch, "download_audio", lambda *args: pytest.fail("audio must come from the video"))
    song_dir, meta = library.prepare_song("https://youtube.com/watch?v=video123", tmp_path)
    assert (song_dir / "video.mp4").read_bytes() == b"mp4" and (song_dir / "audio.wav").is_file()
    assert decoded == ["video.mp4"]
    assert not list(song_dir.glob(".part.*"))


def test_same_name_different_recording_gets_its_own_directory(tmp_path, fakes):
    first, second = tmp_path / "a" / "Song.wav", tmp_path / "b" / "Song.wav"
    for source, content in ((first, b"one"), (second, b"two")):
        source.parent.mkdir()
        source.write_bytes(content)
    out = tmp_path / "songs"
    dir_one, _ = library.prepare_song(str(first), out)
    dir_two, _ = library.prepare_song(str(second), out)
    assert dir_one != dir_two
    assert library.prepare_song(str(first), out)[0] == dir_one
    assert library.prepare_song(str(second), out)[0] == dir_two


def test_failed_decode_can_be_retried(tmp_path, monkeypatch, fakes):
    source = tmp_path / "Song.wav"
    source.write_bytes(b"audio")

    def broken(src, destination):
        destination.write_bytes(b"half")
        raise fetch.FetchError("ffmpeg failed")

    working = fetch.decode_file
    monkeypatch.setattr(fetch, "decode_file", broken)
    with pytest.raises(fetch.FetchError):
        library.prepare_song(str(source), tmp_path / "songs")
    assert not list((tmp_path / "songs").rglob("*.wav"))
    monkeypatch.setattr(fetch, "decode_file", working)
    song_dir, _ = library.prepare_song(str(source), tmp_path / "songs")
    assert (song_dir / "audio.wav").is_file() and (song_dir / "meta.json").is_file()


def test_failed_stage_leaves_no_cache(tmp_path, monkeypatch):
    real_run = pipeline.run  # this test needs the real pipeline, not the `fakes` fixture
    song = tmp_path / "song"
    song.mkdir()
    write_wav(song / "audio.wav")

    def broken(audio, target, **kwargs):
        target.write_bytes(b"half a stem")
        raise RuntimeError("separation crashed")

    monkeypatch.setattr(pipeline.models, "pick_device", lambda requested: "cpu")
    monkeypatch.setattr(pipeline.models, "separate_drums", broken)
    with pytest.raises(RuntimeError):
        real_run(song, SongMeta("file", "x", "Song"), pipeline.Options(), log=lambda line: None)
    assert [p.name for p in song.iterdir()] == ["audio.wav"]


def test_transcribe_existing_song_skips_fetch(tmp_path, monkeypatch, fakes):
    make_song(tmp_path, transcribed=False)
    monkeypatch.setattr(library, "prepare_song", lambda *args, **kwargs: pytest.fail("must not fetch again"))
    with TestClient(create_app(tmp_path)) as client:
        job = finished(client, client.post("/api/songs/groove/transcribe", json={"grid": 8}).json())
        assert job["status"] == "done" and job["slug"] == "groove"
        assert [stage["status"] for stage in job["stages"]][0] == "cached"
        song = client.get("/api/songs/groove").json()
        assert song["transcribed"] is True and song["options"]["grid"] == 8
        assert client.post("/api/songs/missing/transcribe").status_code == 404


def test_scores_made_by_other_code_are_outdated(tmp_path, fakes):
    stale = make_song(tmp_path, "stale")
    current = make_song(tmp_path, "current")
    make_song(tmp_path, "untranscribed", transcribed=False)
    library.save_options(current, library.DEFAULT_OPTIONS)
    options = json.loads((stale / "options.json").read_text()) if (stale / "options.json").exists() else {}
    (stale / "options.json").write_text(json.dumps(options | {"code_version": "older"}))
    with TestClient(create_app(tmp_path)) as client:
        outdated = {song["slug"]: song["outdated"] for song in client.get("/api/songs").json()}
        assert outdated == {"stale": True, "current": False, "untranscribed": False}
        assert client.post("/api/songs/stale/score", json={}).json()["outdated"] is False


def test_retranscribe_queues_outdated_songs_once(tmp_path, monkeypatch, fakes):
    make_song(tmp_path, "stale")
    library.save_options(make_song(tmp_path, "current"), library.DEFAULT_OPTIONS)
    make_song(tmp_path, "untranscribed", transcribed=False)
    with TestClient(create_app(tmp_path)) as client:
        jobs = client.post("/api/songs/retranscribe").json()
        assert [(job["slug"], job["title"]) for job in jobs] == [("stale", "Test groove")]
        assert finished(client, jobs[0])["status"] == "done"
        assert not any(song["outdated"] for song in client.get("/api/songs").json())

        release = threading.Event()
        fake = pipeline.run
        monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: release.wait(5) and fake(*args, **kwargs))
        everything = client.post("/api/songs/retranscribe", json={"scope": "all"}).json()
        assert sorted(job["slug"] for job in everything) == ["current", "stale", "untranscribed"]
        assert client.post("/api/songs/retranscribe", json={"scope": "all"}).json() == []  # already queued
        release.set()
        assert all(finished(client, job)["status"] == "done" for job in everything)
        assert client.post("/api/songs/retranscribe", json={"scope": "some"}).status_code == 422


def test_stale_caches_are_remade_by_jobs_but_not_by_rescoring(tmp_path, monkeypatch):
    path = make_song(tmp_path)
    meta = SongMeta.load(path / "meta.json")
    library.save_options(path, library.DEFAULT_OPTIONS)
    library.stamp_cache(path, "beats.json", "made-by-older-code")
    assert library.stale_caches(path) == ["beats.json"] and library.summarize(path)["outdated"]
    monkeypatch.setattr(pipeline.models, "pick_device", lambda requested: "cpu")
    monkeypatch.setattr(pipeline.models, "pick_hits", lambda *args, **kw: [])
    monkeypatch.setattr(pipeline.models, "track_beats", lambda *args, **kw: ([0, .5, 1, 1.5, 2], [0, 2]))
    events = []

    def run(**options):
        events.clear()
        pipeline.run(path, meta, pipeline.Options(pdf=False, **options), log=lambda line: None,
                     progress=lambda key, status, seconds=None: events.append((key, status)))
        library.save_options(path, library.DEFAULT_OPTIONS)
        return [status for key, status in events if key != "score"]

    assert run() == ["cached", "cached", "cached"]  # re-scoring never starts a model
    assert library.summarize(path)["outdated"]  # and so cannot hide the stale cache
    assert run(refresh_stale=True) == ["cached", "running", "done", "cached"]
    assert library.stale_caches(path) == [] and not library.summarize(path)["outdated"]
    assert run(refresh_stale=True) == ["cached", "cached", "cached"]


def test_songs_without_cache_stamps_are_outdated(tmp_path):
    path = make_song(tmp_path)
    library.save_options(path, library.DEFAULT_OPTIONS)
    assert not library.summarize(path)["outdated"]
    (path / "cache.json").unlink()  # caches from before stamping: their code is unknown
    assert library.summarize(path)["outdated"]
    library.clear_cache(path)
    assert not library.summarize(path)["outdated"]


def test_concurrent_retranscribe_requests_queue_each_song_once(tmp_path, monkeypatch, fakes):
    for slug in ("a", "b", "c"):
        make_song(tmp_path, slug)
    release = threading.Event()
    fake = pipeline.run
    monkeypatch.setattr(pipeline, "run", lambda *args, **kwargs: release.wait(5) and fake(*args, **kwargs))
    with TestClient(create_app(tmp_path)) as client:
        with ThreadPoolExecutor(4) as pool:
            batches = list(pool.map(lambda _: client.post("/api/songs/retranscribe", json={"scope": "all"}).json(), range(4)))
        queued = sorted(job["slug"] for batch in batches for job in batch)
        release.set()
        assert queued == ["a", "b", "c"]
        assert all(finished(client, job)["status"] == "done" for batch in batches for job in batch)


def test_retranscribe_skips_songs_that_vanish_mid_batch(tmp_path, monkeypatch, fakes):
    make_song(tmp_path, "kept")
    listed = library.list_songs(tmp_path)
    monkeypatch.setattr(library, "list_songs", lambda out: [listed[0] | {"slug": "gone"}, *listed])
    with TestClient(create_app(tmp_path)) as client:
        jobs = client.post("/api/songs/retranscribe", json={"scope": "all"}).json()
        assert [job["slug"] for job in jobs] == ["kept"]
        assert finished(client, jobs[0])["status"] == "done"


def sound_map():
    from groovemap import kit

    example = kit.Example(1.5, 0.9, 1.9)
    return kit.SoundMap([
        kit.SoundGroup("kick-1", "kick", 12, 3, [1.0, 0.0], [example], {"kick": 1.0, "snare": 0.7}),
        kit.SoundGroup("extra-1", None, 4, 0, [0.0, 1.0], [], {"kick": 0.4, "snare": 0.9}),
    ], {"drums.htdemucs.wav": "abc"})


def test_sound_groups_are_learned_once_and_named_through_the_score(tmp_path, monkeypatch, fakes):
    from groovemap import kit

    path = make_song(tmp_path)
    library.save_options(path, library.DEFAULT_OPTIONS | {"sound_names": {"kick-1": "snare", "gone-9": "tom"}})
    learned = []
    monkeypatch.setattr(pipeline, "detect_hits", lambda song_dir, opts: ([], np.zeros((2, 5)), np.zeros(0)))
    monkeypatch.setattr(kit, "load_or_learn", lambda song_dir, *args: (learned.append(song_dir), sound_map().save(song_dir), sound_map())[2])
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/api/songs/groove/sounds", json={"names": {}}).status_code == 409  # nothing learned yet
        sounds = client.get("/api/songs/groove/sounds").json()
        assert learned == [path] and (path / "soundmap.json").is_file()
        assert sounds == {"has_video": False, "groups": [
            {"id": "kick-1", "detected": "kick", "count": 12, "mixed": 3, "examples": [{"time": 1.5, "start": 0.9, "end": 1.9}],
             "similar": {"kick": 1.0, "snare": 0.7}, "name": "snare"},
            {"id": "extra-1", "detected": None, "count": 4, "mixed": 0, "examples": [], "similar": {"kick": 0.4, "snare": 0.9}, "name": None},
        ]}
        (path / "video.mp4").write_bytes(b"mp4")
        assert client.get("/api/songs/groove/sounds").json()["has_video"] is True

        assert client.post("/api/songs/groove/sounds", json={"names": {"kick-1": "cowbell"}}).status_code == 422
        assert client.post("/api/songs/groove/sounds", json={"names": {"nobody": "snare"}}).status_code == 422
        assert client.post("/api/songs/groove/sounds", json={"names": {"kick-1": "snare"}, "extra": 1}).status_code == 422
        song = client.post("/api/songs/groove/sounds", json={"names": {"extra-1": "hihat_pedal"}}).json()
        assert song["options"]["sound_names"] == {"extra-1": "hihat_pedal"}  # replaced, not merged
        assert fakes[-1][2].sound_names == {"extra-1": "hihat_pedal"}
        assert json.loads((path / "options.json").read_text())["sound_names"] == {"extra-1": "hihat_pedal"}
        assert client.get("/api/songs/groove/sounds").json()["groups"][1]["name"] == "hihat_pedal"
        cleared = client.post("/api/songs/groove/sounds", json={"names": {}}).json()
        assert cleared["options"]["sound_names"] == {}
        assert client.post("/api/songs/groove/score", json={"sound_names": {"kick-1": "ignore"}}).json()["options"]["sound_names"] == {"kick-1": "ignore"}
        (path / "attacks.htdemucs.npy").unlink()
        assert client.get("/api/songs/groove/sounds").status_code == 409
        assert client.post("/api/songs/groove/score", json={"sound_names": {"kick-1": "ignore"}}).status_code == 409
        assert client.post("/api/songs/groove/score", json={"sound_names": {}}).status_code == 200
