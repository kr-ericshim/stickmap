import gzip
import json

import numpy as np
import pytest

from stickmap import web_hints
from stickmap.fetch import SongMeta, _meta_from_info
from stickmap.meter_suggestion import suggest_signature
from stickmap.score import Bar, Score


HINT = {"songId": 6700, "artist": "Tool", "title": "Schism", "url": "https://www.songsterr.com/a/wsa/tool-schism-tab-s6700",
        "signature": "4/4", "bpm": 110, "bars": 10}


@pytest.fixture
def fake_api(monkeypatch):
    monkeypatch.setenv("STICKMAP_WEB_HINTS", "1")
    # Even a broken mock must not make a real request.
    monkeypatch.setattr(web_hints, "urlopen", lambda *a, **kw: pytest.fail("unexpected network access"))

    def install(results, track=None, meta=None):
        responses = iter([results, meta or {"revisionId": 5, "image": "image", "popularTrackDrum": 0,
                                          "tracks": [{"instrumentId": 1024}]}, track])
        calls = []

        def get(url):
            calls.append(url)
            return next(responses)

        monkeypatch.setattr(web_hints, "_get_json", get)
        return calls

    return install


def test_strict_lookup_rejects_fuzzy_puzzle_pieces(fake_api):
    calls = fake_api([{"songId": 1, "artist": "DOMi & JD BECK", "title": "NOT TiGHT", "tracks": [{"instrumentId": 1024}]}])
    assert web_hints.lookup("domi & jd beck", "Puzzle Pieces") is None
    assert len(calls) == 1


@pytest.mark.parametrize("artist,title,expected", [
    ("The Tool", "Schism (Album Version)", True),
    ("Tool", "Schism feat. Someone", True),
    ("Different Artist", "Schism", False),
    ("Tool", "Schism Live", False),
    ("Tool", "Schism (Drum Cover)", False),
    ("Tool", "Schism (Transcription)", False),
    ("Tool", "Schism (Backing Track)", False),
    ("Tool", "Schism (Lesson)", False),
    ("Tool", "Schism Part Two", False),
])
def test_match(artist, title, expected):
    assert web_hints._matches({"artist": artist, "title": title}, "Tool", "Schism") is expected


def test_explicit_cover_is_allowed():
    assert web_hints._matches({"artist": "Tool", "title": "Schism (Cover)"}, "Tool", "Schism Cover")


def test_lookup_uses_drum_track_and_summarizes(fake_api):
    calls = fake_api([{"songId": 6700, "artist": "Tool", "title": "Schism", "tracks": [{"instrumentId": 1024}]}],
                     {"measures": [{"signature": s} for s in [[5, 8], [7, 8]] * 5],
                      "automations": {"tempo": [{"measure": 0, "position": [0, 1], "bpm": 110}]}},
                     {"revisionId": 5, "image": "image", "tracks": [{"instrumentId": 25}, {"instrumentId": 1024}]})
    hint = web_hints.lookup("Tool", "Schism")
    assert hint["signature"] == "5/8+7/8"
    assert hint["bpm"] == 110 and hint["bars"] == 10
    assert calls[-1].endswith("/6700/5/image/1.json")


@pytest.mark.parametrize("measures,expected", [
    ([{"signature": [4, 4]}] + [{}] * 8 + [{"signature": [7, 8]}], "4/4"),
    ([{"signature": s} for s in [[5, 8], [7, 8]] * 5], "5/8+7/8"),
    ([{"signature": [5, 8]}] + [{}] * 4 + [{"signature": [7, 8]}] + [{}] * 4, None),
    ([{"signature": s} for s in [[3, 4], [7, 8]] * 5], None),
    ([], None),
])
def test_signature_summary(measures, expected):
    assert web_hints._signature_summary(measures) == expected


def test_tempo_is_weighted_by_bars_not_events():
    track = {"automations": {"tempo": [{"measure": 0, "bpm": 110}, {"measure": 8, "bpm": 220},
                                        {"measure": 9, "position": 0.5, "bpm": 200}]}}
    assert web_hints._tempo_summary(track, 10) == 110
    assert web_hints._tempo_summary({}, 10) is None


@pytest.mark.parametrize("signature,web_bpm,current,scale", [
    ("4/4", 110, 55, 2), (None, 110, 220, .5), ("3/4+5/4", 110, 55, 2),
    ("4/4", 117, 55, 2), ("4/4", 120, 55, None), ("4/4", 110, 110, None),
    ("5/8+7/8", 110, 55, None), ("6/8", 110, 55, None), (None, None, 55, None),
    (None, 110, 0, None),
])
def test_tempo_suggestion(signature, web_bpm, current, scale):
    hint = {**HINT, "signature": signature, "bpm": web_bpm}
    result = web_hints.suggest_tempo(hint, current)
    if scale is None:
        assert result is None
    else:
        assert result == {"tempo_scale": scale, "web_bpm": web_bpm, "current_bpm": current, "source": web_hints.source(hint)}


@pytest.mark.parametrize("signature", ["7/8", "5/8+7/8"])
def test_web_signature_without_audio_is_reviewable_only(signature):
    score = Score("Schism", [Bar(0, 4, [0, .5, 1, 1.5, 2], [4]*4, 120)])
    before = score.to_json()
    result = suggest_signature(score, np.zeros((1, 5)), [], [], [], web_hint=signature, source=web_hints.source(HINT))
    assert result["time_signature"] == signature
    assert result["reasons"] == ["web_hint"]
    assert result["audio_agrees"] is False
    assert result["source"]["url"] == HINT["url"]
    assert score.to_json() == before
    assert suggest_signature(score, np.zeros((1, 5)), [], [], [], web_hint="4/4") is None


def test_web_signature_audio_agreement():
    from test_time_signature import synthetic

    a, t, _ = synthetic(7, 8, .15)
    score = Score("Song", [Bar(i, 4, [i*2.4+j*.6 for j in range(5)], [4]*4, 100) for i in range(8)])
    result = suggest_signature(score, a, t[:-1], np.arange(0, t[-1], .6), np.arange(0, t[-1], 1.2), web_hint="7/8")
    assert result["audio_agrees"] is True
    assert result["reasons"][0] == "web_hint"


def test_song_query_and_metadata():
    meta = _meta_from_info({"id": "1", "title": "wrong title", "creator": "Tool", "track": "Schism"})
    assert web_hints.song_query(meta, "overridden") == ("Tool", "Schism")
    assert web_hints.song_query(SongMeta("file", "x", "x"), "Tool - Schism (Official Video) [MV] 4K") == ("Tool", "Schism")
    assert web_hints.song_query(SongMeta("file", "x", "x"), "Tool - Schism (Drum Cover)") == ("Tool", "Schism")
    assert web_hints.song_query(SongMeta("file", "x", "x"), "unknown") is None


def test_cache_positive_negative_and_expiry(tmp_path, monkeypatch):
    monkeypatch.setenv("STICKMAP_WEB_HINTS", "1")
    calls = []
    result = None
    monkeypatch.setattr(web_hints, "_lookup", lambda *query: (calls.append(query), result)[1])
    now = 1_000_000
    monkeypatch.setattr(web_hints.time, "time", lambda: now)
    meta = SongMeta("file", "x", "Tool - Schism")
    assert web_hints.load_or_fetch(tmp_path, meta, meta.title) is None
    assert web_hints.load_or_fetch(tmp_path, meta, meta.title) is None
    assert len(calls) == 1
    now += 7 * 86400 + 1
    result = HINT
    assert web_hints.load_or_fetch(tmp_path, meta, meta.title) == HINT
    now += 8 * 86400
    assert web_hints.load_or_fetch(tmp_path, meta, meta.title) == HINT
    assert len(calls) == 2
    assert web_hints.load_or_fetch(tmp_path, meta, "Tool - Other") == HINT
    assert len(calls) == 3


def test_disabled_never_fetches(tmp_path, monkeypatch):
    monkeypatch.setattr(web_hints, "_get_json", lambda *a: pytest.fail("network disabled"))
    assert web_hints.lookup("Tool", "Schism") is None
    assert web_hints.load_or_fetch(tmp_path, SongMeta("file", "x", "Tool - Schism"), "Tool - Schism") is None
    assert not (tmp_path / "web_hints.json").exists()


def test_network_errors_are_nonfatal_and_not_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("STICKMAP_WEB_HINTS", "1")

    def fail(*args):
        raise OSError("offline")

    monkeypatch.setattr(web_hints, "_get_json", fail)
    assert web_hints.load_or_fetch(tmp_path, SongMeta("file", "x", "Tool - Schism"), "Tool - Schism") is None
    assert not (tmp_path / "web_hints.json").exists()  # retried on the next run


def test_gzip_and_control_char_json(monkeypatch):
    from io import BytesIO

    def response(request, timeout):
        assert timeout == 8
        assert request.get_header("User-agent")
        return BytesIO(gzip.compress(b'{"description":"hello\x01world"}'))

    monkeypatch.setattr(web_hints, "urlopen", response)
    assert web_hints._get_json("https://example.test") == {"description": "hello\x01world"}


def test_score_optional_hints_roundtrip():
    score = Score("Song", [])
    assert "web_hint" not in score.to_json() and "tempo_suggestion" not in score.to_json()
    score.web_hint = {"source": "Songsterr", **HINT}
    score.tempo_suggestion = web_hints.suggest_tempo(HINT, 55)
    assert Score.from_json(score.to_json()) == score


def test_title_punctuation_is_ignored():
    assert web_hints._matches({"artist": "Artist", "title": "Don't Stop"}, "artist", "Dont Stop")


@pytest.mark.parametrize("options,expect_meter,expect_tempo", [
    ({}, True, True), ({"meter": 4}, False, True),
    ({"time_signature": "4/4"}, False, True), ({"web_hints": False}, False, False),
])
def test_pipeline_hints_do_not_change_score(tmp_path, monkeypatch, options, expect_meter, expect_tempo):
    from stickmap import pipeline
    from stickmap import signature_quantize

    np.save(tmp_path / "activations.mix.npy", np.zeros((1, 5)))
    (tmp_path / "beats.json").write_text(json.dumps({"beats": [0, .5, 1, 1.5, 2], "downbeats": [0]}))
    monkeypatch.setattr(pipeline.models, "pick_device", lambda _: "cpu")
    monkeypatch.setattr(pipeline.models, "pick_hits", lambda *args: [])
    original = Score("Song", [Bar(0, 4, [0, .5, 1, 1.5, 2], [4]*4, 55)], tempo=55)
    original_json = original.to_json()
    monkeypatch.setattr(pipeline, "quantize", lambda *args, **kw: Score.from_json(original_json))
    monkeypatch.setattr(signature_quantize, "quantize_signature", lambda *args, **kw: Score.from_json(original_json))
    calls = []
    hint = {**HINT, "signature": "3/4"}
    monkeypatch.setattr(web_hints, "load_or_fetch", lambda *args: (calls.append(args), hint)[1])
    result = pipeline.run(tmp_path, SongMeta("file", "x", "Tool - Schism"),
                          pipeline.Options(onset_input="mix", pdf=False, ride_runs=False, **options), log=lambda _: None)
    score = result.score
    assert bool(score.time_signature_suggestion) is expect_meter
    assert bool(score.tempo_suggestion) is expect_tempo
    assert len(calls) == (0 if options.get("web_hints") is False else 1)
    assert score.bars == original.bars and score.tempo == original.tempo
    assert bool(score.web_hint) is (options.get("web_hints") is not False)
    assert Score.load(result.files["json"]) == score
