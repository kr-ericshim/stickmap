"""Explicit metre corrections must repair timing, not just print a new fraction."""

import json
import xml.etree.ElementTree as ET

import mido
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.ndimage import gaussian_filter1d

from groovemap import library, pipeline
from groovemap.metre import Signature
from groovemap.meter_suggestion import suggest_signature
from groovemap.midi import to_midi
from groovemap.musicxml import to_musicxml
from groovemap.pulse import build_pulse
from groovemap.score import Bar, Hit, Note, Score
from groovemap.server import OptionsPatch, create_app
from groovemap.signature_quantize import quantize_signature


def synthetic(n, denominator, period, drift=0.0, bars=20):
    intervals = np.linspace(period, period * (1 + drift), n * bars)
    times = np.r_[0.1, 0.1 + np.cumsum(intervals)]
    activations = np.zeros((int((times[-1] + 0.05) * 100), 5))
    hits = []
    for i, time in enumerate(times[:-1]):
        activations[round(time * 100), 3] = 0.6
        hits.append(Hit(time, "hihat", 0.6))
        if i % n == 0:
            activations[round(time * 100), 0] = 1
            hits.append(Hit(time, "kick", 1))
        if i % n == n // 2:
            activations[round(time * 100), 1] = 0.8
            hits.append(Hit(time, "snare", 0.8))
    return gaussian_filter1d(activations, 0.7, axis=0), times, hits


@pytest.mark.parametrize("value", ["1/4", "3/4", "5/4", "6/8", "7/8", "12/8", "7/16", "15/16"])
def test_options_accept_signature(value):
    assert OptionsPatch(time_signature=value).time_signature == value
    assert pipeline.Options(time_signature=value).time_signature == value


@pytest.mark.parametrize("value", ["0/8", "16/4", "7/3", "7/32", "07/8", "7 / 8", "automatic", "7/8x", 7, {"numerator": 7}])
def test_options_reject_invalid_signature(value):
    with pytest.raises(ValueError):
        OptionsPatch(time_signature=value)
    with pytest.raises(ValueError):
        pipeline.Options(time_signature=value)


def test_old_options_keep_legacy_meter_and_roundtrip_new_option(tmp_path):
    (tmp_path / "options.json").write_text(json.dumps({"meter": 7, "follow_downbeats": True}))
    loaded = library.load_options(tmp_path)
    assert loaded["meter"] == 7 and loaded["time_signature"] is None
    assert loaded["odd_bars"] == "detected"
    loaded["time_signature"] = "7/8"
    library.save_options(tmp_path, loaded)
    assert library.load_options(tmp_path) == loaded
    assert OptionsPatch.model_validate(loaded).time_signature == "7/8"


@pytest.mark.parametrize("n,d,period", [(7, 8, .15), (5, 4, .42), (6, 8, .2), (3, 4, .5), (12, 8, .18), (7, 16, .09)])
def test_wrong_tracker_and_live_drift(n, d, period):
    activations, expected, hits = synthetic(n, d, period, drift=.2)
    tracker = np.arange(.03, expected[-1], .613)
    pulse = build_pulse(activations, expected[:-1], tracker, tracker[::2], Signature(n, d))
    assert pulse.source == "rebuilt"
    actual = np.array(pulse.times)
    error = np.min(np.abs(actual[:, None] - expected[None, :]), axis=0)
    assert np.quantile(error, .95) < period * .08
    assert np.median(np.diff(actual)[-n*3:]) > 1.12 * np.median(np.diff(actual)[:n*3])
    score = quantize_signature(hits, tracker, tracker[::2], activations=activations,
                               attacks=expected[:-1], time_signature=f"{n}/{d}")
    assert 19 <= len(score.bars) <= 22
    assert all(bar.beats == n and bar.denominator == d for bar in score.bars)
    # A drift-following grid keeps the repeated kick on one tick.
    ticks = [note.tick for bar in score.bars[1:-1] for note in bar.notes if note.inst == "kick"]
    assert ticks.count(0) / len(ticks) >= .95


@pytest.mark.parametrize("stride", [1, 2, 3, 4])
def test_compatible_tracker_units_are_resampled_without_recount(stride):
    a, t, _ = synthetic(7, 8, .15)
    pulse = build_pulse(a, t[:-1], t[::stride], t[::7], Signature(7, 8))
    assert pulse.tracker_compatible
    assert pulse.source == "tracker"
    assert pulse.period == pytest.approx(.15, abs=.002)


def test_scale_and_half_denominator_shift_have_defined_units():
    a, t, _ = synthetic(7, 8, .15)
    args = (a, t[:-1], t, t[::7], Signature(7, 8))
    base = build_pulse(*args)
    shifted = build_pulse(*args, beat_shift=.5)
    assert (shifted.times[0] - base.times[0]) % (7 * .15) == pytest.approx(.075)
    assert build_pulse(*args, tempo_scale=2).period == pytest.approx(.075)
    assert build_pulse(*args, tempo_scale=.5).period == pytest.approx(.3)


@pytest.mark.parametrize("n,d,groups", [(7, 8, [2, 2, 3]), (6, 8, [3, 3]), (12, 8, [3]*4), (7, 16, [2, 2, 3])])
def test_xml_signature_beaming_and_duration(n, d, groups):
    unit = 96 // d
    bar = Bar(0, n, [i * .15 for i in range(n+1)], [1]*n, 200,
              [Note(i * unit, "hihat", 90) for i in range(n)], denominator=d, beat_groups=groups)
    xml = ET.fromstring(to_musicxml(Score("Test", [bar], tempo=200)))
    assert xml.findtext(".//time/beats") == str(n)
    assert xml.findtext(".//time/beat-type") == str(d)
    notes = xml.findall(".//measure/note")
    beams = [note.findtext("beam[@number='1']") for note in notes]
    expected = [state for group in groups for state in ["begin"] + ["continue"] * (group-2) + ["end"]]
    assert beams == expected
    assert sum(int(note.findtext("duration")) for note in notes) == n * unit
    compound = d == 8 and n in (6, 12)
    assert (xml.find(".//metronome/beat-unit-dot") is not None) == compound
    assert int(xml.findtext(".//metronome/per-minute")) == (133 if compound else 200)
    assert xml.find(".//sound").get("tempo") == "200"


def test_compound_eighths_are_not_printed_as_tuplets():
    a, t, hits = synthetic(6, 8, .2)
    score = quantize_signature(hits, t[::3], t[::6], activations=a, attacks=t[:-1], time_signature="6/8")
    xml = ET.fromstring(to_musicxml(score))
    assert not xml.findall(".//time-modification")
    assert score.bars[0].beat_groups == [3, 3]
    assert score.tempo == pytest.approx(150, abs=1)


@pytest.mark.parametrize("n,d", [(7,8), (6,8), (12,8), (7,16), (5,4)])
def test_midi_and_preview_timing_follow_every_pulse(tmp_path, n, d):
    unit = 96 // d
    times = np.r_[.3, .3 + np.cumsum(np.linspace(.12, .2, n))].tolist()
    bar = Bar(0, n, times, [1]*n, 180, [Note(i*unit, "kick", 90) for i in range(n)],
              denominator=d, beat_groups=Signature(n,d).groups)
    path = tmp_path / "test.mid"
    to_midi(Score("Test", [bar]), path)
    midi = mido.MidiFile(path)
    actual, seconds = [], 0
    for event in midi:
        seconds += event.time
        if event.type == "note_on":
            actual.append(seconds)
    assert actual == pytest.approx([t - times[0] for t in times[:-1]], abs=.00001)
    assert midi.length == pytest.approx(times[-1] - times[0], abs=.00001)
    assert [bar.tick_time(i*unit) for i in range(n+1)] == pytest.approx(times)
    signature = next(e for e in midi.tracks[0] if e.type == "time_signature")
    assert (signature.numerator, signature.denominator) == (n,d)


def test_old_score_json_is_byte_stable_and_explicit_roundtrips():
    legacy = Score("Legacy", [Bar(0,4,[0,.5,1,1.5,2],[4]*4,120)])
    text = legacy.to_json()
    assert '"denominator"' not in text and '"pulse"' not in text
    assert Score.from_json(text).to_json() == text
    explicit = Score("7/8", [Bar(0,7,[i*.15 for i in range(8)],[2]*7,200,
                                denominator=8,beat_groups=[2,2,3])])
    assert Score.from_json(explicit.to_json()) == explicit
    assert explicit.bars[0].ticks == 84


def test_title_hint_needs_audio_evidence_and_does_not_change_notes():
    a,t,_ = synthetic(7,8,.15)
    automatic = Score("7/8 practice", [Bar(i,4,[i*2.4+j*.6 for j in range(5)],[4]*4,100) for i in range(8)])
    before = automatic.to_json()
    suggestion = suggest_signature(automatic,a,t[:-1],np.arange(0,t[-1],.6),np.arange(0,t[-1],1.2))
    assert suggestion and suggestion["time_signature"] == "7/8"
    assert "title_hint" in suggestion["reasons"]
    assert automatic.to_json() == before
    assert suggest_signature(automatic,a,t[:-1],t,t[::4]) is None


def test_api_signature_patch_and_missing_cache_never_run_models(tmp_path, monkeypatch):
    from test_server import make_song

    song = make_song(tmp_path)
    called = []
    def run(path, meta, opts, **kwargs):
        called.append(opts)
    monkeypatch.setattr(pipeline, "run", run)
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/songs/groove/score", json={"time_signature":"7/8"})
        assert response.status_code == 200
        assert called[-1].time_signature == "7/8"
        assert response.json()["options"]["time_signature"] == "7/8"
        assert client.post("/api/songs/groove/score", json={"time_signature":"7/9"}).status_code == 422
        (song / "activations.mix.npy").unlink()
        assert client.post("/api/songs/groove/score", json={"beat_shift":.5}).status_code == 409
        assert len(called) == 1


def test_preview_uses_absolute_denominator_times(tmp_path, monkeypatch):
    import soundfile as sf
    from groovemap import render

    sr = 8000
    source, target = tmp_path / "audio.wav", tmp_path / "preview.wav"
    sf.write(source, np.zeros(sr*2), sr)
    monkeypatch.setattr(render, "_voice", lambda inst, rate, rng: np.array([1.0]))
    times = [.13, .29, .46, .62, .81, .98, 1.16, 1.35]
    bar = Bar(0,7,times,[2]*7,200,[Note(0,"kick",100),Note(30,"snare",100),Note(72,"kick",100)],
              denominator=8,beat_groups=[2,2,3])
    render.render_preview(Score("Preview",[bar]),source,target)
    wave, rate = sf.read(target)
    assert rate == sr
    assert np.flatnonzero(wave[:,1]).tolist() == [int(bar.tick_time(n.tick)*sr) for n in bar.notes]


def test_custom_output_group_order_and_two_voices():
    bar = Bar(0,7,[i*.15 for i in range(8)],[1]*7,200,
              [Note(i*12,inst,90) for i in range(7) for inst in ("kick","hihat")],
              denominator=8,beat_groups=[3,2,2])
    xml = ET.fromstring(to_musicxml(Score("Groups",[bar]),voices=2))
    for voice in ("1","2"):
        notes = [n for n in xml.findall(".//measure/note") if n.findtext("voice") == voice]
        assert [n.findtext("beam[@number='1']") for n in notes] == ['begin','continue','end','begin','end','begin','end']
        assert sum(int(n.findtext("duration")) for n in notes) == 84
    assert xml.findtext(".//backup/duration") == "84"


def test_cached_pipeline_uses_explicit_path_and_preserves_tracker_cache(tmp_path, monkeypatch):
    from groovemap.fetch import SongMeta

    a, t, hits = synthetic(7,8,.15)
    beats = np.arange(.03,t[-1],.613).tolist()
    cached = json.dumps({"beats":beats,"downbeats":beats[::2]})
    (tmp_path / "beats.json").write_text(cached)
    np.save(tmp_path / "activations.mix.npy",a)
    def forbidden(*args,**kwargs):
        pytest.fail("signature correction attempted model inference")
    monkeypatch.setattr(pipeline.models,"track_beats",forbidden)
    monkeypatch.setattr(pipeline.models,"drum_activations",forbidden)
    monkeypatch.setattr(pipeline.models,"separate_drums",forbidden)
    monkeypatch.setattr(pipeline.models,"pick_hits",lambda *args,**kwargs:hits)
    result = pipeline.run(tmp_path,SongMeta("file","unused.wav","7/8 test"),
                          pipeline.Options(onset_input="mix",time_signature="7/8",meter=4,odd_bars="detected",pdf=False))
    assert all((bar.beats,bar.denominator)==(7,8) for bar in result.score.bars)
    assert result.score.pulse['source'] == 'rebuilt'
    assert (tmp_path / "beats.json").read_text() == cached
    assert Score.load(result.files['json']) == result.score


def test_sparse_jittered_pattern_without_eighth_note_hat_guide():
    rng = np.random.default_rng(77)
    times = np.r_[.1, .1 + np.cumsum(np.linspace(.15,.1725,7*30))]
    activations = np.zeros((int((times[-1]+.05)*100),5))
    attacks = []
    for i,time in enumerate(times[:-1]):
        if i % 7 in (0,2,4,6):
            frame = round((time + rng.normal(0,.003))*100)
            activations[frame,0 if i%7==0 else 1 if i%7==4 else 3] = .9
            attacks.append(frame/100)
    activations = gaussian_filter1d(activations,.7,axis=0)
    pulse = build_pulse(activations,attacks,np.arange(0,times[-1],.613),[],Signature(7,8))
    assert pulse.source == "rebuilt"
    error = np.min(np.abs(np.array(pulse.times)[:,None] - times[None,:]),axis=0)
    assert np.quantile(error,.95) < .012
    assert (len(pulse.times)-1)//7 == 30
