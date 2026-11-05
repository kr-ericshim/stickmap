import json
from pathlib import Path

import mido
import pytest

from stickmap.midi import GM_NOTES, to_midi
from stickmap.quantize import quantize
from stickmap.score import Bar, Hit, Note, Score


def absolute(track):
    tick = 0
    events = []
    for event in track:
        tick += event.time
        events.append((tick, event))
    return events


def test_midi_positions_mapping_duration_and_velocity(tmp_path):
    bar = Bar(0, 4, [0, 0.5, 1, 1.5, 2], [4, 3, 4, 4], 120,
              [Note(0, "kick", 70), Note(0, "hihat", 110), Note(32, "tom", 95), Note(66, "snare", 85), Note(90, "cymbal", 100)])
    score = Score("Groove", [bar])
    path = tmp_path / "groove.mid"
    to_midi(score, path)
    midi = mido.MidiFile(path)
    assert midi.type == 1 and midi.ticks_per_beat == 480
    assert len(midi.tracks) == 2
    events = absolute(midi.tracks[1])
    ons = [(tick, event) for tick, event in events if event.type == "note_on"]
    assert len(ons) == len(bar.notes)
    assert sorted((tick, event.note, event.velocity) for tick, event in ons) == sorted((n.tick * 20, GM_NOTES[n.inst], n.velocity) for n in bar.notes)
    assert all(event.channel == 9 for _, event in events if event.type in ("note_on", "note_off"))
    offs = {(event.note, tick) for tick, event in events if event.type == "note_off"}
    assert all((event.note, tick + 60) in offs for tick, event in ons)
    assert [sum(event.time for event in track) for track in midi.tracks] == [1920, 1920]
    assert midi.length == pytest.approx(2)


def test_thirty_seconds_and_sextuplets_keep_their_place_without_overlapping(tmp_path):
    bar = Bar(0, 2, [0, 0.5, 1], [8, 6], 120, [Note(t, "snare", 90) for t in (0, 3, 6, 24, 28, 32)])
    path = tmp_path / "fills.mid"
    to_midi(Score("Fills", [bar]), path)
    events = [(tick, event.type) for tick, event in absolute(mido.MidiFile(path).tracks[1]) if event.type in ("note_on", "note_off")]
    assert [tick for tick, kind in events if kind == "note_on"] == [0, 60, 120, 480, 560, 640]
    assert [kind for _, kind in events] == ["note_on", "note_off"] * 6


def test_tempo_meter_changes_and_accumulation(tmp_path):
    bars = [Bar(i, beats, list(range(beats + 1)), [4] * beats, tempo, [Note(0, "kick", 100)])
            for i, (beats, tempo) in enumerate([(4, 120), (3, 120.5), (4, 120.6), (4, 121), (4, 121.2)])]
    path = tmp_path / "changes.mid"
    to_midi(Score("Changes", bars), path)
    midi = mido.MidiFile(path)
    conductor = absolute(midi.tracks[0])
    tempos = [(tick, mido.tempo2bpm(event.tempo)) for tick, event in conductor if event.type == "set_tempo"]
    assert [tick for tick, _ in tempos] == [0, 3360, 7200]
    assert [tempo for _, tempo in tempos] == pytest.approx([120, 120.6, 121.2], abs=0.001)
    assert [(tick, event.numerator, event.denominator) for tick, event in conductor if event.type == "time_signature"] == [(0, 4, 4), (1920, 3, 4), (3360, 4, 4)]
    assert [tick for tick, event in absolute(midi.tracks[1]) if event.type == "note_on"] == [0, 1920, 3360, 5280, 7200]
    assert sum(event.time for event in midi.tracks[1]) == 19 * 480


def test_empty_score_and_rest_only_bars(tmp_path):
    for score in (Score("", []), Score("Rest", [Bar(0, 3, [0, 0.5, 1, 1.5], [4] * 3, 120)])):
        path = tmp_path / "empty.mid"
        to_midi(score, path)
        midi = mido.MidiFile(path)
        assert not any(event.type == "note_on" for track in midi.tracks for event in track)
        assert midi.length == pytest.approx(sum(bar.beats * 60 / bar.tempo for bar in score.bars))


def test_fixture_end_to_end_and_length(tmp_path):
    raw = json.loads((Path(__file__).parent / "fixtures/funkorama.json").read_text())
    score = quantize([Hit(**hit) for hit in raw["hits"]], raw["beats"], raw["downbeats"])
    path = tmp_path / "funkorama.mid"
    to_midi(score, path)
    midi = mido.MidiFile(path)
    assert sum(event.type == "note_on" for event in midi.tracks[1]) == sum(len(bar.notes) for bar in score.bars)
    total_ticks = sum(bar.ticks for bar in score.bars) * 20
    assert all(sum(event.time for event in track) == total_ticks for track in midi.tracks)
    tempo, previous, seconds = 500000, 0, 0.0
    for tick, event in absolute(midi.tracks[0]):
        seconds += mido.tick2second(tick - previous, 480, tempo)
        if event.type == "set_tempo":
            tempo = event.tempo
        previous = tick
    assert midi.length == pytest.approx(seconds)
    # Thresholded tempo events approximate the original bar durations.
    assert midi.length == pytest.approx(score.bars[-1].beat_times[-1] - score.bars[0].start, abs=0.6)


def test_titles_outside_latin1_do_not_break_saving(tmp_path):
    for title in ("CJ THOMPSON - \u201cMighty Long Way\u201d", "아이유 - 블루밍", ""):
        path = tmp_path / "out.mid"
        to_midi(Score(title, [Bar(0, 4, [0.0, 0.5, 1.0, 1.5, 2.0], [4] * 4, 120.0, [Note(0, "kick", 100)])]), path)
        assert len(mido.MidiFile(path).tracks) == 2
