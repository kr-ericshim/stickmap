from collections import defaultdict
from fractions import Fraction
from itertools import combinations, product
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from groovemap.musicxml import to_musicxml
from groovemap.quantize import quantize
from groovemap.score import INSTRUMENTS, Bar, Hit, Note, Score


def bar(notes=(), *, beats=4, subdivisions=None, tempo=120, index=0):
    return Bar(index, beats, [i * 60 / tempo for i in range(beats + 1)], subdivisions or [4] * beats, tempo, list(notes))


def verify(score, voices=1):
    text = to_musicxml(score, voices=voices)
    assert text.startswith('<?xml version="1.0"')
    assert '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"' in text
    root = ET.fromstring(text)
    assert root.attrib == {"version": "4.0"}
    measures = root.findall("part/measure")
    assert len(measures) == len(score.bars)
    for measure, expected in zip(measures, score.bars):
        durations = defaultdict(int)
        active_beams = set()
        active_tuplets = set()
        cursor = 0
        chord_start = None
        for item in measure:
            if item.tag == "backup":
                assert int(item.findtext("duration")) == expected.ticks
                cursor -= expected.ticks
                assert cursor == 0
                chord_start = None
                continue
            if item.tag != "note":
                continue
            voice = item.findtext("voice")
            duration = int(item.findtext("duration"))
            kind = item.findtext("type")
            if kind is not None:
                written = Fraction({"32nd": 3, "16th": 6, "eighth": 12, "quarter": 24, "half": 48}[kind])
                written *= sum(Fraction(1, 2 ** i) for i in range(len(item.findall("dot")) + 1))
                modification = item.find("time-modification")
                if modification is not None:
                    written *= Fraction(int(modification.findtext("normal-notes")), int(modification.findtext("actual-notes")))
                assert duration == written
            else:
                assert item.find("rest").get("measure") == "yes" and duration == expected.ticks
            chord = item.find("chord") is not None
            if chord:
                assert chord_start is not None
                start = chord_start
                assert item.find("beam") is None
            else:
                start = cursor
                chord_start = start
                cursor += duration
                durations[voice] += duration
            if item.find("rest") is not None:
                assert item.find("beam") is None
                assert not any(key[0] == voice for key in active_beams)
            for beam in item.findall("beam"):
                key = (voice, beam.get("number"))
                if beam.text == "begin":
                    assert key not in active_beams
                    active_beams.add(key)
                elif beam.text == "continue":
                    assert key in active_beams
                elif beam.text == "end":
                    assert key in active_beams
                    active_beams.remove(key)
                else:
                    assert beam.text in ("forward hook", "backward hook")
                    assert (voice, "1") in active_beams or item.find("beam").text == "end"
            for tuplet in item.findall("notations/tuplet"):
                key = (voice, tuplet.get("number"))
                if tuplet.get("type") == "start":
                    assert key not in active_tuplets
                    active_tuplets.add(key)
                else:
                    assert tuplet.get("type") == "stop" and key in active_tuplets
                    active_tuplets.remove(key)
            if cursor % 24 == 0:
                assert not any(key[0] == voice for key in active_beams)
                assert not any(key[0] == voice for key in active_tuplets)
            if item.find("rest") is None:
                assert item.findtext("stem") == ("up" if voice == "1" else "down")
                assert item.find("instrument").get("id") in {f"P1-{inst}" for inst in INSTRUMENTS}
            assert 0 <= start < expected.ticks
        assert cursor == expected.ticks
        assert durations["1"] == expected.ticks
        assert all(total == expected.ticks for total in durations.values())
        assert not active_beams and not active_tuplets
    return root


def test_document_metadata_mapping_chords_and_escaping():
    score = Score('A & <B>', [bar([Note(0, inst, 100) for inst in INSTRUMENTS])], subtitle='"Take" & two')
    root = verify(score)
    assert root.findtext("work/work-title") == score.title
    assert root.findtext("movement-title") == score.title
    assert root.findtext("identification/creator[@type='composer']") == score.subtitle
    assert root.findtext("part-list/score-part/part-name") == "Drumset"
    midi = {node.get("id"): int(node.findtext("midi-unpitched")) for node in root.findall("part-list/score-part/midi-instrument")}
    assert midi == {"P1-kick": 37, "P1-snare": 39, "P1-sidestick": 38, "P1-hihat": 43, "P1-hihat_open": 47,
                    "P1-hihat_pedal": 45, "P1-tom_high": 51, "P1-tom": 48, "P1-tom_low": 44, "P1-ride": 52,
                    "P1-ride_bell": 54, "P1-cymbal": 50}
    assert all(node.findtext("midi-channel") == "10" for node in root.findall("part-list/score-part/midi-instrument"))
    assert root.findtext("part/measure/attributes/divisions") == "24"
    assert root.findtext("part/measure/attributes/clef/sign") == "percussion"
    notes = root.findall("part/measure/note")[:12]
    assert sum(n.find("chord") is not None for n in notes) == 11
    assert [(n.findtext("unpitched/display-step"), n.findtext("unpitched/display-octave"), n.findtext("notehead")) for n in notes] == [
        ("F", "4", "normal"), ("C", "5", "normal"), ("C", "5", "x"),
        ("G", "5", "x"), ("G", "5", "circle-x"), ("D", "4", "x"),
        ("E", "5", "normal"), ("D", "5", "normal"), ("A", "4", "normal"),
        ("F", "5", "x"), ("F", "5", "diamond"), ("A", "5", "x")]


def test_feet_share_the_second_voice():
    notes = [Note(0, "kick", 100), Note(0, "hihat", 100), Note(24, "hihat_pedal", 80), Note(24, "sidestick", 90)]
    root = verify(Score("Feet", [bar(notes)]), voices=2)
    voice = {n.find("instrument").get("id"): n.findtext("voice") for n in root.findall("part/measure/note") if n.find("rest") is None}
    assert voice == {"P1-kick": "2", "P1-hihat_pedal": "2", "P1-hihat": "1", "P1-sidestick": "1"}


@pytest.mark.parametrize("subdiv,slots", [
    (4, [0, 6, 12, 18]), (2, [0, 12]), (3, [0, 8, 16]), (8, list(range(0, 24, 3))), (6, list(range(0, 24, 4)))])
def test_every_per_beat_onset_subset(subdiv, slots):
    for size in range(len(slots) + 1):
        for pattern in combinations(slots, size):
            notes = [Note(beat * 24 + tick, "hihat", 90) for beat in range(4) for tick in pattern]
            root = verify(Score("Patterns", [bar(notes, subdivisions=[subdiv] * 4)]))
            assert {n.findtext("type") for n in root.findall("part/measure/note")} <= {"32nd", "16th", "eighth", "quarter", "half", None}


def test_rest_spelling_and_empty_bars():
    root = verify(Score("Rests", [bar([Note(18, "snare", 100)]), bar(beats=3)]))
    measures = root.findall("part/measure")
    first = measures[0].findall("note")
    assert [(n.findtext("duration"), n.findtext("type")) for n in first[:2]] == [("12", "eighth"), ("6", "16th")]
    assert [int(n.findtext("duration")) for n in first[3:]] == [24, 48]
    rest = measures[1].find("note")
    assert rest.find("rest").get("measure") == "yes"
    assert rest.findtext("duration") == "72"


def test_dotted_eighth_and_secondary_beam_hooks():
    root = verify(Score("Hooks", [bar([Note(t, "hihat", 100) for t in (0, 18, 24, 30, 42, 48, 60, 66)])]))
    first = root.find("part/measure/note")
    assert first.findtext("duration") == "18" and first.find("dot") is not None
    beams = [b.text for b in root.findall("part/measure/note/beam[@number='2']")]
    assert "forward hook" in beams and "backward hook" in beams
    assert "begin" in beams and "end" in beams


def test_secondary_hook_points_into_group_after_leading_rest():
    root = verify(Score("Rest then beam", [bar([Note(6, "snare", 90), Note(12, "snare", 90)])]))
    assert root.findtext("part/measure/note/beam[@number='2']") == "forward hook"


def test_triplet_quarter_rest_and_plain_quarter():
    root = verify(Score("Shuffle", [bar([Note(0, "hihat", 100), Note(16, "hihat", 100), Note(40, "snare", 100), Note(48, "snare", 100)], subdivisions=[3] * 4)]))
    notes = root.findall("part/measure/note")
    assert notes[0].findtext("type") == "quarter"
    assert notes[0].findtext("time-modification/actual-notes") == "3"
    assert notes[0].findtext("time-modification/normal-notes") == "2"
    assert notes[2].find("rest") is not None and notes[2].findtext("duration") == "16"
    plain = next(n for n in notes if n.findtext("duration") == "24" and n.find("rest") is None)
    assert plain.find("time-modification") is None


def tempo_marks(tempos, song_tempo=None):
    bars = [bar(tempo=tempo, index=i) for i, tempo in enumerate(tempos)]
    root = verify(Score("Tempo", bars, tempo=song_tempo or tempos[0]))
    marks = {}
    for number, measure in enumerate(root.findall("part/measure")):
        for direction in measure.findall("direction"):
            assert direction.get("placement") == "above" and direction.findtext("direction-type/metronome/beat-unit") == "quarter"
            assert direction.find("sound").get("tempo") == direction.findtext("direction-type/metronome/per-minute")
            marks[number] = int(direction.find("sound").get("tempo"))
    assert len(root.findall("part/measure/direction")) == len(marks) == len(root.findall(".//sound"))
    return marks


def test_time_changes_and_one_tempo_mark_despite_jitter():
    bars = [bar(beats=beats, tempo=tempo, index=i) for i, (beats, tempo) in enumerate([(4, 120), (3, 123), (4, 123.1), (4, 125), (4, 127)])]
    root = verify(Score("Changes", bars, tempo=120.4))
    assert [n.text for n in root.findall("part/measure/attributes/time/beats")] == ["4", "3", "4"]
    assert [n.get("tempo") for n in root.findall("part/measure/direction/sound")] == ["120"]
    assert tempo_marks([85, 90, 88, 91, 89, 90, 86, 92, 88, 90, 87, 91], song_tempo=89.5) == {0: 90}


def test_tempo_mark_only_for_an_abrupt_lasting_change():
    assert tempo_marks([120] * 8 + [170] * 8, song_tempo=145) == {0: 120, 8: 170}
    assert tempo_marks([85 + 10 * i / 29 for i in range(30)], song_tempo=90) == {0: 90}
    assert tempo_marks([100] * 6 + [140] + [100] * 6, song_tempo=100) == {0: 100}
    assert tempo_marks([100] * 8 + [140] * 2 + [100] * 10, song_tempo=100) == {0: 100}
    assert tempo_marks([100] * 8 + [140] * 4 + [180] * 4 + [100] * 10, song_tempo=100) == {0: 100}


def test_ghost_note_head_is_parenthesized():
    root = verify(Score("Ghost", [bar([Note(0, "snare", 100), Note(6, "snare", 45, ghost=True)])]))
    heads = root.findall("part/measure/note/notehead")
    assert [(head.text, head.get("parentheses")) for head in heads] == [("normal", None), ("normal", "yes")]


def test_two_voices_and_omit_absent_kick_voice():
    notes = [Note(t, inst, 100) for t, inst in [(0, "kick"), (0, "hihat"), (12, "hihat"), (24, "snare"), (48, "kick"), (60, "tom")]]
    root = verify(Score("Voices", [bar(notes), bar([Note(0, "snare", 100)]), bar([Note(0, "kick", 100)])]), voices=2)
    measures = root.findall("part/measure")
    assert measures[0].findtext("backup/duration") == "96"
    assert measures[1].find("backup") is None
    assert measures[2].find("note/rest").get("measure") == "yes"
    assert all(n.find("instrument").get("id") == "P1-kick" for n in measures[0].findall("note[voice='2']") if n.find("rest") is None)


@pytest.mark.parametrize("voices", [1, 2])
def test_fixture_end_to_end(voices):
    raw = json.loads((Path(__file__).parent / "fixtures/funkorama.json").read_text())
    score = quantize([Hit(**hit) for hit in raw["hits"]], raw["beats"], raw["downbeats"], title="Funkorama")
    root = verify(score, voices)
    assert len(root.findall("part/measure/note/unpitched")) == sum(len(bar.notes) for bar in score.bars)


def beams(notes, number):
    return [note.findtext(f"beam[@number='{number}']") for note in notes]


def fills():
    beats = [i * 0.6 for i in range(16)]
    hits = [Hit((beat + slot / 4) * 0.6, "hihat", 0.8) for beat in range(16) if beat not in (5, 6, 9) for slot in range(4)]
    hits += [Hit((5 + k / 8) * 0.6, "snare", 0.8) for k in range(8)]
    hits += [Hit((6 + f) * 0.6, "snare", 0.8) for f in (0, 1 / 8, 1 / 4, 1 / 2, 3 / 4)]
    hits += [Hit((9 + k / 6) * 0.6, "tom", 0.8) for k in range(6)]
    return quantize(hits, beats, beats[::4], title="Fills")


@pytest.mark.parametrize("voices", [1, 2])
def test_thirty_second_run_breaks_inner_beams_at_the_eighth(voices):
    score = fills()
    assert score.bars[1].subdivisions == [4, 8, [8, 4], 4]
    notes = verify(score, voices).findall("part/measure")[1].findall("note")
    run = notes[4:12]
    assert [(n.findtext("type"), n.findtext("duration")) for n in run] == [("32nd", "3")] * 8
    assert beams(run, 1) == ["begin"] + ["continue"] * 6 + ["end"]
    for number in (2, 3):
        assert beams(run, number) == ["begin", "continue", "continue", "end"] * 2
    doubles = notes[12:17]
    assert [n.findtext("type") for n in doubles] == ["32nd", "32nd", "16th", "16th", "16th"]
    assert beams(doubles, 1) == ["begin", "continue", "continue", "continue", "end"]
    assert beams(doubles, 2) == ["begin", "continue", "end", "begin", "end"]
    assert beams(doubles, 3) == ["begin", "end", None, None, None]


def compound_bar(ticks):
    return Bar(0, 6, [i * 0.25 for i in range(7)], [2] * 6, 120, [Note(t, "snare", 90) for t in ticks], denominator=8, beat_groups=[3, 3])


def test_plain_sixteenths_keep_one_inner_beam():
    root = verify(Score("Sixteenths", [bar([Note(t, "hihat", 90) for t in range(0, 24, 6)])]))
    notes = root.findall("part/measure/note")[:4]
    assert beams(notes, 2) == ["begin", "continue", "continue", "end"]
    # A 32nd beyond a rest has its own beam and does not subdivide this one.
    notes = verify(Score("Apart", [compound_bar([0, 6, 12, 18, 33])])).findall("part/measure/note")
    assert [n.findtext("type") for n in notes[:6]] == ["16th", "16th", "16th", "eighth", "32nd", "32nd"]
    assert beams(notes[:4], 2) == ["begin", "continue", "end", None]


def test_no_eighth_break_that_would_leave_a_lone_hook():
    notes = verify(Score("Lone", [compound_bar([0, 3, 6, 9, 12, 18, 30])])).findall("part/measure/note")
    assert [n.findtext("type") for n in notes[:7]] == ["32nd"] * 4 + ["16th", "eighth", "16th"]
    assert beams(notes[:5], 2) == ["begin", "continue", "continue", "continue", "end"]
    assert beams(notes[:5], 3) == ["begin", "continue", "continue", "end", None]


def test_sextuplet_beat_is_one_group_of_six():
    score = fills()
    assert score.bars[2].subdivisions == [4, 6, 4, 4]
    group = verify(score).findall("part/measure")[2].findall("note")[4:10]
    assert [(n.findtext("type"), n.findtext("duration")) for n in group] == [("16th", "4")] * 6
    assert {tuple(n.findtext(f"time-modification/{tag}") for tag in ("actual-notes", "normal-notes", "normal-type")) for n in group} == {("6", "4", "16th")}
    assert [[t.get("type") for t in n.findall("notations/tuplet")] for n in group] == [["start"], [], [], [], [], ["stop"]]
    assert group[0].find("notations/tuplet").get("bracket") == "no"
    assert beams(group, 1) == ["begin"] + ["continue"] * 4 + ["end"]
    assert beams(group, 2) == ["begin", "continue", "end"] * 2
    assert beams(group, 3) == [None] * 6


def spelled_tuplets(ticks):
    root = verify(Score("Tuplets", [bar([Note(t, "snare", 90) for t in ticks])]))
    notes = root.findall("part/measure/note")
    starts = [n.find("notations/tuplet[@type='start']") for n in notes]
    return [(n.findtext("time-modification/actual-notes"), start.get("bracket")) for n, start in zip(notes, starts) if start is not None]


def test_sextuplet_grouping_and_brackets():
    # Six in one beat, and two tuplet 8ths around the middle: one beamed 6, no bracket.
    assert spelled_tuplets([0, 4, 8, 12, 16, 20]) == [("6", "no")]
    assert spelled_tuplets([0, 8, 12, 20]) == [("6", "no")]
    # A leading rest stays inside the 6, which then keeps its bracket.
    assert spelled_tuplets([4, 8, 12, 16, 20]) == [("6", "yes")]
    # Nothing on the half beat, or a note held over it: two triplets show where it falls.
    assert spelled_tuplets([4, 8, 16, 20]) == [("3", "yes"), ("3", "yes")]
    assert spelled_tuplets([0, 8, 16, 20]) == [("3", "no"), ("3", "yes")]
    assert spelled_tuplets([0, 4, 12]) == [("3", "no")]
    # Eighth triplets: beamed ones need only the number, a quarter inside keeps the bracket.
    assert spelled_tuplets([0, 8, 16]) == [("3", "no")]
    assert spelled_tuplets([0, 16]) == [("3", "yes")]


def test_half_sextuplet_beside_a_plain_eighth():
    ticks = [(0, "snare"), (16, "snare"), (20, "snare"), (36, "tom"), (40, "tom"), (44, "tom")]
    root = verify(Score("Half", [bar([Note(t, inst, 90) for t, inst in ticks], subdivisions=[6, 6, 4, 4])]))
    notes = root.findall("part/measure/note")
    spelled = [(n.findtext("type"), n.findtext("duration"), n.find("rest") is not None) for n in notes]
    assert spelled[:4] == [("eighth", "12", False), ("16th", "4", True), ("16th", "4", False), ("16th", "4", False)]
    assert spelled[4:8] == [("eighth", "12", True), ("16th", "4", False), ("16th", "4", False), ("16th", "4", False)]
    assert notes[0].find("time-modification") is None and notes[0].find("beam") is None
    assert [n.findtext("time-modification/normal-type") for n in notes[:8]] == [None, "16th", "16th", "16th", None, "16th", "16th", "16th"]
    assert [[t.get("type") for t in n.findall("notations/tuplet")] for n in notes[1:4]] == [["start"], [], ["stop"]]
    assert beams(notes[1:4], 1) == beams(notes[1:4], 2) == [None, "begin", "end"]
    assert beams(notes[5:8], 2) == ["begin", "continue", "end"]
    # Swung eighths inside a sextuplet beat are still an eighth-note triplet.
    swung = verify(Score("Swung", [bar([Note(0, "snare", 90), Note(16, "snare", 90)], subdivisions=[6, 4, 4, 4])])).findall("part/measure/note")
    assert [(n.findtext("type"), n.findtext("time-modification/normal-type")) for n in swung[:2]] == [("quarter", "eighth"), ("eighth", "eighth")]


def test_dotted_sixteenths_thirty_second_rests_and_hooks():
    root = verify(Score("Dots", [bar([Note(t, "snare", 90) for t in (0, 9, 12, 27, 72, 75)])]))
    notes = root.findall("part/measure/note")
    spelled = [(n.findtext("type"), n.findtext("duration"), n.find("dot") is not None, n.find("rest") is not None) for n in notes]
    assert spelled == [
        ("16th", "9", True, False), ("32nd", "3", False, False), ("eighth", "12", False, False),
        ("32nd", "3", False, True), ("16th", "9", True, False), ("eighth", "12", False, True),
        ("quarter", "24", False, True),
        ("32nd", "3", False, False), ("16th", "9", True, False), ("eighth", "12", False, True)]
    assert beams(notes[:3], 1) == ["begin", "continue", "end"]
    assert beams(notes[:3], 2) == ["begin", "end", None]
    assert beams(notes[:3], 3) == [None, "backward hook", None]
    assert beams(notes[7:9], 1) == beams(notes[7:9], 2) == ["begin", "end"]
    assert beams(notes[7:9], 3) == ["forward hook", None]
    assert [child.tag for child in notes[0]] == ["unpitched", "duration", "instrument", "voice", "type", "dot", "stem", "notehead", "beam", "beam"]
    tuplet = verify(Score("Order", [bar([Note(0, "snare", 90), Note(4, "snare", 90)], subdivisions=[6, 4, 4, 4])])).find("part/measure/note")
    assert [child.tag for child in tuplet] == ["unpitched", "duration", "instrument", "voice", "type", "time-modification", "stem", "notehead", "beam", "beam", "notations"]


SONGS = sorted((Path(__file__).parents[1] / "songs").glob("*/beats.json"))


@pytest.mark.parametrize("beats_file", SONGS, ids=lambda path: path.parent.name[:20])
def test_real_songs_spell_into_whole_measures(beats_file):
    np = pytest.importorskip("numpy")
    models = pytest.importorskip("groovemap.models")
    pytest.importorskip("adtof_pytorch")
    grid = json.loads(beats_file.read_text())
    hits = models.pick_hits(np.load(next(beats_file.parent.glob("activations.drums-*.npy"))))
    score = quantize(hits, grid["beats"], grid["downbeats"], title=beats_file.parent.name)
    assert sum(len(bar.notes) for bar in score.bars) > 0.95 * len(hits)
    for voices in (1, 2):
        root = verify(score, voices)
        assert len(root.findall("part/measure/note/unpitched")) == sum(len(bar.notes) for bar in score.bars)
        assert {n.text for n in root.findall("part/measure/note/type")} <= {"32nd", "16th", "eighth", "quarter", "half"}


def test_invalid_voice_count():
    with pytest.raises(ValueError):
        to_musicxml(Score("", []), voices=3)


@pytest.mark.parametrize("voices", [1, 2])
@pytest.mark.parametrize("first,second", list(product((4, 8, 6), repeat=2)) + [(3, 3)])
def test_every_candidate_onset_subset_has_correct_written_durations(first, second, voices):
    slots = [0, 8, 16] if first == 3 else sorted(set(range(0, 12, 24 // first)) | set(range(12, 24, 24 // second)))
    for size in range(len(slots) + 1):
        for onsets in combinations(slots, size):
            notes = [Note(tick, "snare", 90) for tick in onsets]
            notes += [Note(tick, "kick", 90) for tick in onsets[::2]]
            root = verify(Score("Mixed halves", [bar(notes, beats=1, subdivisions=[[first, second]])]), voices)
            assert len(root.findall("part/measure/note/unpitched")) == len(notes)


def test_incompatible_simultaneous_grids_are_rejected():
    with pytest.raises(ValueError, match="Unsupported simultaneous grids"):
        verify(Score("Unsupported", [bar([Note(3, "snare", 90), Note(4, "hihat", 90)])]))
