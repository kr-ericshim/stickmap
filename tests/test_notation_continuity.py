from copy import deepcopy
from fractions import Fraction
import xml.etree.ElementTree as ET

import pytest

from groovemap.musicxml import _continuous_tuplets, _duration_choice, _rhythm, to_musicxml
from groovemap.score import Bar, Note, Score


def make_bar(index, ticks, modes, inst="snare"):
    return Bar(index, len(modes), [index * 2 + i * .5 for i in range(len(modes) + 1)], modes, 120,
               [Note(t, inst, 90, raw_time=index * 2 + t / 48) for t in ticks])


def signature(events):
    return [(e.tick, e.duration, e.notes) for e in events]


def test_sextuplet_continuity_across_barline_preserves_notes_rests_and_durations():
    bars = [make_bar(0, [0, 4, 8, 12, 16, 20], [6]),
            make_bar(1, [0, 4, 8, 16, 20], [6]),
            make_bar(2, [0, 4, 8, 12, 16, 20], [6])]
    original = [_rhythm(b, b.notes) for b in bars]
    snapshot = deepcopy(original)
    result = _continuous_tuplets(bars, original)
    assert [signature(e) for e in result] == [signature(e) for e in original]
    assert original == snapshot
    assert {e.ratio for events in result for e in events} == {(6, 4)}
    assert sum("start" in e.tuplets for e in result[1]) == 1
    # The absent halfway onset leaves a leading rest in the second old group.
    assert any(not e.notes for e in result[1])
    assert result[1][0].bracket


def test_straight_half_is_never_swallowed_and_standalone_triplets_stay_triplets():
    bar = make_bar(0, [0, 4, 8, 12, 18, 24, 28, 32, 36, 40, 44], [[6, 4], 6])
    source = [_rhythm(bar, bar.notes)]
    assert _continuous_tuplets([bar], source) == source
    triplets = make_bar(0, [0, 8, 16, 24, 32, 40], [3, 3])
    assert _continuous_tuplets([triplets], [_rhythm(triplets, triplets.notes)]) == [_rhythm(triplets, triplets.notes)]


@pytest.mark.parametrize("voices", [1, 2])
def test_xml_tuplet_ratios_and_independent_voice_durations(voices):
    bars = [make_bar(0, [0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 44], [6, 6])]
    bars[0].notes += [Note(n.tick, "kick", 90, raw_time=n.raw_time) for n in bars[0].notes]
    xml = ET.fromstring(to_musicxml(Score("Tuplets", bars), voices=voices))
    totals = {}
    written = {"32nd": 3, "16th": 6, "eighth": 12, "quarter": 24}
    for n in xml.findall(".//note"):
        duration = int(n.findtext("duration"))
        ratio = n.find("time-modification")
        nominal = Fraction(written[n.findtext("type")])
        if ratio is not None:
            nominal *= Fraction(int(ratio.findtext("normal-notes")), int(ratio.findtext("actual-notes")))
        assert nominal == duration
        if n.find("chord") is None:
            voice = n.findtext("voice")
            totals[voice] = totals.get(voice, 0) + duration
    assert totals == ({"1": 48} if voices == 1 else {"1": 48, "2": 48})


def test_exact_gap_choice_keeps_onsets_leading_rest_and_beat_edges():
    bar = make_bar(0, [3, 15, 24, 30, 36, 42], [8, 4])
    events = _duration_choice(bar, bar.notes)
    assert [(e.tick, e.notes) for e in events if e.notes] == [(e.tick, e.notes) for e in _rhythm(bar, bar.notes) if e.notes]
    # The exact 12-tick gap remains eligible; the isolated attack at 15 now
    # keeps its 32nd value instead of stretching into a dotted 16th.
    assert [(e.tick, e.duration) for e in events if not e.notes] == [(0, 3), (18, 6)]
    assert next(e.duration for e in events if e.tick == 3) == 12
    assert all(e.tick // 24 == (e.tick + e.duration - 1) // 24 for e in events)


def test_ordered_32nd_chops_are_unchanged():
    bar = make_bar(0, list(range(0, 96, 3)), [8] * 4)
    source = _rhythm(bar, bar.notes)
    assert _duration_choice(bar, bar.notes) == source
    assert _continuous_tuplets([bar], [source]) == [source]
    assert len(source) == 32
    assert all(e.notes and e.duration == 3 and 1 in e.beams and not e.tuplet for e in source)
