from notation_metric import measure
from groovemap.musicxml import to_musicxml
from groovemap.score import Bar, Hit, Note, Score


def test_metric_counts_chords_once_and_matches_raw_timing():
    bar = Bar(0, 1, [0, 0.5], [8], 120, [Note(3, "snare", 45, True), Note(3, "kick", 90), Note(9, "snare", 90)])
    hits = [Hit(0.06, "snare", 0.1), Hit(0.06, "kick", 1), Hit(0.19, "snare", 1)]
    result = measure(Score("Audit", [bar]), to_musicxml(Score("Audit", [bar])), hits)["total"]
    assert result["notes"] == 3 and result["events"] == 2
    assert result["timing_samples"] == 3 and result["timing_unmatched"] == 0
    assert abs(result["timing_mae_ms"] - 2.5) < 1e-9
    assert result["ghost_ratio"] == 1 / 3 and result["snare_ghost_ratio"] == 0.5


def test_unmatched_raw_onset_is_not_zero_error():
    score = Score("Missing", [Bar(0, 1, [0, 0.5], [4], 120, [Note(0, "snare", 90)])])
    result = measure(score, to_musicxml(score), [Hit(5, "snare", 1)])["total"]
    assert result["timing_samples"] == 0 and result["timing_unmatched"] == 1
    assert result["timing_mae_ms"] is None


def test_all_engraving_counters_against_an_independent_xml_example():
    # Two beats: 32nd + 32nd rest + two dotted 16ths, then six tuplet 16ths.
    # Handwritten XML keeps this test independent of the speller under audit.
    xml = '<score-partwise><part><measure>'
    xml += '<note><duration>3</duration><type>32nd</type></note>'
    xml += '<note><chord/><duration>3</duration><type>32nd</type></note>'
    xml += '<note><rest/><duration>3</duration><type>32nd</type></note>'
    for beam in ('begin', 'end'):
        xml += f'<note><duration>9</duration><type>16th</type><dot/><beam number="1">{beam}</beam></note>'
    for beam in ('begin', 'continue', 'continue', 'continue', 'continue', 'end'):
        xml += f'<note><duration>4</duration><type>16th</type><time-modification><actual-notes>6</actual-notes><normal-notes>4</normal-notes></time-modification><beam number="1">{beam}</beam></note>'
    xml += '</measure></part></score-partwise>'
    notes = [Note(t, 'snare', 90) for t in (0, 6, 15, 24, 28, 32, 36, 40, 44)] + [Note(0, 'kick', 90)]
    score = Score('Counters', [Bar(0, 2, [0, .5, 1], [8, 6], 120, notes)])
    hits = [Hit(n.tick / 48, n.inst, 1) for n in notes]
    total = measure(score, xml, hits)['total']
    assert [total[k] for k in ('notes', 'events', 'notes_32nd', 'short_rests', 'unbeamed_flags', 'dots', 'tuplet_switches')] == [10, 9, 1, 1, 1, 2, 1]
