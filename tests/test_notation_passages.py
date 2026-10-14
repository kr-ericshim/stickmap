import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from groovemap.musicxml import to_musicxml
from groovemap.quantize import _choices, _strokes, quantize
from groovemap.score import Hit

FIXTURE = json.loads((Path(__file__).parent / "fixtures/notation_passages.json").read_text())


def passage(name, triplets="auto"):
    data = FIXTURE["regions"][name]
    hits = [Hit(**h) for h in data["hits"]]
    return quantize(hits, data["beats"], data["downbeats"], attacks=data["attacks"], triplets=triplets), hits


@pytest.mark.parametrize("triplets", ["off", "auto"])
def test_tony_74_is_a_preserved_sixteenth_run(triplets):
    score, hits = passage("m74", triplets)
    bar = score.bars[74 - FIXTURE["regions"]["m74"]["first_bar"]]
    assert bar.subdivisions == [4] * 4
    assert len(bar.notes) == 23 and len({n.tick for n in bar.notes}) == 16
    assert score.merged_strokes == 0
    assert {n.raw_time for n in bar.notes} <= {h.time for h in hits}
    xml = ET.fromstring(to_musicxml(score)).findall("part/measure")[bar.index]
    assert not xml.findall("note/rest")
    assert {n.findtext("type") for n in xml.findall("note")} == {"16th"}
    assert len(xml.findall("note/beam[@number='1'][.='begin']")) == 4


def test_tony_93_94_sextuplet_run_connects_to_six_to_four_musicxml():
    score, hits = passage("sextuplet93")
    start = 93 - FIXTURE["regions"]["sextuplet93"]["first_bar"]
    assert score.bars[start].subdivisions[2:] == [6, 6]
    assert score.bars[start + 1].subdivisions[:2] == [6, 6]
    assert score.merged_strokes == 0
    assert sum(len(b.notes) for b in score.bars) == len(_strokes(hits))
    measures = ET.fromstring(to_musicxml(score)).findall("part/measure")
    for bar in measures[start:start + 2]:
        assert bar.findall("note/time-modification[actual-notes='6'][normal-notes='4']")
    off, _ = passage("sextuplet93", "off")
    assert not ET.fromstring(to_musicxml(off)).findall(".//time-modification")


def test_stable_tony_85_86_groove_stays_straight():
    score, _ = passage("straight85")
    assert all(b.subdivisions == [4] * 4 for b in score.bars[:2])


def test_unison_has_one_grid_evidence_vote():
    fractions = (0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6)
    solo = [(Hit(f * 0.5, "snare", 1), f) for f in fractions]
    chord = solo + [(Hit(f * 0.5, "kick", 1), f) for f in fractions]
    a = min(_choices(solo, .5, True, False), key=lambda c: (c.lost, c.cost))
    b = min(_choices(chord, .5, True, False), key=lambda c: (c.lost, c.cost))
    assert a.modes == b.modes == (6, 6)
    assert a.cost == pytest.approx(b.cost)
