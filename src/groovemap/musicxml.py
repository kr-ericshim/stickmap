"""Spell a quantized drum score as readable MusicXML 4.0."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from statistics import median
import xml.etree.ElementTree as ET

from .score import DIVISIONS, INSTRUMENTS, KIT, Bar, Note, Score

_FEET = ("kick", "hihat_pedal")  # the second voice, stems down
_DOCTYPE = '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">'
# Written value per duration in ticks: 9 and 18 are dotted, 4, 8 and 16 only occur inside tuplets.
_TYPES = {3: "32nd", 4: "16th", 6: "16th", 8: "eighth", 9: "16th", 12: "eighth", 16: "quarter", 18: "eighth", 24: "quarter", 36: "quarter", 48: "half"}
_BEAMS = {"eighth": 1, "16th": 2, "32nd": 3}
# Straight values, longest first, each with the tick spacing it may start on
# (an 8th note may sit on any 16th: 16th-8th-16th is the usual syncopation).
_NOTES = ((36, 12), (24, 24), (18, 6), (12, 6), (9, 3), (6, 6), (3, 3))
_RESTS = ((36, 12), (24, 24), (12, 12), (6, 6), (3, 3))


def _element(parent: ET.Element, tag: str, text: object | None = None, **attributes: str) -> ET.Element:
    child = ET.SubElement(parent, tag, attributes)
    if text is not None:
        child.text = str(text)
    return child


@dataclass
class _Rhythm:
    tick: int
    duration: int
    notes: list[Note] = field(default_factory=list)
    tuplet: str = ""  # inside a tuplet of this note value
    measure_rest: bool = False
    beams: dict[int, str] = field(default_factory=dict)
    tuplets: list[str] = field(default_factory=list)
    ratio: tuple[int, int] = (3, 2)  # actual:normal notes of that tuplet
    bracket: bool = True  # False when a beam already shows the tuplet's extent

    @property
    def note_type(self) -> str:
        return _TYPES[self.duration]


def _tuplet_edge(events: list[_Rhythm], i: int) -> bool:
    """A tuplet starts at event i or ended just before it."""
    return i > 0 and ("start" in events[i].tuplets or "stop" in events[i - 1].tuplets)


def _eighth_breaks(events: list[_Rhythm], depth: list[int]) -> set[int]:
    """Events before which the inner beams break at an 8th.

    Only inside a primary beam that holds 32nds or 16th triplets, so a run reads
    as countable packets, and only where the 16th beam keeps at least two notes
    on each side: a lone hook there would not survive import into MuseScore.
    """
    breaks = set()
    start = 0
    while start < len(events):
        end = start
        while end < len(events) and depth[end]:
            end += 1
        if any(depth[i] == 3 or events[i].tuplet == "16th" for i in range(start, end)):
            for i in range(start + 2, end - 1):
                if (events[i].tick % (DIVISIONS // 2) == 0 and min(depth[i - 2:i + 2]) >= 2
                        and not any(_tuplet_edge(events, j) for j in (i - 1, i, i + 1))):
                    breaks.add(i)
        start = end + 1
    return breaks


def _beam(events: list[_Rhythm]) -> None:
    """Beam contiguous flagged notes; rests and written quarters break groups.

    Each level (8th, 16th, 32nd) joins the neighbours that carry it, and a note
    alone at its level gets a hook. Inner beams also break between tuplets, and
    at the 8ths `_eighth_breaks` picks.
    """
    depth = [_BEAMS.get(event.note_type, 0) if event.notes else 0 for event in events]
    eighths = _eighth_breaks(events, depth)
    for level in (1, 2, 3):
        runs: list[list[int]] = []
        for i, event in enumerate(events):
            if depth[i] < level:
                continue
            split = level > 1 and (_tuplet_edge(events, i) or i in eighths)
            if runs and runs[-1][-1] == i - 1 and not split:
                runs[-1].append(i)
            else:
                runs.append([i])
        for run in runs:
            for i in run:
                outer = events[i].beams.get(1)
                if len(run) > 1:
                    events[i].beams[level] = "begin" if i == run[0] else "end" if i == run[-1] else "continue"
                elif outer:  # a hook points into the beam, and inside it toward the rest of its 8th or 16th
                    forward = outer == "begin" or (outer == "continue" and events[i].tick % (24 >> level - 1) == 0)
                    events[i].beams[level] = "forward hook" if forward else "backward hook"


def _spell(tick: int, end: int, values: tuple[tuple[int, int], ...]) -> Iterator[tuple[int, int]]:
    """Cut [tick, end) into the longest values that may start where they fall."""
    while tick < end:
        duration = next(d for d, step in values if tick % step == 0 and tick + d <= end)
        yield tick, duration
        tick += duration


def _straight(start: int, end: int, onsets: dict[int, list[Note]]) -> list[_Rhythm]:
    """Each note takes one value, at most up to the next onset; rests fill the remainder."""
    events = []
    edges = sorted({start, end, *(t for t in onsets if start <= t < end)})
    for tick, stop in zip(edges, edges[1:]):
        if tick in onsets:
            duration = next(_spell(tick, stop, _NOTES))[1]
            events.append(_Rhythm(tick, duration, onsets[tick]))
            tick += duration
        events.extend(_Rhythm(t, d) for t, d in _spell(tick, stop, _RESTS))
    return events


def _tuplet(
    start: int, end: int, onsets: dict[int, list[Note]], unit: str, ratio: tuple[int, int] = (3, 2),
) -> list[_Rhythm]:
    """One tuplet over [start, end): every onset (or the leading rest) lasts until the next.

    The number stands alone when every event is a flagged note, since their beam
    then spans exactly the tuplet; anything else keeps the bracket.
    """
    edges = sorted({start, end, *(t for t in onsets if start <= t < end)})
    events = [_Rhythm(tick, stop - tick, onsets.get(tick, []), unit, ratio=ratio) for tick, stop in zip(edges, edges[1:])]
    events[0].tuplets.append("start")
    events[-1].tuplets.append("stop")
    events[0].bracket = len(events) == 1 or not all(e.notes and e.note_type in _BEAMS for e in events)
    return events


def _sextuplet(start: int, end: int, onsets: dict[int, list[Note]]) -> list[_Rhythm] | None:
    """A beat of two 16th-triplet halves as one 6:4 group, when that stays simple.

    The second half must begin with a note and nothing may last longer than a
    tuplet 8th; otherwise the two halves read better as separate triplets.
    """
    middle = (start + end) // 2
    if middle not in onsets:
        return None
    events = _tuplet(start, end, onsets, "16th", (6, 4))
    return events if all(event.duration in (4, 8) for event in events) else None


def _rhythm(bar: Bar, notes: list[Note]) -> list[_Rhythm]:
    if not notes:
        return [_Rhythm(0, bar.ticks, measure_rest=True)]
    onsets: dict[int, list[Note]] = defaultdict(list)
    order = {inst: i for i, inst in enumerate(INSTRUMENTS)}
    for note in sorted(notes, key=lambda n: (n.tick, order[n.inst])):
        onsets[note.tick].append(note)
    if bar.denominator != 4:
        return _grouped_rhythm(bar, onsets)
    result: list[_Rhythm] = []
    beat = 0
    while beat < bar.beats:
        start, end = beat * DIVISIONS, (beat + 1) * DIVISIONS
        inside = [t - start for t in onsets if start <= t < end]
        if not inside:
            merge = bar.beats == 4 and beat in (0, 2) and not any(end <= t < end + DIVISIONS for t in onsets)
            result.append(_Rhythm(start, 2 * DIVISIONS if merge else DIVISIONS))
            beat += 2 if merge else 1
            continue
        # The ticks show the grid: 32nds are multiples of 3, triplet 8ths of 8, sextuplets of 4.
        if all(t % 3 == 0 for t in inside):
            events = _straight(start, end, onsets)
        elif all(t % 8 == 0 for t in inside):
            events = _tuplet(start, end, onsets, "eighth")
        else:  # Each half may use a different rhythmic family.
            halves = ((start, start + DIVISIONS // 2), (start + DIVISIONS // 2, end))
            straight = []
            for a, b in halves:
                local = [t - a for t in onsets if a <= t < b]
                straight.append(all(t % 3 == 0 for t in local))
                if not straight[-1] and any(t % 4 for t in local):
                    raise ValueError(f"Unsupported simultaneous grids at tick {a}: {local}")
            events = None if any(straight) else _sextuplet(start, end, onsets)
            if events is None:
                events = []
                for (a, b), plain in zip(halves, straight):
                    events += _straight(a, b, onsets) if plain else _tuplet(a, b, onsets, "16th")
        _beam(events)
        result.extend(events)
        beat += 1
    return result


def _grouped_rhythm(bar: Bar, onsets: dict[int, list[Note]]) -> list[_Rhythm]:
    """Beam the signature's groups, including the long beat in 2+2+3."""
    from .metre import Signature

    result = []
    start = 0
    for size in bar.beat_groups or Signature(bar.beats, bar.denominator).groups:
        end = start + size * bar.unit_ticks
        if all(t % 3 == 0 for t in onsets if start <= t < end):
            events = _straight(start, end, onsets)
        else:
            events = []
            for a in range(start, end, bar.unit_ticks):
                b = a + bar.unit_ticks
                local = [t - a for t in onsets if a <= t < b]
                events += (_straight(a, b, onsets) if all(t % 3 == 0 for t in local)
                           else _tuplet(a, b, onsets, "16th"))
        _beam(events)
        result.extend(events)
        start = end
    return result


def _write_voice(measure: ET.Element, bar: Bar, notes: list[Note], voice: int) -> None:
    for event in _rhythm(bar, notes):
        for i, drum in enumerate(event.notes or [None]):
            note = _element(measure, "note")
            if i:
                _element(note, "chord")
            if drum is None:
                _element(note, "rest", **({"measure": "yes"} if event.measure_rest else {}))
            else:
                unpitched = _element(note, "unpitched")
                _, step, octave, _, _ = KIT[drum.inst]
                _element(unpitched, "display-step", step)
                _element(unpitched, "display-octave", octave)
            _element(note, "duration", event.duration)
            if drum is not None:
                _element(note, "instrument", id=f"P1-{drum.inst}")
            _element(note, "voice", voice)
            if not event.measure_rest:
                _element(note, "type", event.note_type)
                if event.duration in (9, 18, 36):
                    _element(note, "dot")
            if event.tuplet:
                modification = _element(note, "time-modification")
                _element(modification, "actual-notes", event.ratio[0])
                _element(modification, "normal-notes", event.ratio[1])
                _element(modification, "normal-type", event.tuplet)
            if drum is not None:
                _element(note, "stem", "up" if voice == 1 else "down")
                _element(note, "notehead", KIT[drum.inst][3], **({"parentheses": "yes"} if drum.ghost else {}))
            if i == 0:
                for number, state in event.beams.items():
                    _element(note, "beam", state, number=str(number))
                if event.tuplets:
                    notations = _element(note, "notations")
                    for state in event.tuplets:
                        shown = {"bracket": "yes" if event.bracket else "no"} if state == "start" else {}
                        _element(notations, "tuplet", type=state, number="1", **shown)


def _tempo_marks(tempos: list[float], song_tempo: float) -> dict[int, int]:
    """Bar index -> BPM to print.

    Bar-to-bar differences are tracker jitter and drift. Only a tempo the song
    holds is printed: a stretch of at least eight bars that stay within 8% of
    their median, when that median is more than 12% from the tempo in force.
    """
    held: list[tuple[int, float]] = []  # (first bar, tempo) of each steady stretch
    start = 0
    while start + 8 <= len(tempos):
        end = start + 8
        if any(abs(tempo / median(tempos[start:end]) - 1) > 0.08 for tempo in tempos[start:end]):
            start += 1
            continue
        while end < len(tempos) and abs(tempos[end] / median(tempos[start:end]) - 1) <= 0.08:
            end += 1
        held.append((start, median(tempos[start:end])))
        start = end
    current = held[0][1] if held else song_tempo
    marks = {0: round(current)}
    for start, tempo in held[1:]:
        if abs(tempo / current - 1) > 0.12:
            marks[start] = round(tempo)
            current = tempo
    return marks


def to_musicxml(score: Score, *, voices: int = 1) -> str:
    """Return a complete percussion score, with one or two rhythmic voices."""
    if voices not in (1, 2):
        raise ValueError("voices must be 1 or 2")
    root = ET.Element("score-partwise", version="4.0")
    # MuseScore prints movement-title as the title, so the song name goes in both.
    _element(_element(root, "work"), "work-title", score.title)
    _element(root, "movement-title", score.title)
    if score.subtitle:
        _element(_element(root, "identification"), "creator", score.subtitle, type="composer")
    score_part = _element(_element(root, "part-list"), "score-part", id="P1")
    _element(score_part, "part-name", "Drumset")
    for inst in INSTRUMENTS:
        instrument = _element(score_part, "score-instrument", id=f"P1-{inst}")
        _element(instrument, "instrument-name", inst.capitalize())
    for inst in INSTRUMENTS:
        instrument = _element(score_part, "midi-instrument", id=f"P1-{inst}")
        _element(instrument, "midi-channel", 10)
        _element(instrument, "midi-program", 1)
        _element(instrument, "midi-unpitched", KIT[inst][4] + 1)
    part = _element(root, "part", id="P1")
    previous_meter = None
    tempo_marks = _tempo_marks([bar.tempo for bar in score.bars], score.tempo)
    for index, bar in enumerate(score.bars):
        measure = _element(part, "measure", number=str(index + 1))
        if index == 0 or (bar.beats, bar.denominator) != previous_meter:
            attributes = _element(measure, "attributes")
            if index == 0:
                _element(attributes, "divisions", DIVISIONS)
            time = _element(attributes, "time")
            _element(time, "beats", bar.beats)
            _element(time, "beat-type", bar.denominator)
            if index == 0:
                clef = _element(attributes, "clef")
                _element(clef, "sign", "percussion")
                _element(clef, "line", 2)
            previous_meter = (bar.beats, bar.denominator)
        if index in tempo_marks:
            direction = _element(measure, "direction", placement="above")
            metronome = _element(_element(direction, "direction-type"), "metronome")
            _element(metronome, "beat-unit", "quarter")
            compound = bar.denominator == 8 and bar.beats >= 6 and bar.beats % 3 == 0
            if compound:
                _element(metronome, "beat-unit-dot")
            _element(metronome, "per-minute", round(tempo_marks[index] * 2 / 3) if compound else tempo_marks[index])
            _element(direction, "sound", tempo=str(tempo_marks[index]))
        feet = [n for n in bar.notes if n.inst in _FEET] if voices == 2 else []
        _write_voice(measure, bar, [n for n in bar.notes if n not in feet], 1)
        if feet:
            _element(_element(measure, "backup"), "duration", bar.ticks)
            _write_voice(measure, bar, feet, 2)
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + _DOCTYPE + "\n" + ET.tostring(root, encoding="unicode") + "\n"
