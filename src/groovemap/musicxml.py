"""Spell a quantized drum score as readable MusicXML 4.0."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from statistics import median
import xml.etree.ElementTree as ET

from .score import DIVISIONS, INSTRUMENTS, KIT, Bar, Note, Score

_FEET = ("kick", "hihat_pedal")  # the second voice, stems down
# A five-piece kit's names. MuseScore builds a drum part from its alphabetically
# first score-instrument, so none of these may equal one of its template names
# ("Bass Drum", "Ride Cymbal" and "Snare Drum" are templates).
_NAMES = {
    "kick": "Kick", "snare": "Snare", "sidestick": "Cross-stick", "hihat": "Closed Hi-Hat",
    "hihat_open": "Open Hi-Hat", "hihat_pedal": "Pedal Hi-Hat", "tom_high": "High Tom", "tom": "Mid Tom",
    "tom_low": "Floor Tom", "ride": "Ride", "ride_bell": "Ride Bell", "cymbal": "Crash",
}
_DOCTYPE = '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">'
# Written value per duration in ticks: 9 and 18 are dotted, 4, 8 and 16 only occur inside tuplets.
_TYPES = {3: "32nd", 4: "16th", 6: "16th", 8: "eighth", 9: "16th", 12: "eighth", 16: "quarter", 18: "eighth", 24: "quarter", 36: "quarter", 48: "half"}
_BEAMS = {"eighth": 1, "16th": 2, "32nd": 3}
# Isolated attacks and rests use binary alignment relative to the group origin.
# Longer syncopations are admitted only by complete cells below.
_NOTES = ((24, 24), (12, 12), (6, 6), (3, 3))
_RESTS = ((24, 24), (12, 12), (6, 6), (3, 3))


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
    """Beam contiguous flagged notes; longer rests and written quarters break groups.

    Each level (8th, 16th, 32nd) joins the neighbours that carry it, and a note
    alone at its level gets a hook. Inner beams also break between tuplets, and
    at the 8ths `_eighth_breaks` picks.
    """
    depth = [_BEAMS.get(event.note_type, 0) if event.notes else 0 for event in events]
    eighths = _eighth_breaks(events, depth)
    # The primary beam crosses one inner rest of at most a 16th, so the beat
    # still reads as one group; inner beams break there and hook instead.
    bridged = {i for i in range(1, len(events) - 1)
               if not events[i].notes and events[i].duration <= 6 and depth[i - 1] and depth[i + 1]
               and not _tuplet_edge(events, i) and not _tuplet_edge(events, i + 1)}
    for level in (1, 2, 3):
        runs: list[list[int]] = []
        for i, event in enumerate(events):
            if depth[i] < level and not (level == 1 and i in bridged):
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


def _spell(tick: int, end: int, values: tuple[tuple[int, int], ...], *, origin: int = 0) -> Iterator[tuple[int, int]]:
    """Cut [tick, end) into the longest values that may start where they fall."""
    while tick < end:
        duration = next(d for d, step in values if (tick - origin) % step == 0 and tick + d <= end)
        yield tick, duration
        tick += duration


def _straight(start: int, end: int, onsets: dict[int, list[Note]], *, exact_gaps: bool = False) -> list[_Rhythm]:
    """Spell isolated hits metrically, retaining complete syncopated cells."""
    events = []
    edges = sorted({start, end, *(t for t in onsets if start <= t < end)})
    inside = {t - start for t in onsets if start <= t < end}
    longer = {}
    # Match the entire cell: extra attacks may subdivide a short component,
    # but cannot hide inside a long one. Eighth cells also work in mixed beats.
    for width in (12, 24):
        unit = width // 4
        shapes = ((0, 3), (0, 1), (0, 1, 3), (1, 3))
        for base in range(0, end - start - width + 1, 12):
            local = {t - base for t in inside if base <= t < base + width}
            for shape in shapes:
                attacks = [i * unit for i in shape]
                spans = list(zip(attacks, attacks[1:] + [width]))
                if not set(attacks) <= local:
                    continue
                if any(t not in attacks and not any(a < t < b and b - a == unit for a, b in spans)
                       for t in local):
                    continue
                for a, b in spans:
                    if b - a > unit:
                        longer[start + base + a] = b - a
    if end - start == 36:
        compound = {
            frozenset({0}): {0: 36},
            frozenset({0, 24}): {0: 24, 24: 12},
            frozenset({0, 12}): {0: 12, 12: 24},
            frozenset({0, 18}): {0: 18, 18: 18},
        }
        longer.update({start + t: d for t, d in compound.get(frozenset(inside), {}).items()})
        if not inside:
            return [_Rhythm(start, 36)]
    for tick, stop in zip(edges, edges[1:]):
        if tick in onsets:
            duration = longer.get(tick, next(_spell(tick, stop, _NOTES, origin=start))[1])
            if duration < 6 and stop - tick == 6 and stop in onsets:
                # A displaced run of 16ths stays 16ths: r32 16 16 16 32, not split cells.
                duration = 6
            if exact_gaps and (tick - start) % 6 == 3 and stop - tick in (12, 18):
                # Opt-in alternative; _duration_choice still requires that
                # none of its fragmentation counters get worse.
                duration = stop - tick
            events.append(_Rhythm(tick, duration, onsets[tick]))
            tick += duration
        events.extend(_Rhythm(t, d) for t, d in _spell(tick, stop, _RESTS, origin=start))
    # Coalesce adjoining straight rests using their metrical alignment. A note
    # or tuplet never disappears, and the caller's beat edge remains visible.
    merged = []
    for event in events:
        if merged and not merged[-1].notes and not event.notes:
            previous = merged[-1]
            duration = previous.duration + event.duration
            if any(duration == d and (previous.tick - start) % step == 0 for d, step in _RESTS):
                previous.duration = duration
                continue
        merged.append(event)
    return merged


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


def _rhythm(bar: Bar, notes: list[Note], *, exact_gaps: bool = False) -> list[_Rhythm]:
    if not notes:
        return [_Rhythm(0, bar.ticks, measure_rest=True)]
    onsets: dict[int, list[Note]] = defaultdict(list)
    order = {inst: i for i, inst in enumerate(INSTRUMENTS)}
    for note in sorted(notes, key=lambda n: (n.tick, order[n.inst])):
        onsets[note.tick].append(note)
    if bar.denominator != 4:
        return _grouped_rhythm(bar, onsets, exact_gaps=exact_gaps)
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
            events = _straight(start, end, onsets, exact_gaps=exact_gaps)
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
                    events += _straight(a, b, onsets, exact_gaps=exact_gaps) if plain else _tuplet(a, b, onsets, "16th")
        _beam(events)
        result.extend(events)
        beat += 1
    return result


def _grouped_rhythm(bar: Bar, onsets: dict[int, list[Note]], *, exact_gaps: bool = False) -> list[_Rhythm]:
    """Beam the signature's groups, including the long beat in 2+2+3."""
    from .metre import Signature

    result = []
    start = 0
    for size in bar.beat_groups or Signature(bar.beats, bar.denominator).groups:
        end = start + size * bar.unit_ticks
        if all(t % 3 == 0 for t in onsets if start <= t < end):
            events = _straight(start, end, onsets, exact_gaps=exact_gaps)
        else:
            events = []
            for a in range(start, end, bar.unit_ticks):
                b = a + bar.unit_ticks
                local = [t - a for t in onsets if a <= t < b]
                events += (_straight(a, b, onsets, exact_gaps=exact_gaps) if all(t % 3 == 0 for t in local)
                           else _tuplet(a, b, onsets, "16th"))
        _beam(events)
        result.extend(events)
        start = end
    return result


def _duration_choice(bar: Bar, notes: list[Note]) -> list[_Rhythm]:
    """Use an exact-gap spelling only if no fragmentation counter gets worse.

    Kept outside _rhythm's default path so engraving improvements do not
    silently change the quantizer's existing grid-choice objective.
    """
    original = _rhythm(bar, notes)
    alternative = _rhythm(bar, notes, exact_gaps=True)

    def counts(events: list[_Rhythm]) -> tuple[int, int, int]:
        return (sum(not e.notes and e.duration < 6 for e in events),
                sum(bool(e.notes) and e.note_type in _BEAMS and 1 not in e.beams for e in events),
                sum(e.duration in (9, 18, 36) for e in events))

    before, after = counts(original), counts(alternative)
    return alternative if after != before and all(a <= b for a, b in zip(after, before)) else original


def _continuous_tuplets(
    bars: list[Bar], rhythms: list[list[_Rhythm]],
) -> list[list[_Rhythm]]:
    """Choose equivalent tuplet groups across beats, including bar boundaries.

    Only existing all-tuplet beats in a selected sextuplet run are eligible.
    Every onset, duration and rest stays fixed; mixed straight/ternary halves
    can never be wrapped into one ratio. The original spelling remains a
    candidate, so continuity wins over blindly replacing every 3 with a 6.
    """
    context = {(index, i) for index, b in enumerate(bars) if b.denominator == 4
               for i, mode in enumerate(b.subdivisions) if mode == 6}
    neighbours = set()
    units = [(index, i) for index, b in enumerate(bars) for i in range(b.beats)]
    for a, b in zip(units, units[1:]):
        if a in context and b in context:
            neighbours.update((a, b))
    blocks = []
    for index, (bar, events) in enumerate(zip(bars, rhythms, strict=True)):
        for event in events:
            key = (index, event.tick // DIVISIONS)
            if not blocks or blocks[-1][0] != key:
                blocks.append((key, []))
            blocks[-1][1].append(event)

    # State is the last printed ratio. Cost first minimizes switches (also
    # across measures), then group starts, then changes to the old spelling.
    states = {None: ((0, 0, 0), [])}
    for key, current in blocks:
        candidates = [current]
        start = key[1] * DIVISIONS
        if (key in neighbours and current[0].tick == start
                and current[-1].tick + current[-1].duration == start + DIVISIONS
                and all(e.tuplet and e.duration in (4, 8, 16) for e in current)
                and any(e.ratio != (6, 4) for e in current)):
            joined = [replace(e, tuplet="16th", ratio=(6, 4), tuplets=[], beams={}) for e in current]
            joined[0].tuplets.append("start")
            joined[-1].tuplets.append("stop")
            joined[0].bracket = not all(e.notes and e.note_type in _BEAMS for e in joined)
            _beam(joined)
            candidates.append(joined)
        nxt = {}
        for changed, candidate in enumerate(candidates):
            families = [e.ratio[0] if e.tuplet else 0 for e in candidate]
            switches = sum(a != b for a, b in zip(families, families[1:]))
            groups = sum("start" in e.tuplets for e in candidate)
            for previous, (cost, path) in states.items():
                value = (cost[0] + switches + int(previous is not None and previous != families[0]),
                         cost[1] + groups, cost[2] + changed)
                if families[-1] not in nxt or value < nxt[families[-1]][0]:
                    nxt[families[-1]] = (value, path + [(key[0], candidate)])
        states = nxt
    result = [[] for _ in bars]
    if states:
        for index, events in min(states.values(), key=lambda v: v[0])[1]:
            result[index].extend(events)
    return result


def _write_voice(measure: ET.Element, events: list[_Rhythm], voice: int) -> None:
    for event in events:
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
                        shown = {"bracket": "yes" if event.bracket else "no", "placement": "above", "show-number": "actual"} if state == "start" else {}
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
        _element(instrument, "instrument-name", _NAMES[inst])
        # The kit's sound on every piece, as MuseScore exports it: without it MuseScore imports a piano.
        _element(instrument, "instrument-sound", "drum.group.set")
    for inst in INSTRUMENTS:
        instrument = _element(score_part, "midi-instrument", id=f"P1-{inst}")
        _element(instrument, "midi-channel", 10)
        _element(instrument, "midi-program", 1)
        _element(instrument, "midi-unpitched", KIT[inst][4] + 1)
    part = _element(root, "part", id="P1")
    feet = [[n for n in bar.notes if n.inst in _FEET] if voices == 2 else [] for bar in score.bars]
    hands = [[n for n in bar.notes if n not in lower] for bar, lower in zip(score.bars, feet)]
    upper_rhythm = _continuous_tuplets(score.bars, [_duration_choice(b, n) for b, n in zip(score.bars, hands)])
    lower_rhythm = _continuous_tuplets(score.bars, [_duration_choice(b, n) if n else [] for b, n in zip(score.bars, feet)])
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
        _write_voice(measure, upper_rhythm[index], 1)
        if feet[index]:
            _element(_element(measure, "backup"), "duration", bar.ticks)
            _write_voice(measure, lower_rhythm[index], 2)
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + _DOCTYPE + "\n" + ET.tostring(root, encoding="unicode") + "\n"
