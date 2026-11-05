"""Export musical score positions to a Standard MIDI File."""

from __future__ import annotations

from pathlib import Path

import mido

from .score import DIVISIONS, KIT, Score

GM_NOTES = {inst: note for inst, (_, _, _, _, note) in KIT.items()}
PPQ = 480


def to_midi(score: Score, path: Path) -> None:
    """Write tempo/meter metadata and a channel-10 drum track at 480 PPQ."""
    midi = mido.MidiFile(type=1, ticks_per_beat=PPQ)
    conductor = mido.MidiTrack()
    drums = mido.MidiTrack()
    midi.tracks.extend([conductor, drums])
    # MIDI text is latin-1: drop what it cannot hold (Hangul, curly quotes, ...).
    title = score.title.encode("latin-1", "ignore").decode("latin-1").strip()
    conductor.append(mido.MetaMessage("track_name", name=title or "Tempo"))
    drums.append(mido.MetaMessage("track_name", name="Drumset"))
    position = previous_meta = 0
    tempo = meter = None
    events: list[tuple[int, int, mido.Message]] = []
    for bar in score.bars:
        messages = []
        if not (bar.beat_groups or bar.denominator != 4) and (tempo is None or abs(bar.tempo - tempo) > 0.5):
            messages.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bar.tempo)))
            tempo = bar.tempo
        if meter != (bar.beats, bar.denominator):
            messages.append(mido.MetaMessage("time_signature", numerator=bar.beats, denominator=bar.denominator,
                                             clocks_per_click=36 if bar.denominator == 8 and bar.beats >= 6 and bar.beats % 3 == 0 else 24))
            meter = bar.beats, bar.denominator
        for message in messages:
            message.time = position - previous_meta
            conductor.append(message)
            previous_meta = position
        if bar.beat_groups or bar.denominator != 4:
            # Explicit scores retain the local pulse timing, including drift
            # inside a bar. MIDI tempo is always microseconds per quarter.
            for i, (a, b) in enumerate(zip(bar.beat_times, bar.beat_times[1:])):
                at = position + i * bar.unit_ticks * (PPQ // DIVISIONS)
                microseconds = round((b - a) * 1_000_000 * DIVISIONS / bar.unit_ticks)
                conductor.append(mido.MetaMessage("set_tempo", tempo=microseconds, time=at - previous_meta))
                previous_meta = at
        for note in bar.notes:
            start = position + note.tick * (PPQ // DIVISIONS)
            events.append((start, 1, mido.Message("note_on", channel=9, note=GM_NOTES[note.inst], velocity=note.velocity)))
            events.append((min(start + PPQ // 8, position + bar.ticks * (PPQ // DIVISIONS)), 0, mido.Message("note_off", channel=9, note=GM_NOTES[note.inst], velocity=0)))
        position += bar.ticks * (PPQ // DIVISIONS)
    conductor.append(mido.MetaMessage("end_of_track", time=position - previous_meta))
    previous = 0
    for tick, _, message in sorted(events, key=lambda event: (event[0], event[1], event[2].note)):
        message.time = tick - previous
        drums.append(message)
        previous = tick
    drums.append(mido.MetaMessage("end_of_track", time=position - previous))
    midi.save(path)
