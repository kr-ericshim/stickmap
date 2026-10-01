"""Shared data model: detected hits in seconds -> quantized score in musical time."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Ticks per quarter note. 24 divides evenly into 32nds (3), 16ths (6),
# sextuplets (4) and triplet 8ths (8).
DIVISIONS = 24

# Everything the score can show: name -> (family, staff step, octave, note head, General MIDI note).
# The detector tells five families apart; the other names are what it, or the player, was doing
# within a family. "tom" is the mid tom and any tom whose size is not known; "cymbal" is a crash
# and any cymbal that is not known to be the ride.
KIT = {
    "kick": ("kick", "F", 4, "normal", 36),
    "snare": ("snare", "C", 5, "normal", 38),
    "sidestick": ("snare", "C", 5, "x", 37),
    "hihat": ("hihat", "G", 5, "x", 42),
    "hihat_open": ("hihat", "G", 5, "circle-x", 46),
    "hihat_pedal": ("hihat", "D", 4, "x", 44),
    "tom_high": ("tom", "E", 5, "normal", 50),
    "tom": ("tom", "D", 5, "normal", 47),
    "tom_low": ("tom", "A", 4, "normal", 43),
    "ride": ("cymbal", "F", 5, "x", 51),
    "ride_bell": ("cymbal", "F", 5, "diamond", 53),
    "cymbal": ("cymbal", "A", 5, "x", 49),
}
INSTRUMENTS = tuple(KIT)


def family(inst: str) -> str:
    """The drum an articulation belongs to: kick, snare, hihat, tom or cymbal."""
    return KIT[inst][0]


@dataclass(frozen=True)
class Hit:
    """One detected drum onset, before quantization."""

    time: float  # seconds from start of audio
    inst: str  # one of INSTRUMENTS
    strength: float  # peak activation, 0..1; a snare's loudness (1 = the song's typical snare) when measured
    ghost: bool = False  # a ghost note: played far softer than the song's typical hit


@dataclass
class Note:
    """One quantized drum note inside a bar."""

    tick: int  # offset from bar start, in DIVISIONS per quarter
    inst: str  # one of INSTRUMENTS
    velocity: int  # MIDI velocity 1..127
    ghost: bool = False


@dataclass
class Bar:
    index: int  # 0-based
    beats: int  # numerator; beat_times and subdivisions use denominator-note units
    beat_times: list[float]  # seconds; len == beats + 1, last entry is the bar end
    subdivisions: list[int | list[int]]  # per denominator note; /4 may use [first-half grid, second-half grid]
    tempo: float  # quarter-note BPM over this bar
    notes: list[Note] = field(default_factory=list)

    denominator: int = 4
    beat_groups: list[int] = field(default_factory=list)  # denominator notes per beaming group

    @property
    def unit_ticks(self) -> int:
        return DIVISIONS * 4 // self.denominator

    @property
    def start(self) -> float:
        return self.beat_times[0]

    @property
    def ticks(self) -> int:
        return self.beats * self.unit_ticks

    def tick_time(self, tick: int) -> float:
        """Seconds at which `tick` falls, interpolating inside its beat."""
        beat, rem = divmod(tick, self.unit_ticks)
        if beat >= self.beats:
            return self.beat_times[-1]
        t0, t1 = self.beat_times[beat], self.beat_times[beat + 1]
        return t0 + (t1 - t0) * rem / self.unit_ticks


@dataclass
class Score:
    title: str
    bars: list[Bar]
    grid: int = 16  # 8 or 16: finest straight note value used
    tempo: float = 120.0  # median BPM, used for the metronome mark
    subtitle: str = ""
    divisions: int = DIVISIONS  # ticks per quarter the note ticks are counted in
    merged_strokes: int = 0  # in-range strokes sharing a slot, excluding reading-mode drops

    pulse: dict | None = None  # explicit-signature pulse diagnostics; absent in legacy scores
    time_signature_suggestion: dict | None = None

    def to_json(self) -> str:
        raw = asdict(self)
        # Keep the automatic path byte-compatible with existing saved scores.
        if self.pulse is None:
            raw.pop("pulse")
        if self.time_signature_suggestion is None:
            raw.pop("time_signature_suggestion")
        for bar in raw["bars"]:
            if bar["denominator"] == 4 and not bar["beat_groups"]:
                bar.pop("denominator")
                bar.pop("beat_groups")
        return json.dumps(raw, ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "Score":
        raw = json.loads(text)
        # Files written before `divisions` existed count 12 ticks per quarter.
        scale = DIVISIONS // raw.pop("divisions", 12)
        bars = [
            Bar(**{**b, "notes": [Note(**{**n, "tick": n["tick"] * scale}) for n in b["notes"]]})
            for b in raw.pop("bars")
        ]
        return cls(bars=bars, **raw)

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Score":
        return cls.from_json(path.read_text(encoding="utf-8"))
