import numpy as np
import soundfile as sf

from stickmap.score import Hit
from stickmap.toms import HIGH, LOW, MID, _groups, _sizes, tom_sizes

SR = 44100


def kit(path, strokes, seconds=12.0):
    """A drum stem of decaying sines: (time, Hz) per stroke."""
    audio = np.zeros(int(seconds * SR), dtype=np.float32)
    t = np.arange(int(0.3 * SR)) / SR
    for time, hz in strokes:
        start = int(time * SR)
        audio[start:start + len(t)] += (np.sin(2 * np.pi * hz * t) * np.exp(-t * 12)).astype(np.float32)
    sf.write(path, np.stack([audio, audio], axis=1), SR)
    return path


def test_toms_are_named_by_their_place_in_the_kit(tmp_path):
    drums = [(80, LOW), (98, LOW), (135, MID), (172, HIGH), (215, HIGH)]
    strokes = [(0.5 + 0.45 * i, drums[i % 5][0] * (1 + 0.02 * ((i * 7) % 3 - 1))) for i in range(25)]
    hits = [Hit(time, "tom", 0.8) for time, _ in strokes] + [Hit(0.2, "snare", 0.9)]
    sized = tom_sizes(kit(tmp_path / "drums.wav", strokes), hits)
    assert [hit.inst for hit in sized[:25]] == [drums[i % 5][1] for i in range(25)]
    assert sized[25] == hits[25] and [hit.time for hit in sized] == [hit.time for hit in hits]


def test_kit_layouts():
    assert _sizes(1) == [LOW]
    assert _sizes(2) == [LOW, HIGH]  # one rack tom is written as the high tom
    assert _sizes(3) == [LOW, MID, HIGH]
    assert _sizes(4) == [LOW, LOW, MID, HIGH]
    assert _sizes(5) == [LOW, LOW, MID, HIGH, HIGH]


def spread(*drums):
    """Ten hits per (Hz, count) drum, wobbling a little."""
    return [hz * 2 ** (((i * 7) % 5 - 2) * 0.1 / 12) for hz, count in drums for i in range(count)]


def test_groups_go_by_where_the_toms_lie_against_each_other_not_in_hertz():
    # A gospel kit: low, close toms, all of them under 110 Hz.
    assert len(_groups(spread((60, 10), (85, 10), (107, 10)))) == 3
    # The same kit an octave up is grouped the same way.
    low, high = _groups(spread((60, 10), (85, 10), (107, 10))), _groups(spread((120, 10), (170, 10), (214, 10)))
    assert np.allclose(np.array(high) / np.array(low), 2, rtol=0.02)
    # A floor tom whose pitch wobbles between two close peaks is one drum.
    assert len(_groups(spread((70, 12), (82, 4), (140, 10)))) == 2
    # A few hits far above the toms (a snare, a ringing mode) are not a rack tom.
    assert len(_groups(spread((70, 20), (100, 15), (130, 15), (230, 4)))) == 3
    # With only a few toms, every pitch counts.
    assert len(_groups([90.0, 170.0, 170.0])) == 2


def test_a_few_stray_toms_go_by_pitch_alone_and_a_kick_does_not_pass_for_a_floor_tom(tmp_path):
    strokes = [(1.0, 170.0), (2.0, 90.0), (3.0, 170.0), (3.0, 55.0)]
    hits = [Hit(1.0, "tom", 0.8), Hit(2.0, "tom", 0.8), Hit(3.0, "tom", 0.8), Hit(3.0, "kick", 0.9), Hit(11.95, "tom", 0.5)]
    sized = tom_sizes(kit(tmp_path / "drums.wav", strokes), hits)
    # The last hit is too close to the end of the recording to have a pitch: it stays a plain tom.
    assert [hit.inst for hit in sized] == [HIGH, LOW, HIGH, "kick", MID]


def bells(path, toms, pings):
    """Toms as low sines and cymbal bells as high ones."""
    return kit(path, [(time, 120.0) for time in toms] + [(time, 1100.0) for time in pings])


def test_a_tom_that_no_drum_head_rang_in_is_dropped(tmp_path):
    toms, pings = [1.0, 2.0, 3.0, 4.0], [5.0, 6.0]
    hits = [Hit(time, "tom", 0.8) for time in toms + pings]
    sized = tom_sizes(bells(tmp_path / "drums.wav", toms, pings), hits)
    assert [hit.time for hit in sized] == toms and all(hit.inst in (LOW, MID, HIGH) for hit in sized)


def test_a_song_full_of_them_is_played_on_a_kit_tuned_that_high(tmp_path):
    toms = [0.5 + 0.4 * i for i in range(6)]
    pings = [3.0 + 0.25 * i for i in range(30)]
    hits = [Hit(time, "tom", 0.8) for time in toms + pings]
    sized = tom_sizes(bells(tmp_path / "drums.wav", toms, pings), hits)
    assert [hit.time for hit in sized] == toms + pings
    assert all(hit.inst in (LOW, MID, HIGH) for hit in sized)


def test_a_floor_tom_at_the_bottom_of_the_band_is_not_read_by_its_ringing_mode(tmp_path):
    # A floor tom at 56 Hz rings a mode 1.6 times higher, at half the strength.
    audio = np.zeros(int(12.0 * SR), dtype=np.float32)
    t = np.arange(int(0.3 * SR)) / SR
    strokes = [(0.5 + 0.45 * i, (56, 70, 110)[i % 3]) for i in range(24)]
    for time, hz in strokes:
        ring = np.sin(2 * np.pi * hz * t) + (0.5 * np.sin(2 * np.pi * 1.6 * hz * t) if hz == 56 else 0)
        start = int(time * SR)
        audio[start:start + len(t)] += (ring * np.exp(-t * 12)).astype(np.float32)
    sf.write(tmp_path / "drums.wav", np.stack([audio, audio], axis=1), SR)
    sized = tom_sizes(tmp_path / "drums.wav", [Hit(time, "tom", 0.8) for time, _ in strokes])
    assert [hit.inst for hit in sized] == [(LOW, MID, HIGH)[i % 3] for i in range(24)]
