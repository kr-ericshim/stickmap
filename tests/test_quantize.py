import json
import random
from pathlib import Path

import numpy as np
import pytest

from stickmap.quantize import quantize
from stickmap.score import DIVISIONS, INSTRUMENTS, Hit, Score


def steady(beats=16, interval=0.5):
    return [i * interval for i in range(beats)]


def positions(score):
    return [(bar.index, note.tick, note.inst) for bar in score.bars for note in bar.notes]


def grids(score):
    return [subdivision for bar in score.bars for subdivision in bar.subdivisions]


def ticks(score, inst):
    return [(bar.index, note.tick) for bar in score.bars for note in bar.notes if note.inst == inst]


def pattern(beats, fractions, inst, interval=0.5, late=0.0):
    return [Hit(beat * interval + fraction * interval + late, inst, 0.8) for beat in beats for fraction in fractions]


def test_straight_grid_jitter_and_global_offset():
    beats = steady()
    hits = [Hit(beat * 0.5 + slot * 0.125 + 0.02 + (-0.015 if (beat + slot) % 2 == 0 else 0.015), "hihat", 0.8)
            for beat in range(16) for slot in range(4)]
    score = quantize(hits, beats, beats[::4], title="Groove", subtitle="Test")
    assert positions(score) == [(beat // 4, (beat % 4) * 24 + slot * 6, "hihat") for beat in range(16) for slot in range(4)]
    assert all(bar.subdivisions == [4] * 4 for bar in score.bars)
    assert score.tempo == 120
    assert (score.title, score.subtitle) == ("Groove", "Test")


@pytest.mark.parametrize("grid", [8, 16])
@pytest.mark.parametrize("offset", [0, 0.02])
def test_shuffle_uses_triplets(grid, offset):
    beats = steady(8)
    hits = [Hit(beat * 0.5 + slot / 3 * 0.5 + offset, "hihat", 1) for beat in range(8) for slot in (0, 2)]
    score = quantize(hits, beats, beats[::4], grid=grid)
    assert all(bar.subdivisions == [3] * 4 for bar in score.bars)
    assert positions(score) == [(beat // 4, (beat % 4) * 24 + slot, "hihat") for beat in range(8) for slot in (0, 16)]


def test_triplets_off_and_conservative_auto():
    beats = steady(4)
    straight = [Hit(beat * 0.5 + slot * 0.125, "snare", 1) for beat in range(4) for slot in (1, 2, 3)]
    assert quantize(straight, beats, [0]).bars[0].subdivisions == [4] * 4
    shuffle = [Hit(beat * 0.5 + slot * 0.5, "hihat", 1) for beat in range(4) for slot in (0, 2 / 3)]
    assert quantize(shuffle, beats, [0], triplets="off").bars[0].subdivisions == [4] * 4
    # One stray triplet-like hit cannot outweigh several clear straight hits.
    mixed = straight + [Hit(1 / 3, "tom", 1)]
    assert quantize(mixed, beats, [0]).bars[0].subdivisions[0] == 4


def test_beginner_drops_odd_sixteenths_without_moving():
    beats = steady(4)
    hits = [Hit(beat * 0.5 + slot * 0.125, "hihat", 1) for beat in range(4) for slot in range(4)]
    score = quantize(hits, beats, [0], grid=8)
    assert [n.tick for n in score.bars[0].notes] == list(range(0, 96, 12))
    assert score.bars[0].subdivisions == [2] * 4


def test_irregular_bar_and_nearest_downbeat_matching():
    # A real odd bar shifts every later downbeat, so the evidence accumulates.
    # It needs enough bars on both sides: next to a short run, shifting that run is cheaper.
    starts = [4 * i for i in range(9)] + [35 + 4 * i for i in range(9)]
    beats = steady(starts[-1] + 4)
    downbeats = [beats[i] + 0.01 for i in starts]
    score = quantize([], beats, downbeats, odd_bars="auto")
    assert [bar.beats for bar in score.bars] == [4] * 8 + [3] + [4] * 9
    assert {bar.beats for bar in quantize([], beats, downbeats).bars} == {4}  # default: regular bars
    assert all(len(bar.beat_times) == bar.beats + 1 for bar in score.bars)
    assert all(a.beat_times[-1] == b.start for a, b in zip(score.bars, score.bars[1:]))


def test_meter_override_uses_most_common_phase():
    beats = steady(19)
    score = quantize([], beats, [beats[i] for i in (1, 5, 8, 13, 17)], meter=4)
    assert all(bar.beats == 4 for bar in score.bars)
    assert [bar.start for bar in score.bars] == [-1.5, 0.5, 2.5, 4.5, 6.5, 8.5]


def test_pickup_and_tail_extrapolation_and_outside_hits():
    beats = steady(13)
    hits = [Hit(0.0, "kick", 1), Hit(6.0, "snare", 1), Hit(-2, "tom", 1), Hit(9, "cymbal", 1)]
    score = quantize(hits, beats, [beats[i] for i in (2, 6, 10)])
    assert [bar.start for bar in score.bars] == [-1, 1, 3, 5]
    assert score.bars[-1].beat_times == [5, 5.5, 6, 6.5, 7]
    assert positions(score) == [(0, 48, "kick"), (3, 48, "snare")]


def test_large_downbeat_gap_is_split():
    beats = steady(28)
    score = quantize([], beats, [beats[i] for i in (0, 4, 8, 24)])
    assert [bar.beats for bar in score.bars] == [4] * 7
    assert all(1 <= bar.beats <= 12 for bar in score.bars)


@pytest.mark.parametrize("meter", [None, 3])
def test_without_downbeats(meter):
    score = quantize([], steady(10), [], meter=meter)
    assert [bar.beats for bar in score.bars] == [meter or 4] * (4 if meter else 3)


def test_tempo_drift_snaps_using_local_intervals():
    beats = np.concatenate(([0], np.cumsum(np.linspace(0.6, 0.4, 15)))).tolist()
    intervals = np.diff(beats).tolist() + [beats[-1] - beats[-2]]
    hits = [Hit(time + intervals[i] * slot / 4 + 0.02, "snare", 0.5) for i, time in enumerate(beats) for slot in (0, 1, 3)]
    score = quantize(hits, beats, beats[::4])
    assert positions(score) == [(i // 4, (i % 4) * 24 + slot * 6, "snare") for i in range(16) for slot in (0, 1, 3)]
    assert score.bars[0].tempo < score.bars[-1].tempo
    for bar in score.bars:
        assert bar.tempo == pytest.approx(60 * bar.beats / (bar.beat_times[-1] - bar.start))
    assert score.tempo == round(float(np.median([bar.tempo for bar in score.bars])), 1)


def test_beat_end_carries_to_next_bar_and_deduplicates():
    beats = steady(8)
    hits = [Hit(0, "hihat", 1), Hit(0.5, "hihat", 1), Hit(1, "hihat", 1),
            Hit(1.99, "snare", 0.3), Hit(2.01, "snare", 0.9), Hit(2.0, "kick", 0.7)]
    score = quantize(hits, beats, beats[::4], triplets="off")
    assert [(n.tick, n.inst) for n in score.bars[1].notes] == [(0, "kick"), (0, "snare")]
    assert score.bars[1].notes[1].velocity == 110


def test_velocity_normalizes_each_instrument():
    hits = [Hit(i * 0.5, inst, strength) for inst, strengths in [("kick", [0.1, 0.2, 0.4]), ("snare", [0.3, 0.6, 0.9])]
            for i, strength in enumerate(strengths)] + [Hit(1.5, "tom", 0)]
    score = quantize(hits, steady(4), [0])
    expected = {"kick": [59, 79, 110], "snare": [65, 90, 110], "tom": [40]}
    for inst, velocities in expected.items():
        assert [n.velocity for n in score.bars[0].notes if n.inst == inst] == velocities
    assert [n.inst for n in score.bars[0].notes[:2]] == list(INSTRUMENTS[:2])


@pytest.mark.parametrize("beats", [[], [0]])
def test_too_few_beats(beats):
    with pytest.raises(ValueError):
        quantize([], beats, [])


@pytest.mark.parametrize("kwargs", [{"grid": 32}, {"triplets": "yes"}, {"meter": 0}, {"meter": 13}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        quantize([], steady(4), [], **kwargs)


def test_invalid_detector_data():
    with pytest.raises(ValueError):
        quantize([], [0, 0], [])
    with pytest.raises(ValueError):
        quantize([Hit(0, "unknown", 1)], [0, 0.5], [])


def test_funkorama_fixture():
    raw = json.loads((Path(__file__).parent / "fixtures/funkorama.json").read_text())
    score = quantize([Hit(**hit) for hit in raw["hits"]], raw["beats"], raw["downbeats"])
    assert 83 <= len(score.bars) <= 85
    assert all(bar.beats == 4 for bar in score.bars)
    assert score.tempo == pytest.approx(100, abs=2)
    assert sum(len(bar.notes) for bar in score.bars) > 500


def test_spurious_mid_bar_downbeats_keep_regular_bars():
    # Trackers often flag beat 3 as a downbeat, or drift to it for a few bars.
    beats = steady(64)
    halves = list(range(0, 16, 2))  # every two beats in the intro
    shifted = [18, 22, 26]  # three bars heard half a bar late
    rest = list(range(32, 64, 4))
    score = quantize([], beats, [beats[i] for i in halves + [16] + shifted + rest], odd_bars="auto")
    assert [bar.beats for bar in score.bars] == [4] * 16
    assert score.bars[0].start == beats[0]


def test_all_downbeats_two_beats_apart_become_four_four():
    beats = steady(32)
    score = quantize([], beats, beats[::2])
    assert [bar.beats for bar in score.bars] == [4] * 8


def test_skipped_beat_is_filled_in():
    beats = steady(32)
    hits = [Hit(beats[9], "snare", 0.8)]
    score = quantize(hits, beats[:9] + beats[10:], beats[::4])
    assert [bar.beats for bar in score.bars] == [4] * 8
    assert positions(score) == [(2, 24, "snare")]


def test_zero_penalty_follows_detected_downbeats():
    beats = steady(16)
    score = quantize([], beats, [beats[i] for i in (0, 4, 6, 12)], odd_bars="detected")
    assert [bar.beats for bar in score.bars] == [4, 2, 6, 4]


def test_double_time_stretch_is_brought_back_to_one_tempo_level():
    # The tracker counts 16 beats of a 64-beat song twice as fast and scatters downbeats there.
    beats = steady(64)
    doubled = sorted(beats + [t + 0.25 for t in beats[24:40]])
    downbeats = beats[0:24:4] + [t + 0.25 for t in beats[24:40:2]] + beats[40::4]
    hits = [Hit(t, "kick", 0.8) for t in beats]
    score = quantize(hits, doubled, downbeats)
    assert [bar.beats for bar in score.bars] == [4] * 16
    assert all([n.tick for n in bar.notes] == [0, 24, 48, 72] for bar in score.bars)


def test_real_tempo_change_is_kept():
    beats = steady(16) + [8 + 0.35 * i for i in range(17)]
    score = quantize([], beats, [beats[i] for i in range(0, 32, 4)])
    assert round(score.bars[1].tempo) == 120 and round(score.bars[-2].tempo) == 171


def test_beat_shift_moves_the_bar_line():
    beats = steady(16)
    hits = [Hit(beats[1], "kick", 0.8)]
    assert positions(quantize(hits, beats, beats[::4])) == [(0, 24, "kick")]
    assert positions(quantize(hits, beats, beats[::4], beat_shift=1)) == [(1, 0, "kick")]


def test_tempo_scale_halves_and_doubles_the_count():
    beats = steady(32)
    assert len(quantize([], beats, beats[::4], tempo_scale=0.5).bars) == 4
    assert len(quantize([], beats, beats[::4], tempo_scale=2).bars) == 16


def test_fast_fills_choose_thirty_seconds_and_sextuplets():
    beats = steady(16, 0.6)
    groove = pattern([b for b in range(16) if b not in (5, 6, 9)], [0, 1 / 4, 1 / 2, 3 / 4], "hihat", 0.6)
    run = pattern([5], [k / 8 for k in range(8)], "snare", 0.6)
    doubles = pattern([6], [0, 1 / 8, 1 / 4, 1 / 2, 3 / 4], "snare", 0.6)
    sextuplet = pattern([9], [k / 6 for k in range(6)], "tom", 0.6)
    score = quantize(groove + run + doubles + sextuplet, beats, beats[::4])
    assert grids(score) == [4] * 5 + [8, [8, 4], 4, 4, 6] + [4] * 6
    assert ticks(score, "snare") == [(1, 24 + 3 * k) for k in range(8)] + [(1, 48 + t) for t in (0, 3, 6, 12, 18)]
    assert ticks(score, "tom") == [(2, 24 + 4 * k) for k in range(6)]
    assert len(positions(score)) == len(groove + run + doubles + sextuplet)


@pytest.mark.parametrize("bpm", [70, 90, 98, 118, 140])
def test_jittered_straight_sixteenths_stay_sixteenths(bpm):
    # Players within 15 ms, a detector on a 10 ms grid, a beat tracker on a 20 ms grid.
    rng, interval = random.Random(bpm), 60 / bpm
    beats = [round(i * interval / 0.02) * 0.02 for i in range(65)]
    # Backbeat on the beat (plus a syncopation), so the count is not read as double time.
    parts = [("hihat", (0, 1, 2, 3)), ("kick", (0, 3)), ("snare", (0, 2))]
    hits = [Hit(round((beat + slot / 4) * interval + rng.uniform(-0.015, 0.015), 2), inst, 0.8)
            for beat in range(64) for inst, slots in parts for slot in slots if inst == "hihat" or beat % 2 == (inst == "snare")]
    score = quantize(hits, beats, beats[::4])
    assert set(grids(score)) == {4}
    assert len(positions(score)) == len(hits)
    assert all(note.tick % 6 == 0 for bar in score.bars for note in bar.notes)


def test_same_instrument_double_is_kept_but_a_flam_is_one_stroke():
    beats = steady(8, 0.6)
    hats = pattern(range(8), [0, 1 / 4, 1 / 2, 3 / 4], "hihat", 0.6)
    # 45 ms apart and both nearest the same 16th; the second is not close enough to a 32nd to count by itself.
    double = [Hit(1.35, "kick", 0.8), Hit(1.395, "kick", 0.6)]
    score = quantize(hats + double, beats, beats[::4])
    assert ticks(score, "kick") == [(0, 54), (0, 57)]
    assert grids(score) == [4, 4, [8, 4], 4] + [4] * 4
    assert grids(quantize(hats + double[1:], beats, beats[::4])) == [4] * 8
    flam = [Hit(1.38, "kick", 0.6), Hit(1.40, "kick", 0.9)]  # 20 ms apart, but either side of where 32nds would split them
    score = quantize(hats + flam, beats, beats[::4])
    assert [(n.tick, n.velocity) for n in score.bars[0].notes if n.inst == "kick"] == [(54, 110)]
    assert grids(score) == [4] * 8


def test_uneven_fast_strokes_keep_a_slot_each():
    beats = steady(8, 0.6)
    # Five strokes in one beat: 32nds hold them all, sextuplets would merge two.
    run = [Hit(2.4 + t, "snare", 0.8) for t in (0, 0.06, 0.12, 0.21, 0.42)]
    score = quantize(run, beats, beats[::4])
    assert ticks(score, "snare") == [(1, 0), (1, 3), (1, 6), (1, 9), (1, 18)]
    # 65 ms apart and both nearest the same 32nd: the one nearer a free slot moves over.
    double = [Hit(2.4 + 0.27, "tom", 0.8), Hit(2.4 + 0.335, "tom", 0.8)]
    assert ticks(quantize(double, beats, beats[::4]), "tom") == [(1, 12), (1, 15)]
    # Equal total error across the boundary: the later assignment keeps both strokes.
    across = [Hit(3.0 - 0.035, "tom", 0.8), Hit(3.0 + 0.035, "tom", 0.8)]
    assert ticks(quantize(across, beats, beats[::4]), "tom") == [(1, 24), (1, 27)]


def test_second_stroke_of_a_double_is_not_read_as_lateness():
    from stickmap.quantize import _offsets

    times = [i * 0.5 for i in range(24)]
    doubles = [Hit(beat + late, "tom", 1) for beat in times[:20] for late in (0.0, 0.055)]
    assert _offsets(doubles, times)["tom"] == 0
    assert _offsets(doubles[:6], times)["snare"] == 0  # three beats are no basis for an offset
    late = [Hit(beat + 0.02, "hihat", 1) for beat in times[:20]]
    assert _offsets(late, times)["hihat"] == pytest.approx(0.02)


def test_ghost_notes_stay_marked_and_quiet():
    beats = steady(8)
    hits = [Hit(0.5, "snare", 1.0), Hit(0.75, "snare", 0.2, ghost=True), Hit(1.5, "snare", 1.1)]
    notes = [note for bar in quantize(hits, beats, beats[::4]).bars for note in bar.notes]
    assert [(note.tick, note.ghost) for note in notes] == [(24, False), (36, True), (72, False)]
    assert notes[1].velocity < notes[0].velocity - 40


def test_triplets_off_allows_only_straight_grids():
    beats = steady(12, 0.6)
    hits = (pattern(range(4), [0, 2 / 3], "hihat", 0.6) + pattern(range(4, 8), [k / 6 for k in range(6)], "snare", 0.6)
            + pattern(range(8, 12), [k / 8 for k in range(8)], "tom", 0.6))
    assert grids(quantize(hits, beats, beats[::4])) == [3] * 4 + [6] * 4 + [8] * 4
    straight = grids(quantize(hits, beats, beats[::4], triplets="off"))
    assert set(straight) <= {4, 8} and straight[8:] == [8] * 4


def test_lone_sloppy_beat_follows_its_straight_neighbours():
    beats = steady(16)
    sloppy = [Hit(3.0 + 0.31 * 0.5, "snare", 0.8)]  # an "e" played late, nearer the triplet
    quiet = quantize(pattern(range(16), [0], "kick") + sloppy, beats, beats[::4])
    # An isolated late stroke is insufficient evidence for changing rhythmic
    # family; the final spelling cost keeps this sparse beat on its base grid.
    assert grids(quiet)[6] == 4
    busy = quantize(pattern(range(16), [0], "kick") + pattern([b for b in range(16) if b != 6], [0, 1 / 4, 1 / 2, 3 / 4], "hihat") + sloppy, beats, beats[::4])
    assert set(grids(busy)) == {4}
    assert ticks(busy, "snare") == [(1, 48 + 6)]


def test_weak_beat_inside_a_shuffle_stays_triplet():
    beats = steady(16)
    weak = [Hit(3.0, "hihat", 0.8), Hit(3.0 + 0.71 * 0.5, "hihat", 0.8)]  # the swung note dragged toward the "a"
    quiet = quantize(pattern([b for b in range(16) if b != 6], [0], "hihat") + weak, beats, beats[::4])
    assert set(grids(quiet)) == {4}
    shuffle = quantize(pattern([b for b in range(16) if b != 6], [0, 2 / 3], "hihat") + weak, beats, beats[::4])
    assert set(grids(shuffle)) == {3}
    assert ticks(shuffle, "hihat") == [(bar, beat * 24 + tick) for bar in range(4) for beat in range(4) for tick in (0, 16)]


def test_consistently_late_instrument_lands_on_its_slots():
    beats = steady(48, 0.45)
    hats = pattern(range(48), [0], "hihat", 0.45)
    backbeats = pattern(range(1, 48, 2), [0], "snare", 0.45, late=0.025)
    ghosts = pattern(range(2, 48, 8), [1 / 4], "snare", 0.45, late=0.025)  # late "e"s drift toward the triplet
    score = quantize(hats + backbeats + ghosts, beats, beats[::4])
    assert set(grids(score)) == {4}
    assert ticks(score, "snare") == sorted([(b // 4, b % 4 * 24) for b in range(1, 48, 2)] + [(b // 4, b % 4 * 24 + 6) for b in range(2, 48, 8)])
    assert ticks(score, "hihat") == [(b // 4, b % 4 * 24) for b in range(48)]


def test_offset_ignores_hits_outside_the_covered_beats():
    beats = [2 + i * 0.2 for i in range(8)]
    intro = [Hit(0.1 + i * 0.05, "hihat", 0.5) for i in range(20)]
    score = quantize(intro + [Hit(t, "kick", 1) for t in beats[:4]], beats, [2])
    assert positions(score) == [(0, tick, "kick") for tick in (0, 24, 48, 72)]


def test_exact_triplet_hits_are_not_read_as_late_straight_hits():
    beats = steady(8)
    score = quantize([Hit(i * 0.5 + 1 / 6, "hihat", 1) for i in range(8)], beats, beats[::4])
    assert set(grids(score)) == {3}
    assert ticks(score, "hihat") == [(bar, beat * 24 + 8) for bar in range(2) for beat in range(4)]


def test_hits_at_the_edges_of_the_score_are_kept():
    early = quantize([Hit(0.49, "kick", 1), Hit(1.0, "snare", 1)], [0.5, 1, 1.5, 2], [0.5])
    assert positions(early) == [(0, 0, "kick"), (0, 24, "snare")]
    late = quantize([Hit(0, "kick", 1), Hit(0.5, "kick", 1), Hit(1.99, "kick", 1)], [0, 0.5, 1, 1.5], [0])
    assert len(late.bars) == 1
    assert positions(late) == [(0, 0, "kick"), (0, 24, "kick"), (0, 93, "kick")]
    eighths = quantize([Hit(1.99, "kick", 1)], [0, 0.5, 1, 1.5], [0], grid=8)
    assert positions(eighths) == [(0, 84, "kick")]


def test_old_score_json_without_divisions_is_scaled():
    bar = {"index": 0, "beats": 4, "beat_times": [0, 0.5, 1, 1.5, 2], "subdivisions": [4, 3, 4, 4], "tempo": 120,
           "notes": [{"tick": 3, "inst": "snare", "velocity": 90}, {"tick": 16, "inst": "tom", "velocity": 80}]}
    score = Score.from_json(json.dumps({"title": "Old", "bars": [bar], "grid": 16, "tempo": 120.0, "subtitle": ""}))
    assert score.divisions == DIVISIONS == 24
    assert [note.tick for note in score.bars[0].notes] == [6, 32]
    assert score.bars[0].tick_time(6) == pytest.approx(0.125)
    assert score.bars[0].ticks == 96
    assert json.loads(score.to_json())["divisions"] == 24
    assert Score.from_json(score.to_json()) == score


def test_backbeat_between_the_beats_doubles_the_count():
    # Kick on every tracked beat, snare halfway between: that is kick-snare at twice the tempo.
    beats = steady(64)
    hits = [Hit(t, "kick", 0.8) for t in beats] + [Hit(t + 0.25, "snare", 0.8) for t in beats[:-1]]
    score = quantize(hits, beats, beats[::4])
    assert round(score.tempo) == 240 and {bar.beats for bar in score.bars} == {4}
    snare_ticks = {note.tick for bar in score.bars for note in bar.notes if note.inst == "snare"}
    assert snare_ticks == {24, 72}  # beats 2 and 4
    assert round(quantize(hits, beats, beats[::4], tempo_scale=0.5).tempo) == 120


def test_backbeat_on_the_beats_keeps_the_count():
    beats = steady(64)
    hits = [Hit(t, "kick", 0.8) for t in beats[::2]] + [Hit(t, "snare", 0.8) for t in beats[1::2]]
    assert round(quantize(hits, beats, beats[::4]).tempo) == 120


def test_tracker_phase_slip_follows_the_longer_side():
    # After two bars the tracker shifts by half a beat and stays there for the rest of the song.
    early, late = steady(8), [4.25 + 0.5 * i for i in range(56)]
    hits = [Hit(t, "kick", 0.8) for t in late[::2]]
    score = quantize(hits, early + late, late[::4])
    assert {note.tick % 24 for bar in score.bars for note in bar.notes} == {0}


def test_erratic_stretch_does_not_leave_the_grid_adrift():
    # Eight beats of tracker noise, then steady again a little off the walk's own time.
    before = steady(16)
    noise = [8.0, 8.13, 8.61, 8.9, 9.37, 9.5, 10.05, 10.2, 10.71, 11.1, 11.43, 11.9]
    after = [12.12 + 0.5 * i for i in range(32)]
    hits = [Hit(t, "kick", 0.8) for t in after]
    score = quantize(hits, before + noise + after, before[::4] + after[::4])
    late = [note.tick % 24 for bar in score.bars[8:] for note in bar.notes]
    assert late and set(late) == {0}


def test_backbeat_after_every_other_beat_means_the_tracker_is_half_a_beat_off():
    # Tracker beats sit on the off-beats: the real beats are halfway between, tempo unchanged.
    tracked = [0.25 + 0.5 * i for i in range(64)]
    real = [0.5 * i for i in range(1, 64)]
    hits = [Hit(t, "snare", 0.8) for t in real[::2]] + [Hit(t, "kick", 0.8) for t in real[1::2]]
    score = quantize(hits, tracked, tracked[::4])
    assert round(score.tempo) == 120
    assert {n.tick for bar in score.bars for n in bar.notes if n.inst == "snare"} == {24, 72}
    assert {n.tick for bar in score.bars for n in bar.notes if n.inst == "kick"} == {0, 48}


def test_snare_on_the_beat_puts_the_backbeat_on_two_and_four():
    # The tracker calls the snare beats downbeats; the snare still belongs on 2 and 4.
    beats = steady(64)
    hits = [Hit(t, "snare", 0.8) for t in beats[::2]] + [Hit(t, "kick", 0.8) for t in beats[1::2]]
    score = quantize(hits, beats, beats[::4])
    assert {n.tick for bar in score.bars for n in bar.notes if n.inst == "snare"} == {24, 72}


def test_half_beat_shift_moves_the_bar_line_by_an_eighth():
    beats = steady(32)
    hits = [Hit(t, "hihat", 0.8) for t in beats]
    shifted = quantize(hits, beats, beats[::4], beat_shift=0.5)
    assert {n.tick % 24 for bar in shifted.bars for n in bar.notes} == {12}
    with pytest.raises(ValueError):
        quantize(hits, beats, beats[::4], beat_shift=0.125)


def test_quarter_beat_shift_moves_the_bar_line_by_a_sixteenth():
    beats = steady(32)
    hits = [Hit(t, "hihat", 0.8) for t in beats]
    later = quantize(hits, beats, beats[::4], beat_shift=0.25)
    earlier = quantize(hits, beats, beats[::4], beat_shift=-0.25)
    assert {n.tick % 24 for bar in later.bars for n in bar.notes} == {18}
    assert {n.tick % 24 for bar in earlier.bars for n in bar.notes} == {6}


@pytest.mark.parametrize("onsets,subdivision", [
    ([0, 3, 6, 12, 18], [8, 4]),
    ([0, 6, 12, 16, 20], [4, 6]),
    ([0, 3, 6, 12, 16, 20], [8, 6]),
    ([0, 4, 8, 12, 15, 18, 21], [6, 8]),
], ids=["herta-and-landing", "sixteenths-then-sextuplet", "thirty-seconds-then-sextuplet", "sextuplet-then-thirty-seconds"])
def test_half_beat_figures(onsets, subdivision):
    beats = steady(16, 0.6)
    hits = [Hit(3 + tick * 0.6 / 24, "snare", 0.8) for tick in onsets]
    score = quantize(hits, beats, beats[::4])
    assert ticks(score, "snare") == [(1, 24 + tick) for tick in onsets]
    assert score.bars[1].subdivisions[1] == subdivision
    assert score.merged_strokes == 0
    assert Score.from_json(score.to_json()) == score


@pytest.mark.parametrize("beat", [0, 3, 7])
def test_four_strokes_survive_a_beat_or_bar_boundary(beat):
    hits = [Hit(beat * 0.6 + t, "tom", 1) for t in (0.49, 0.535, 0.58, 0.625)]
    score = quantize(hits, steady(16, 0.6), [0])
    times = [bar.tick_time(note.tick) for bar in score.bars for note in bar.notes]
    assert times == pytest.approx([beat * 0.6 + t for t in (0.45, 0.525, 0.6, 0.675)])
    assert score.merged_strokes == 0


def test_next_beat_first_stroke_does_not_reserve_an_unused_start_slot():
    hits = [Hit(0.576, "tom", 1), Hit(0.66, "tom", 1)]
    score = quantize(hits, steady(16, 0.6), [0])
    assert ticks(score, "tom") == [(0, 24), (0, 27)]
    assert score.merged_strokes == 0


@pytest.mark.parametrize("stronger", [0, 1])
def test_terminal_pair_counts_the_merge_and_keeps_the_stronger_stroke(stronger):
    hits = [Hit(t, "tom", 1 if i == stronger else 0.2, i != stronger) for i, t in enumerate((9.54, 9.59))]
    score = quantize(hits, steady(16, 0.6), [0])
    # Tick 90 is over three ticks away from both: only the last 32nd is allowed.
    assert ticks(score, "tom") == [(3, 93)]
    assert score.merged_strokes == 1
    assert score.bars[-1].notes[0].velocity == 110
    assert score.bars[-1].notes[0].ghost is False
    assert Score.from_json(score.to_json()) == score


def test_terminal_assignment_keeps_two_slots_when_within_the_movement_limit():
    score = quantize([Hit(t, "tom", 1) for t in (9.50, 9.59)], steady(16, 0.6), [0])
    assert ticks(score, "tom") == [(3, 90), (3, 93)]
    assert score.merged_strokes == 0


def test_different_tom_sizes_do_not_collide_at_the_boundary():
    hits = [Hit(t, inst, 1) for inst in ("tom_high", "tom", "tom_low") for t in (0.576, 0.66)]
    score = quantize(hits, steady(16, 0.6), [0])
    for inst in ("tom_high", "tom", "tom_low"):
        assert ticks(score, inst) == [(0, 24), (0, 27)]
    assert score.merged_strokes == 0


def test_reading_mode_drops_are_not_counted_as_collisions():
    hits = [Hit(t, "hihat", 1) for t in (0, 0.15, 0.3, 0.45)]
    score = quantize(hits, steady(16, 0.6), [0], grid=8)
    assert ticks(score, "hihat") == [(0, 0), (0, 12)]
    assert score.merged_strokes == 0


def test_mixed_figures_with_triplets_disabled_have_only_straight_halves():
    hits = [Hit(tick * 0.6 / 24, "snare", 1) for tick in (0, 4, 8, 12, 15, 18, 21)]
    score = quantize(hits, steady(16, 0.6), [0], triplets="off")
    assert all(g in (4, 8) for s in grids(score) for g in (s if isinstance(s, list) else [s]))
    assert all(note.tick % 3 == 0 for bar in score.bars for note in bar.notes)
    assert len(positions(score)) == len(hits)


@pytest.mark.parametrize("onsets", [
    [0, 3, 6, 12, 18],
    [0, 6, 12, 15, 18],
    [0, 4, 8, 12, 16, 20],
])
def test_reading_mode_keeps_a_terminal_fine_grid_supported_by_other_strokes(onsets):
    hits = [Hit(1.8 + tick * 0.6 / 24, "tom", 1) for tick in onsets] + [Hit(2.39, "tom", 1)]
    normal = quantize(hits, steady(4, 0.6), [0])
    reading = quantize(hits, steady(4, 0.6), [0], grid=8)
    expected = positions(normal)
    if onsets == [0, 6, 12, 15, 18]:
        expected.remove((0, 78, "tom"))  # the first half's odd 16th is still omitted
    assert positions(reading) == expected
    assert reading.merged_strokes == normal.merged_strokes == 0


def test_reading_mode_terminal_merge_is_counted_and_retains_the_stronger_hit():
    hits = [Hit(1.75, "tom", 0.2, True), Hit(1.99, "tom", 1)]
    score = quantize(hits, steady(4), [0], grid=8)
    assert ticks(score, "tom") == [(0, 84)]
    assert score.merged_strokes == 1
    assert score.bars[0].notes[0].ghost is False
