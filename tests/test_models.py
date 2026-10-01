import numpy as np
import pytest

pytest.importorskip("adtof_pytorch")

from groovemap import models

KICK, SNARE, TOM = (models.ADTOF_CLASSES.index(inst) for inst in ("kick", "snare", "tom"))


def activations(frames=400, **peaks):
    """Silence with a three-frame bump of the given height at each (frame, height)."""
    act = np.zeros((frames, 5), dtype=np.float32)
    for inst, bumps in peaks.items():
        for frame, height in bumps:
            act[frame - 1:frame + 2, models.ADTOF_CLASSES.index(inst)] = height * np.array([0.4, 1, 0.4])
    return act


def times(hits, inst):
    return [round(h.time, 2) for h in hits if h.inst == inst]


def test_second_stroke_of_a_fast_double_is_kept():
    # 80 ms apart: the first stroke lifts the average the second is measured against.
    act = activations(tom=[(100, 0.9), (108, 0.6), (200, 0.9)], kick=[(300, 0.9), (306, 0.5)])
    hits = models.pick_hits(act)
    assert times(hits, "tom") == [1.0, 1.08, 2.0]
    assert times(hits, "kick") == [3.0, 3.06]


def test_bump_on_the_tail_of_a_hit_is_not_a_stroke():
    act = activations(tom=[(100, 0.9)])
    act[101:112, TOM] = np.linspace(0.5, 0.35, 11)  # rings on without dipping
    act[108, TOM] = 0.45
    assert times(models.pick_hits(act), "tom") == [1.0]


def test_mix_steadies_toms_and_drops_a_snares_echo():
    stem = activations(tom=[(100, 0.9), (200, 0.5), (300, 0.45)], snare=[(200, 0.9)])
    mix = activations(tom=[(100, 0.7), (200, 0.5), (300, 0.2)], snare=[(200, 0.9)])
    assert times(models.pick_hits(stem), "tom") == [1.0, 2.0, 3.0]
    # 3.0 s is too weak once averaged with the mix; 2.0 s sits under a stronger snare.
    assert times(models.pick_hits(stem, mix=mix), "tom") == [1.0]


def test_snare_loudness_marks_ghosts_and_adds_quiet_strokes():
    act = activations(snare=[(100, 0.9), (200, 0.9), (300, 0.02)], kick=[(150, 0.9)])
    response = np.zeros((400, 2), dtype=np.float32)
    for frame, loudness in [(100, 1.0), (200, 0.2), (300, 0.1), (153, 0.3), (350, 0.3)]:
        response[frame] = loudness, 0.9
    hits = [h for h in models.pick_hits(act, snare=response) if h.inst == "snare"]
    # 1.53 s is right after the kick and 3.5 s has no activation at all: neither is added.
    assert [(round(h.time, 2), round(h.strength, 2), h.ghost) for h in hits] == [
        (1.0, 1.0, False), (2.0, 0.2, True), (3.0, 0.1, True),
    ]
    assert not any(h.ghost for h in models.pick_hits(act, snare=np.zeros((0, 2))))


def test_a_level_stretch_of_snare_loudness_is_one_stroke():
    act = np.full((100, 5), 0.0, dtype=np.float32)
    act[:, SNARE] = 0.02
    response = np.zeros((105, 2), dtype=np.float32)  # a cache a few frames longer than the activations
    response[30:41] = 0.1, 0.9
    response[102] = 0.5, 0.9
    peaks = {inst: np.array([], dtype=int) for inst in models.ADTOF_CLASSES}
    assert [h.time for h in models._snares(act, response, peaks)] == [0.3]


def test_fast_stroke_at_the_very_end_is_kept():
    act = np.zeros((38, 5), dtype=np.float32)
    for frame, height in [(30, 0.9), (36, 0.5)]:
        act[frame - 1:frame + 2, TOM] = height * np.array([0.4, 1, 0.4])
    assert models.picked_frames(act)["tom"].tolist() == [30, 36]


def test_drum_sounds_are_learned_from_hits_that_stand_alone():
    from groovemap import dynamics

    spectrogram = np.ones((4, 200), dtype=np.float32)
    together = np.arange(10, 150, 10)
    peaks = {inst: np.array([], dtype=int) for inst in models.ADTOF_CLASSES}
    peaks["snare"], peaks["kick"] = together, np.r_[together[:8], 170, 180, 190]
    spectrogram[0, [170, 180, 190]] = 5  # the kick alone is low ...
    spectrogram[3, together[8:]] = 5  # ... and the snare alone is high
    patches, names = dynamics._patches(spectrogram, peaks)
    assert names == ["kick", "snare"]
    # Six snares have no kick on their frame: they, not the eight shared ones, are the snare's sound.
    assert patches[3, 1, 1] > 0.9 and patches[0, 1, 1] < 0.1
    peaks = {inst: np.array([], dtype=int) for inst in models.ADTOF_CLASSES}
    peaks["snare"] = np.array([0, 1, 2, 3])  # too close to the start to cut a patch from
    assert dynamics._patches(np.ones((2, 20), dtype=np.float32), peaks)[1] == []
