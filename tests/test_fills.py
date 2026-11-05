import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("adtof_pytorch")

from stickmap import models
from stickmap.fills import attack_times, fill_snares, tom_runs
from stickmap.score import Hit

TOM = models.ADTOF_CLASSES.index("tom")


def test_strokes_the_picker_passed_over_complete_the_fill():
    act = np.zeros((600, 5), dtype=np.float32)
    act[99:102, TOM] = 0.4, 0.9, 0.7
    act[102:135, TOM] = 0.6  # the activation never comes down during the fill
    act[116, TOM] = act[132, TOM] = 0.9
    act[250:300, TOM] = np.linspace(0, 0.6, 50)  # a slow swell: nothing to pick, no run to join
    act[300:340, TOM] = 0.6
    act[320, TOM] = 0.9
    hits = models.pick_hits(act, mix=act)
    assert [h.time for h in hits] == [1.0]
    everywhere = np.array([1.0, 1.16, 1.32, 3.2])
    assert [h.time for h in tom_runs(hits, act, act, everywhere)] == [1.0, 1.16, 1.32]
    # Without an attack in the audio, a bump in the activation is not a stroke.
    assert [h.time for h in tom_runs(hits, act, act, np.array([1.0, 1.32]))] == [1.0]
    assert tom_runs(hits, act, act, np.empty(0)) == hits


def test_attacks_are_found_whatever_the_drum(tmp_path):
    sr = 44100
    audio = np.zeros(3 * sr, dtype=np.float32)
    t = np.arange(int(0.6 * sr)) / sr
    rng = np.random.default_rng(0)
    for time, hz in [(0.5, 90), (1.0, 180), (1.08, 140)]:
        audio[int(time * sr):int(time * sr) + len(t)] += (np.sin(2 * np.pi * hz * t) * np.exp(-t * 15)).astype(np.float32)
    burst = (rng.standard_normal(len(t)) * np.exp(-t * 40) * 0.5).astype(np.float32)
    audio[int(2.0 * sr):int(2.0 * sr) + len(t)] += burst
    sf.write(tmp_path / "drums.wav", np.stack([audio, audio], axis=1), sr)
    found = attack_times(tmp_path / "drums.wav")
    # Levels are judged against the recording itself, so the near-silence here also
    # yields stray attacks; a real one is never missed.
    assert all(np.abs(found - time).min() < 0.015 for time in (0.5, 1.0, 1.08, 2.0))


def test_quiet_snares_between_the_toms_of_a_fill_are_kept():
    # A fill of toms every 80 ms with a quiet snare stroke between two of them.
    fill = [Hit(1.0 + 0.08 * i, "tom", 0.8) for i in range(6)]
    response = np.zeros((300, 2), dtype=np.float32)  # snare loudness, snare share of the sound
    act = np.zeros((300, 5), dtype=np.float32)
    act[:, models.ADTOF_CLASSES.index("snare")] = 0.05
    response[123] = 0.3, 0.8  # between the toms at 1.20 and 1.28 s
    response[200] = 0.3, 0.8  # long after the fill
    response[140] = 0.3, 0.4  # in the fill, but the snare is not most of what sounds
    attacks = np.array([1.23, 1.4, 2.0])
    added = [h for h in fill_snares(fill, act, response, attacks) if h.inst == "snare"]
    assert [(h.time, h.ghost) for h in added] == [(1.23, True)]
    # Without an attack in the audio there is no stroke, and outside a fill nothing changes.
    assert fill_snares(fill, act, response, np.array([2.0])) == fill
    assert fill_snares(fill[:2], act, response, attacks) == fill[:2]
