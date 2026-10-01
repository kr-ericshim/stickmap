import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("adtof_pytorch")

from groovemap import models
from groovemap.fills import attack_times, tom_runs

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
