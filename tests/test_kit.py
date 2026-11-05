import json

import numpy as np
import pytest
import soundfile as sf

from stickmap import kit
from stickmap.score import Hit

SR = 44100
LOW, HIGH, BRIGHT = 60.0, 900.0, 5000.0  # a drum-like tone, a much higher tone, and a hat-like one


def stem(path, strokes, seconds=40.0):
    """A drum stem of decaying sines: (time, Hz, level) per stroke; strokes of one Hz are the same sound."""
    audio = np.zeros(int(seconds * SR), dtype=np.float32)
    t = np.arange(int(0.15 * SR)) / SR
    for time, hz, level in strokes:
        start = int(time * SR)
        audio[start:start + len(t)] += (level * np.sin(2 * np.pi * hz * t) * np.exp(-t * 25)).astype(np.float32)
    sf.write(path, np.stack([audio, audio], axis=1), SR, subtype="FLOAT")  # 16-bit rounding would make levels sounds
    return path


def two_sounds(tmp_path):
    """Forty "kick" hits: half a low tone, half a high one, well apart in time."""
    times = 0.5 + 0.8 * np.arange(40)
    strokes = [(time, LOW if i % 2 == 0 else HIGH, 1.0) for i, time in enumerate(times)]
    hits = [Hit(float(time), "kick", 0.9) for time in times]
    return stem(tmp_path / "drums.wav", strokes), hits, np.array(times)


def test_one_detected_class_splits_into_its_two_sounds_with_clean_spread_examples(tmp_path):
    drums, hits, times = two_sounds(tmp_path)
    learned = kit.learn(drums, hits, times, {"drums.wav": "abc"})
    assert [(g.id, g.detected, g.count, g.mixed) for g in learned.groups] == [("kick-1", "kick", 20, 0), ("kick-2", "kick", 20, 0)]
    low, high = learned.groups
    assert all(round((e.time - 0.5) / 0.8) % 2 == 0 for e in low.examples)  # dark first, and the right strokes
    assert all(round((e.time - 0.5) / 0.8) % 2 == 1 for e in high.examples)
    for group in learned.groups:
        assert len(group.examples) == 6
        gaps = np.diff([e.time for e in group.examples])
        assert gaps.min() >= 0.03 * 40 - 1e-9  # from different parts of the song
        assert all((e.start, e.end) == (max(0.0, e.time - 0.6), e.time + 0.4) for e in group.examples)
    assert low.similar == {"kick": 1.0} and high.similar["kick"] < 0.9  # against the class's main (larger or first) sound
    assert learned == kit.learn(drums, hits[::-1], times, {"drums.wav": "abc"})


def test_unexplained_attacks_form_their_own_groups_and_named_ones_become_hits(tmp_path):
    times = 0.5 + 0.8 * np.arange(40)
    strokes = [(time, LOW if i % 2 == 0 else HIGH, 1.0 if i % 4 != 1 else 0.2) for i, time in enumerate(times)]
    drums = stem(tmp_path / "drums.wav", strokes)
    hits = [Hit(float(time), "kick", 0.9) for time in times[::2]]  # the high tones were never detected
    learned = kit.learn(drums, hits, times, {})
    assert [(g.id, g.detected, g.count) for g in learned.groups] == [("kick-1", "kick", 20), ("extra-1", None, 20)]
    assert kit.rename(hits, learned, {}, drums, times) is hits
    assert kit.rename(hits, learned, {"extra-1": "ignore"}, drums, times) == hits
    named = kit.rename(hits, learned, {"extra-1": "snare"}, drums, times)
    assert [h.inst for h in named] == ["kick", "snare"] * 20
    added = [h for h in named if h.inst == "snare"]
    assert [h.ghost for h in added] == [True, False] * 10  # the quiet strokes, against the group's typical one
    assert all(h.strength < 0.5 for h in added if h.ghost) and all(h.strength >= 0.5 for h in added if not h.ghost)
    assert [h.ghost for h in kit.rename(hits, learned, {"extra-1": "tom"}, drums, times) if h.inst == "tom"] == [False] * 20


def test_renaming_keeps_order_strength_and_only_snare_ghosts(tmp_path):
    drums, hits, times = two_sounds(tmp_path)
    hits = [Hit(h.time, h.inst, 0.3 + 0.01 * i, ghost=False) for i, h in enumerate(hits)]
    hits += [Hit(float(times[1]), "snare", 0.2, ghost=True), Hit(float(times[3]), "hihat", 0.5)]
    hits.sort(key=lambda h: h.time)
    learned = kit.learn(drums, hits, times, {})
    ids = {g.detected: g.id for g in learned.groups}
    renamed = kit.rename(hits, learned, {"kick-2": "tom", ids["snare"]: "kick", ids["hihat"]: "ignore", "nobody": "ride"}, drums, times)
    assert len(renamed) == len(hits) - 1 and all(h.inst != "hihat" for h in renamed)
    assert [h.time for h in renamed] == [h.time for h in hits if h.inst != "hihat"]
    kicks = [h for h in renamed if h.inst == "kick"]
    assert len(kicks) == 21 and not any(h.ghost for h in kicks)  # a ghost renamed off the snare is no ghost
    assert [round(h.strength, 2) for h in renamed if h.inst == "tom"] == [round(0.3 + 0.01 * i, 2) for i in range(1, 40, 2)]
    still_snare = kit.rename(hits, learned, {ids["snare"]: "sidestick"}, drums, times)
    assert [h.ghost for h in still_snare if h.inst == "sidestick"] == [True]


def test_map_round_trips_and_is_kept_until_its_caches_change(tmp_path, monkeypatch):
    drums, hits, times = two_sounds(tmp_path)
    activations, attacks = tmp_path / "activations.npy", tmp_path / "attacks.npy"
    np.save(activations, np.zeros((10, 5)))
    np.save(attacks, times)
    learned = kit.load_or_learn(tmp_path, drums, activations, attacks, hits, times)
    saved = kit.SoundMap.from_dict(json.loads((tmp_path / kit.FILE).read_text()))
    assert saved == learned and set(saved.inputs) == {"drums.wav", "activations.npy", "attacks.npy"}
    monkeypatch.setattr("stickmap.dynamics._spectrogram", lambda _: pytest.fail("a kept map reads no audio"))
    assert kit.load_or_learn(tmp_path, drums, activations, attacks, [], np.array([])) == learned
    monkeypatch.undo()
    # The same sounds, and a new one: a rebuilt map keeps the old ids and numbers the new sound after them.
    more = 0.5 + 0.8 * np.arange(40, 55)
    stem(drums, [(t, LOW if i % 2 == 0 else HIGH, 1.0) for i, t in enumerate(times)] + [(t, BRIGHT, 1.0) for t in more], seconds=50)
    rebuilt = kit.load_or_learn(tmp_path, drums, activations, attacks, hits + [Hit(float(t), "kick", 0.9) for t in more], times)
    assert rebuilt != learned and [g.id for g in rebuilt.groups] == ["kick-1", "kick-2", "kick-3"]
    assert [g.count for g in rebuilt.groups] == [20, 20, 15]
    assert np.allclose(rebuilt.groups[0].centre, learned.groups[0].centre, atol=0.05)


def test_bad_saved_maps_are_learned_again():
    assert kit.SoundMap.load(__import__("pathlib").Path("/nonexistent")) is None
    with pytest.raises(ValueError, match="version"):
        kit.SoundMap.from_dict({"version": 2, "groups": [], "inputs": {}})
    group = dict(id="kick-1", detected="kick", count=1, mixed=0, centre=[1.0], examples=[], similar={})
    with pytest.raises(ValueError, match="unique"):
        kit.SoundMap.from_dict({"version": 1, "groups": [group, group], "inputs": {}})


def test_similarity_says_which_class_a_group_sounds_like(tmp_path):
    times = 0.5 + 0.8 * np.arange(60)
    kinds = ["kick", "snare", "tom"]
    strokes = [(t, {"kick": LOW, "snare": HIGH, "tom": HIGH}[kinds[i % 3]], 1.0) for i, t in enumerate(times)]
    drums = stem(tmp_path / "drums.wav", strokes, seconds=50)
    hits = [Hit(float(t), kinds[i % 3], 0.9) for i, t in enumerate(times)]
    learned = kit.learn(drums, hits, times, {})
    by_class = {g.detected: g for g in learned.groups}
    assert len(learned.groups) == 3
    assert by_class["tom"].similar["snare"] > 0.95 > by_class["tom"].similar["kick"]
    assert set(by_class["kick"].similar) == {"kick", "snare", "tom"} and by_class["kick"].similar["kick"] == 1.0


def test_strokes_struck_together_are_counted_but_never_shown(tmp_path):
    times = 0.5 + 0.8 * np.arange(30)
    strokes = [(t, LOW, 1.0) for t in times] + [(t + 0.01, BRIGHT, 1.0) for t in times[:10]]
    drums = stem(tmp_path / "drums.wav", strokes)
    hits = sorted([Hit(float(t), "kick", 0.9) for t in times] + [Hit(float(t + 0.01), "hihat", 0.9) for t in times[:10]], key=lambda h: h.time)
    learned = kit.learn(drums, hits, np.array([]), {})
    kick = next(g for g in learned.groups if g.detected == "kick")
    assert (kick.count, kick.mixed) == (30, 10)
    assert all(e.time >= times[10] for e in kick.examples)


def gaussians(n, seeds=20):
    return [np.random.default_rng(seed).normal(0, 0.01, (n, 32)) + 1 for seed in range(seeds)]


def test_one_gaussian_sound_rarely_splits_and_never_once_it_has_many_strokes():
    # Held out, the split index of one Gaussian is 1.33 on average; with few strokes it is noisy.
    assert sum(len(kit._centres(one)) > 1 for one in gaussians(100)) <= 4
    assert sum(len(kit._centres(one)) > 1 for one in gaussians(400)) == 0
    assert len(kit._centres(gaussians(2000, 1)[0])) == 1


def test_two_sounds_split_but_a_few_odd_strokes_do_not():
    one = gaussians(400, 1)[0]
    two = np.concatenate([one, one + 0.03 * np.r_[np.ones(16), -np.ones(16)]])
    assert len(kit._centres(two)) == 2
    assert len(kit._centres(np.concatenate([one, one[:5] + 3]))) == 1  # five odd strokes are not a sound
    assert len(kit._centres(one[:5])) == 1
    assert len(kit._centres(np.ones((50, 32)))) == 1  # identical strokes


def test_short_or_silent_stems_give_no_groups(tmp_path):
    silent = tmp_path / "silence.wav"
    sf.write(silent, np.zeros((SR, 2)), SR)
    assert kit.learn(silent, [], np.array([]), {}).groups == []
    assert kit.learn(silent, [Hit(5.0, "kick", 0.9)], np.array([5.0, -1.0, np.nan]), {}).groups == []
    assert kit.rename([Hit(5.0, "kick", 0.9)], kit.SoundMap([], {}), {"kick-1": "snare"}, silent, np.array([])) == [Hit(5.0, "kick", 0.9)]
