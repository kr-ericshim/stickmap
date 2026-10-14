"""Synthetic positive-bag loss, crop masks and channel/song exclusion tests."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train  # noqa: E402
from report_v3 import report  # noqa: E402


@pytest.fixture
def song_factory(tmp_path):
    def make(slug="synthetic", channel="drumshin", labels=None, dense=True, frames=120,
             withheld=(), teacher=0.05, events=None, **dense_options):
        folder = tmp_path / slug
        folder.mkdir()
        meta = {"slug": slug, "channel": channel, "intervals": [[0, 0.2]],
                "events": events or {c: [0.1] if c == "kick" else [] for c in train.CLASSES},
                "withheld_heads": list(withheld)}
        (folder / "truth.json").write_text(json.dumps(meta))
        np.save(folder / "feat.npy", np.zeros((frames, 84), np.float32))
        teacher = np.full((frames, 5), teacher, np.float32) if np.isscalar(teacher) else teacher
        np.save(folder / "teacher0.npy", teacher)
        if labels is not None:
            (folder / "dense.json").write_text(json.dumps({
                "fps": 100, "tolerance_frames": 2, "positive_only": True, "labels": labels}))
        return train.Song(folder, None, "cpu", dense=dense, **dense_options)
    return make


LABELS = [{"time": 0.5, "class": "kick"}, {"time": 0.9, "class": "tom"}]


def fixed_batch(monkeypatch, song, window, length=None, dense_mode="peak"):
    monkeypatch.setattr(train, "crop", lambda *args: window)
    return train.batch([song], np.ones(1), 1, length or window.stop - window.start,
                       np.random.default_rng(0), "cpu", dense_mode)


def test_dense_adds_no_frame_targets_or_negatives(song_factory):
    base = song_factory("base", labels=LABELS, dense=False)
    dense = song_factory("dense", labels=LABELS)
    for attr in ("target", "truth_w", "kd_w", "inside", "inside_frames", "fill_frames"):
        np.testing.assert_array_equal(getattr(base, attr), getattr(dense, attr))
    np.testing.assert_array_equal(dense.dense_windows, [[48, 53, 0], [88, 93, 2]])
    assert not dense.truth_w[21:].any()
    assert not base.dense_windows.size


def test_whole_windows_mask_only_the_labelled_class(song_factory, monkeypatch):
    song = song_factory(labels=LABELS)
    window = slice(48, 93)  # both inclusive windows fit exactly at the crop edges
    _, y, m, kw, teacher, labels = fixed_batch(monkeypatch, song, window, dense_mode="window")
    assert labels == [(0, 0, 5, 0), (0, 40, 45, 2)]
    expected = song.kd_w[window].copy()
    expected[:5, 0] = expected[40:45, 2] = 0
    np.testing.assert_array_equal(kw[0].numpy(), expected)
    assert torch.all(kw[0, :, (1, 3, 4)] == 1)
    assert not y.any() and not m.any()
    assert torch.all(teacher == 0.05)
    assert np.all(song.kd_w[window] == 1)  # batch must not mutate the source song


def test_partial_windows_are_neither_labels_nor_masks(song_factory, monkeypatch):
    song = song_factory(labels=LABELS)
    window = slice(49, 92)
    _, _, _, kw, _, labels = fixed_batch(monkeypatch, song, window)
    assert labels == []
    np.testing.assert_array_equal(kw[0].numpy(), song.kd_w[window])


def test_short_song_padding_is_unknown(song_factory, monkeypatch):
    song = song_factory(labels=LABELS)
    x, y, m, kw, teacher, labels = fixed_batch(monkeypatch, song, slice(0, 200), 200)
    assert x.shape == (1, 200, 84, 1)
    assert labels == [(0, 48, 53, 0), (0, 88, 93, 2)]
    for tensor in (y, m, kw, teacher):
        assert not tensor[:, 120:].any()


def test_dense_loss_is_weighted_mean_of_class_window_maxima():
    out = torch.full((2, 10, 5), 0.01)
    labels = [(0, 1, 4, 0), (0, 5, 8, 2), (1, 0, 3, 1)]
    out[0, 3, 0], out[0, 7, 2], out[1, 0, 1] = 0.8, 0.4, 0.2
    out.requires_grad_()
    zero = torch.zeros_like(out)
    loss = train.loss_fn(out, zero, zero, zero, zero, 0.3, labels, 1.7)
    assert loss.item() == pytest.approx(-1.7 * sum(map(math.log, (0.8, 0.4, 0.2))) / 3)
    loss.backward()
    expected = torch.zeros_like(out)
    expected[0, 3, 0], expected[0, 7, 2], expected[1, 0, 1] = -1.7 / 2.4, -1.7 / 1.2, -1.7 / 0.6
    torch.testing.assert_close(out.grad, expected)
    changed = out.detach().clone()
    changed[1, 9, :] = 0.99  # unlabelled classes/frames never add dense negatives
    torch.testing.assert_close(train.loss_fn(changed, zero, zero, zero, zero, 0.3, labels, 1.7), loss.detach())


def test_dense_term_adds_to_existing_truth_and_kd():
    out = torch.full((1, 10, 5), 0.4)
    target, teacher = torch.zeros_like(out), torch.full_like(out, 0.2)
    truth_w, kd_w = torch.ones_like(out), torch.ones_like(out)
    kd_w[:, 2:5, 2] = 0
    base = train.loss_fn(out, target, truth_w, kd_w, teacher, 0.3)
    assert base.item() == pytest.approx(-math.log(0.6) + 0.3 * (-0.2 * math.log(0.4) - 0.8 * math.log(0.6)))
    labels = [(0, 2, 5, 2)]
    dense = train.loss_fn(out, target, truth_w, kd_w, teacher, 0.3, labels, 2.0, "window")
    assert (dense - base).item() == pytest.approx(-2 * math.log(0.4))
    torch.testing.assert_close(train.loss_fn(out, target, truth_w, kd_w, teacher, 0.3, labels, 0), base)
    torch.testing.assert_close(train.loss_fn(out, target, truth_w, kd_w, teacher, 0.3, [], 2), base)


def test_empty_bags_have_finite_loss_and_gradients():
    out = torch.full((1, 10, 5), 0.4, requires_grad=True)
    zero = torch.zeros_like(out)
    loss = train.loss_fn(out, zero, zero, zero, zero, 0.3, [], 1)
    loss.backward()
    assert loss.item() == 0 and torch.isfinite(out.grad).all()
    assert not out.grad.any()


def test_quantization_and_per_label_tolerance(song_factory):
    song = song_factory(labels=[{"time": 0.516, "class": "snare", "tolerance_frames": 1}])
    np.testing.assert_array_equal(song.dense_windows, [[51, 54, 1]])
    np.testing.assert_array_equal(song.dense_frames, [52])


@pytest.mark.parametrize("labels", [None, []])
def test_missing_or_empty_dense_keeps_original_sampling(song_factory, labels):
    song = song_factory(labels=labels)
    base = song_factory("base", labels=labels, dense=False)
    rng, baseline_rng = np.random.default_rng(4), np.random.default_rng(4)
    assert [train.crop(song, 30, rng) for _ in range(100)] == [train.crop(base, 30, baseline_rng) for _ in range(100)]


class FixedRng:
    def __init__(self, r):
        self.r = r
        self.chosen = None

    def random(self):
        return self.r

    def choice(self, values):
        self.chosen = values
        return values[0]

    def integers(self, upper):
        return upper // 2


@pytest.mark.parametrize("r,bucket", [(0.0, "fill_frames"), (0.399, "fill_frames"),
                                     (0.4, "dense_frames"), (0.599, "dense_frames"),
                                     (0.6, "inside_frames"), (0.899, "inside_frames"),
                                     (0.9, None)])
def test_crop_buckets_keep_fills_and_reserve_twenty_percent(song_factory, r, bucket):
    song = song_factory(labels=LABELS)
    song.fill_frames = np.array([30])
    rng = FixedRng(r)
    window = train.crop(song, 20, rng)
    if bucket is None:
        assert rng.chosen is None
    else:
        assert rng.chosen is getattr(song, bucket)
    if bucket == "dense_frames":
        assert window == slice(40, 60)


@pytest.mark.parametrize("fold", ["0", "1", "2"])
def test_channel_and_same_song_exclusions_apply_to_dense_songs(song_factory, fold):
    held_channel = train.FOLDS[int(fold)][0]
    other_channel = train.FOLDS[(int(fold) + 1) % 3][0]
    held = song_factory("held-love-me-or-leave-me", held_channel, LABELS)
    duplicate = song_factory("other-love-me-or-leave-me", other_channel, LABELS)
    eligible = song_factory("different-song", other_channel, LABELS)
    training, held_out = train.split_songs([held, duplicate, eligible], fold)
    assert training == [eligible] and held_out == [held]
    assert train.split_songs([held, duplicate, eligible], "all") == ([held, duplicate, eligible], [])


@pytest.mark.parametrize("change", [{"fps": 50}, {"positive_only": False},
                                    {"labels": [{"time": 0, "class": "kick"}]},
                                    {"labels": [{"time": 0.5, "class": "hihat"}]}])
def test_invalid_dense_metadata_fails_before_training(song_factory, change):
    song = song_factory(labels=LABELS, dense=False)
    file = song.folder / "dense.json"
    meta = json.loads(file.read_text())
    meta.update(change)
    file.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        train.Song(song.folder, None, "cpu", dense=True)


@pytest.fixture
def score_totals(tmp_path):
    for tag, extra in (("v2", 0), ("v3", 1)):
        global_totals = {c: [8 + extra, 10 + extra * 2, 20] for c in train.CLASSES[:3]}
        (tmp_path / f"{tag}-global.json").write_text(json.dumps(global_totals))
        totals = {f"{key}.{c}.{field}": value + (extra if field != "ref" else 0)
                  for key, counts in (("all", (4, 5, 10)), ("fast", (2, 3, 4)))
                  for c in train.CLASSES[:3] for field, value in zip(("tp", "est", "ref"), counts)}
        for scorer in ("covers", "solos"):
            (tmp_path / f"{tag}-{scorer}.json").write_text(json.dumps(totals))
    return tmp_path


def test_report_uses_integer_totals_and_correct_fast_keys(score_totals):
    output = score_totals / "comparison.md"
    report(score_totals, output)
    text = output.read_text()
    assert "| All harvested songs | tom | 20 | 0.8000 / 0.4000 | 0.7500 / 0.4500 | +0.0292 |" in text
    assert "| Cover fills, gap < 90 ms | tom | 4 | 0.6667 / 0.5000 | 0.7500 / 0.7500 | +0.1786 |" in text
    assert "| Solo passages, gap < 90 ms | tom | 4 |" in text
    assert "[global.json](v3-global.json)" in text


@pytest.mark.parametrize("ref", [0, 21])
def test_report_rejects_empty_or_mismatched_evaluations(score_totals, ref):
    file = score_totals / "v3-global.json"
    totals = json.loads(file.read_text())
    totals["kick"][2] = ref
    file.write_text(json.dumps(totals))
    output = score_totals / "comparison.md"
    with pytest.raises(ValueError):
        report(score_totals, output)
    assert not output.exists()


def test_peak_batch_keeps_all_kd_until_student_argmax(song_factory, monkeypatch):
    song = song_factory(labels=LABELS)
    _, _, _, kw, _, labels = fixed_batch(monkeypatch, song, slice(48, 93))
    assert len(labels) == 2
    np.testing.assert_array_equal(kw[0].numpy(), song.kd_w[48:93])


@pytest.mark.parametrize('peak', [2, 5, 8])
def test_peak_masks_only_local_class_negatives_and_kd(peak):
    out = torch.full((1, 12, 5), 0.4)
    out[0, peak, 2] = 0.8
    out.requires_grad_()
    target = torch.zeros_like(out)
    # A verified positive next to the selected peak must survive the negative mask.
    positive = peak + 1 if peak < 8 else peak - 1
    target[0, positive, 2] = 1
    weights = torch.ones_like(out)
    teacher = torch.full_like(out, 0.05)
    loss = train.loss_fn(out, target, weights, weights, teacher, .3, [(0, 2, 9, 2)], .3)
    loss.backward()
    lo, hi = max(2, peak - 1), min(9, peak + 2)
    for frame in range(12):
        g = out.grad[0, frame, 2]
        if frame == peak or frame == positive:
            assert g < 0
        elif lo <= frame < hi:
            assert g == 0
        else:
            assert g > 0  # surrounding probabilities are still pushed down
    assert torch.all(out.grad[0, :, (0, 1, 3, 4)] > 0)
    assert torch.all(weights == 1)  # loss must not change the batch/song weights


def test_peak_loss_penalizes_a_broad_plateau_and_breaks_ties():
    narrow = torch.full((1, 10, 5), .05)
    narrow[0, 5, 1] = .8
    broad = narrow.clone()
    broad[0, 2:9, 1] = .8
    zeros = torch.zeros_like(narrow)
    kd = torch.ones_like(narrow)
    teacher = torch.full_like(narrow, .05)
    bags = [(0, 2, 9, 1)]
    narrow_loss = train.loss_fn(narrow, zeros, zeros, kd, teacher, .3, bags, .3)
    broad_loss = train.loss_fn(broad, zeros, zeros, kd, teacher, .3, bags, .3)
    assert broad_loss > narrow_loss
    broad.requires_grad_()
    train.loss_fn(broad, zeros, zeros, zeros, teacher, .3, bags, .3).backward()
    assert broad.grad[0, 2, 1] < 0  # first argmax only, not every tied frame
    assert torch.count_nonzero(broad.grad) == 1


@pytest.mark.parametrize('teacher', [.05, .8])
def test_withheld_masks_quiet_and_doubt_negatives_but_keeps_positives(song_factory, monkeypatch, teacher):
    monkeypatch.setattr(train, 'DOUBT', .3)
    base = song_factory('base', dense=False, teacher=teacher)
    song = song_factory('withheld', dense=False, teacher=teacher,
                        withheld=[{'time': .1, 'class': 'kick'}, {'time': .17, 'class': 'tom'}])
    assert song.target[10, 0] == song.truth_w[10, 0] == 1
    assert not song.truth_w[14:21, 2].any()
    assert base.truth_w[14:21, 2].all()
    assert song.truth_w[13, 2] == base.truth_w[13, 2]
    np.testing.assert_array_equal(song.kd_w, base.kd_w)
    np.testing.assert_array_equal(song.truth_w[:, 1], base.truth_w[:, 1])


def test_fold_cache_required_and_quality_scope_verified(song_factory):
    song = song_factory(labels=LABELS, dense=False)
    with pytest.raises(FileNotFoundError):
        train.Song(song.folder, None, 'cpu', dense=True, dense_fold='0')
    meta = json.loads((song.folder / 'dense.json').read_text())
    file = song.folder / 'dense-fold0.json'
    file.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='training-only'):
        train.Song(song.folder, None, 'cpu', dense=True, dense_fold='0')
    meta.update(quality_fold='0', reference_scope='training-only')
    file.write_text(json.dumps(meta))
    assert len(train.Song(song.folder, None, 'cpu', dense=True, dense_fold='0').dense_windows) == 2


def test_new_channels_are_unique_and_mixed_nuts_is_excluded():
    from splits import ORIGINAL_CHANNELS
    assert len(ORIGINAL_CHANNELS) == 14
    assert len([c for f in train.FOLDS for c in f]) == len({c for f in train.FOLDS for c in f}) == 18
    rows = [dict(channel='moyin', slug='moyin-SRKKu5mTeAE'),
            dict(channel='nakamura', slug='nakamura-DqSJyUZiHIk'),
            dict(channel='golly', slug='another-song')]
    training, held = train.split_songs(rows, '1')
    assert training == [rows[2]] and held == rows[:2]
    assert train.same_song(rows[0]['slug'], rows[1]['slug'])
    # Exercise the ID exclusion itself even if a later assignment separates the pair.
    rows[0]['channel'] = 'swartz'
    training, held = train.split_songs(rows, '1')
    assert training == [rows[2]] and held == [rows[1]]
    assert not train.same_song(rows[0]['slug'], 'unrelated-mixed-nuts-ABCDEFGHIJK')
    with pytest.raises(ValueError, match='missing from FOLDS'):
        train.split_songs([dict(channel='new-unassigned', slug='song')], '0')


def test_report_v4_uses_actual_candidate_totals(score_totals):
    for scorer in ('global', 'covers', 'solos'):
        (score_totals / f'v4-{scorer}.json').write_bytes((score_totals / f'v3-{scorer}.json').read_bytes())
    output = score_totals / 'v4.md'
    report(score_totals, output, candidate='v4')
    text = output.read_text()
    assert 'original 14 channels' in text and 'v4 P / R' in text
    assert '| All harvested songs | tom | 20 | 0.8000 / 0.4000 | 0.7500 / 0.4500 | +0.0292 |' in text
    assert '[global.json](v4-global.json)' in text


def test_legacy_withheld_switch_reproduces_old_negative_weights(song_factory, monkeypatch):
    monkeypatch.setattr(train, 'DOUBT', .3)
    base = song_factory('base', dense=False, teacher=.8)
    masked = song_factory('partial', dense=False, teacher=.8,
                          withheld=[{'time': .17, 'class': 'tom'}])
    legacy = train.Song(masked.folder, None, 'cpu', withheld_mask=False)
    np.testing.assert_array_equal(legacy.truth_w, base.truth_w)
    assert not masked.truth_w[14:21, 2].any()


@pytest.mark.parametrize("source", ["strict", "dense", "teacher"])
def test_exclusive_snare_positive_cannot_land_on_neighboring_tom(song_factory, monkeypatch, source):
    labels = [{"time": .5, "class": "snare", "tolerance_frames": 3,
               "system": 0, "moment": 0}]
    events = {c: [.5] if c == "tom" and source == "strict" else [] for c in train.CLASSES}
    teacher = np.full((120, 5), .05, np.float32)
    if source == "teacher":
        teacher[50, 2] = .9
    if source == "dense":
        # Same estimated frame but a different notated moment, hence no unison exemption.
        labels.append({"time": .5, "class": "tom", "tolerance_frames": 0,
                       "system": 0, "moment": 1})
    song = song_factory(labels=labels, events=events, teacher=teacher, dense_mode="exclusive")
    _, y, m, kw, t, bags = fixed_batch(monkeypatch, song, slice(47, 54), dense_mode="exclusive")
    snare = next(b for b in bags if b.k == 1)
    assert snare.candidates == (0, 6)  # tom +/-2 is protected, both endpoints remain usable
    out = torch.full_like(y, .1)
    out[0, 3, 1] = .95  # the tempting shortcut, exactly on the tom
    out[0, 6, 1] = .7
    out.requires_grad_()
    train.loss_fn(out, y, m, kw, t, .3, bags, .3, "exclusive").backward()
    assert out.grad[0, 6, 1] < 0  # snare positive is assigned to an eligible frame
    assert out.grad[0, 3, 1] > 0  # original KD still pushes down the shortcut
    assert out.grad[0, 5, 1] > 0  # even the blocked neighbor of the selected peak keeps KD
    assert out.grad[0, 6, 2] > 0  # competing class loss survives at the selected snare frame
    np.testing.assert_array_equal(kw[0].numpy(), song.kd_w[47:54])


def test_exclusive_tom_positive_cannot_land_on_kick(song_factory, monkeypatch):
    teacher = np.full((120, 5), .05, np.float32)
    teacher[50, 0] = .9
    song = song_factory(labels=[{"time": .5, "class": "tom", "tolerance_frames": 3}],
                        teacher=teacher, dense_mode="exclusive")
    _, y, m, kw, t, bags = fixed_batch(monkeypatch, song, slice(47, 54), dense_mode="exclusive")
    out = torch.full_like(y, .1)
    out[0, 3, 2], out[0, 0, 2] = .95, .7
    out.requires_grad_()
    train.loss_fn(out, y, m, kw, t, .3, bags, .3, "exclusive").backward()
    assert out.grad[0, 3, 2] > 0 and out.grad[0, 0, 2] < 0


def test_exclusive_empty_candidates_skip_loss_and_dense_sampling(song_factory, monkeypatch):
    teacher = np.full((120, 5), .05, np.float32)
    teacher[47:54, 2] = .9
    song = song_factory(labels=[{"time": .5, "class": "snare", "tolerance_frames": 3}],
                        teacher=teacher, dense_mode="exclusive")
    assert not song.dense_candidates[0].size and not song.dense_frames.size
    base = song_factory("base", dense=False, teacher=teacher)
    rng, base_rng = np.random.default_rng(4), np.random.default_rng(4)
    assert [train.crop(song, 30, rng) for _ in range(100)] == [train.crop(base, 30, base_rng) for _ in range(100)]
    _, y, m, kw, t, bags = fixed_batch(monkeypatch, song, slice(47, 54), dense_mode="exclusive")
    assert bags == []
    out = torch.full_like(y, .4, requires_grad=True)
    loss = train.loss_fn(out, y, m, kw, t, .3, bags, .3, "exclusive")
    torch.testing.assert_close(loss, train.loss_fn(out, y, m, kw, t, .3))
    loss.backward()
    assert torch.isfinite(out.grad).all() and torch.all(out.grad[:, :, 1] > 0)
    assert torch.all(out.grad[:, :, 2] < 0)  # KD follows the strong tom, not a dense snare positive
    assert train.dense_eligibility([song])["snare"] == {
        "labels": 1, "usable": 0, "skipped": 1, "candidate_frames": 0}


def test_exclusive_dense_unisons_override_teacher_but_not_unrelated_strokes(song_factory):
    labels = [{"time": .5, "class": c, "tolerance_frames": 3,
               "system": 1, "moment": 2} for c in ("snare", "tom")]
    teacher = np.full((120, 5), .05, np.float32)
    teacher[47:54, 2] = .9
    song = song_factory(labels=labels, teacher=teacher, dense_mode="exclusive")
    np.testing.assert_array_equal(song.dense_candidates[0], np.arange(47, 54))
    labels.append({"time": .55, "class": "tom", "tolerance_frames": 0,
                   "system": 1, "moment": 3})
    other = song_factory("other", labels=labels, teacher=teacher, dense_mode="exclusive")
    np.testing.assert_array_equal(other.dense_candidates[0], np.arange(47, 53))


def test_exclusive_strict_unisons_keep_verified_positives(song_factory, monkeypatch):
    teacher = np.full((120, 5), .05, np.float32)
    teacher[49:52, 2] = .9
    events = {c: [.5] if c in ("snare", "tom") else [] for c in train.CLASSES}
    song = song_factory(labels=[{"time": .5, "class": "snare", "tolerance_frames": 3}],
                        teacher=teacher, events=events, dense_mode="exclusive")
    np.testing.assert_array_equal(song.dense_candidates[0], np.arange(47, 54))
    _, y, m, kw, t, bags = fixed_batch(monkeypatch, song, slice(47, 54), dense_mode="exclusive")
    out = torch.full_like(y, .1)
    out[0, 3, 1] = .8
    out.requires_grad_()
    train.loss_fn(out, y, m, kw, t, .3, bags, .3, "exclusive").backward()
    assert out.grad[0, 3, 1] < 0 and out.grad[0, 3, 2] < 0
    assert m[0, 3, 1] == m[0, 3, 2] == 1  # weights are not mutated


def test_exclusive_quantization_and_window_overlap_do_not_imply_simultaneous(song_factory):
    labels = [{"time": .5, "class": "snare", "tolerance_frames": 3},
              {"time": .504, "class": "tom", "tolerance_frames": 0}]
    song = song_factory(labels=labels, dense_mode="exclusive")
    np.testing.assert_array_equal(song.dense_candidates[0], [47, 53])
    assert not train.simultaneous(*labels)


def test_exclusive_competing_bags_and_teacher_outside_crop_still_guard_candidates(song_factory, monkeypatch):
    labels = [{"time": .5, "class": "snare", "tolerance_frames": 3},
              {"time": .55, "class": "tom", "tolerance_frames": 2}]
    song = song_factory(labels=labels, dense_mode="exclusive")
    _, _, _, _, _, bags = fixed_batch(monkeypatch, song, slice(47, 54), dense_mode="exclusive")
    assert len(bags) == 1 and bags[0].candidates == (0, 1, 2, 3)
    teacher = np.full((120, 5), .05, np.float32)
    teacher[55, 2] = .5
    song = song_factory("teacher", labels=labels[:1], teacher=teacher, dense_mode="exclusive")
    _, _, _, _, _, bags = fixed_batch(monkeypatch, song, slice(47, 54), dense_mode="exclusive")
    assert bags[0].candidates == tuple(range(6))  # frame 53 sees teacher frame 55


def test_exclusive_radius_never_widens_bags_or_narrows_competitor_protection(song_factory):
    labels = [{"time": .5, "class": "snare", "tolerance_frames": 3},
              {"time": .55, "class": "tom", "tolerance_frames": 2}]
    song = song_factory(labels=labels, dense_mode="exclusive", dense_radius=1)
    np.testing.assert_array_equal(song.dense_candidates[0], [49, 50])  # frame 51 remains blocked
    wide = song_factory("wide", labels=labels[:1], dense_mode="exclusive", dense_radius=20)
    np.testing.assert_array_equal(wide.dense_candidates[0], np.arange(47, 54))
    np.testing.assert_array_equal(song.dense_windows, [[47, 54, 1], [53, 58, 2]])


def test_exclusive_teacher_threshold_is_inclusive_and_only_competes_learned_classes(song_factory):
    labels = [{"time": .5, "class": "snare", "tolerance_frames": 3}]
    teacher = np.full((120, 5), .05, np.float32)
    teacher[:, 3:] = .99  # a simultaneous hat/cymbal must not censor hand/foot supervision
    teacher[50, 2] = .49
    song = song_factory(labels=labels, teacher=teacher, dense_mode="exclusive")
    np.testing.assert_array_equal(song.dense_candidates[0], np.arange(47, 54))
    teacher[50, 2] = .5
    song = song_factory("boundary", labels=labels, teacher=teacher, dense_mode="exclusive", dense_guard_frames=0)
    np.testing.assert_array_equal(song.dense_candidates[0], [47, 48, 49, 51, 52, 53])


def test_exclusive_downweights_snare_bce_without_renormalizing_other_classes():
    out = torch.full((1, 4, 5), .4, requires_grad=True)
    zero = torch.zeros_like(out)
    bags = [train.ExclusiveBag(0, 0, 2, 1, (0, 1)), train.ExclusiveBag(0, 2, 4, 2, (2, 3))]
    loss = train.loss_fn(out, zero, zero, zero, zero, .3, bags, .3, "exclusive", .25)
    assert loss.item() == pytest.approx(-.3 * (1 + .25) * math.log(.4) / 2)
    loss.backward()
    assert out.grad[0, 0, 1].item() == pytest.approx(.25 * out.grad[0, 2, 2].item())
    assert torch.count_nonzero(out.grad) == 2  # first eligible argmax on ties


def test_exclusive_disabled_snare_still_guards_tom_and_adds_no_frame_targets(song_factory, monkeypatch):
    labels = [{"time": .5, "class": "snare", "tolerance_frames": 0},
              {"time": .52, "class": "tom", "tolerance_frames": 3}]
    song = song_factory(labels=labels, dense_mode="exclusive", dense_snare_weight=0)
    assert not song.dense_candidates[0].size
    np.testing.assert_array_equal(song.dense_candidates[1], [53, 54, 55])
    base = song_factory("base", labels=labels, dense=False)
    for attr in ("target", "truth_w", "kd_w", "inside", "inside_frames", "fill_frames"):
        np.testing.assert_array_equal(getattr(base, attr), getattr(song, attr))
    _, _, _, _, _, bags = fixed_batch(monkeypatch, song, slice(0, 120), dense_mode="exclusive")
    assert len(bags) == 1 and bags[0].k == 2


@pytest.mark.parametrize("labels", [None, []])
def test_exclusive_missing_or_empty_legacy_labels_are_safe(song_factory, labels):
    song = song_factory(labels=labels, dense_mode="exclusive")
    assert not song.dense_frames.size and song.dense_candidates == []


def test_exclusive_cli_variants_are_tagged_and_defaults_preserve_peak():
    assert train.parse_args([]).dense_mode == "peak"
    args = train.parse_args(["--tag", "v5ex_s025", "--dense", ".3", "--dense-mode", "exclusive",
                             "--dense-radius", "1", "--dense-snare-weight", ".25"])
    assert args.tag == "v5ex_s025" and args.dense == .3 and args.dense_snare_weight == .25
    assert args.dense_radius == 1 and args.dense_guard_frames == 2 and args.dense_teacher_threshold == .5


@pytest.mark.parametrize("flags", [
    ["--dense-guard-frames", "-1"], ["--dense-radius", "-1"],
    ["--dense-teacher-threshold", "0"], ["--dense-teacher-threshold", "1.1"],
    ["--dense-teacher-threshold", "nan"], ["--dense-snare-weight", "-1"],
    ["--dense-snare-weight", "inf"],
])
def test_exclusive_cli_rejects_invalid_controls(flags):
    with pytest.raises(SystemExit):
        train.parse_args(["--dense-mode", "exclusive", *flags])


@pytest.mark.parametrize("mode", ["peak", "window"])
def test_legacy_modes_reject_exclusive_controls_instead_of_silently_ignoring_them(mode):
    with pytest.raises(SystemExit):
        train.parse_args(["--dense-mode", mode, "--dense-radius", "1"])
