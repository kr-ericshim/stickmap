"""Fine-tune ADTOF's Frame-RNN on the harvested transcription truth (cached by data.py).

Kick, snare and tom are learned from the truth inside the accepted intervals;
everywhere else, and for hi-hat and cymbal (which many channels never write),
the model is held to what the original model said (distillation), so it does
not forget what it was good at. Songs are split by channel into folds: a fold
is trained on the other channels and predicts its own songs, so every song is
scored by a model that never saw its channel.

    .venv/bin/python tools/finetune/train.py --fold 0           # train on folds 1,2; predict fold 0
    .venv/bin/python tools/finetune/train.py --fold all         # train on everything (deployment)
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from splits import FOLDS, SAME_SONG, same_song, split_songs
import vault

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".eval-cache" / "ft"
MODELS = CACHE / "models"
CLASSES = ("kick", "snare", "tom", "hihat", "cymbal")
LEARNED = (0, 1, 2)  # classes taught by the truth
FPS = 100
THRESHOLDS = (0.22, 0.24, 0.32, 0.22, 0.30)
DOUBT = 0.0  # weight of "nobody wrote a kick/tom here" where the original model hears one (--doubt)


def original_model(device: str) -> torch.nn.Module:
    import adtof_pytorch as adtof

    model = adtof.create_frame_rnn_model(adtof.calculate_n_bins())
    return adtof.load_pytorch_weights(model, adtof.get_default_weights_path(), strict=False).to(device)


def run(model: torch.nn.Module, feat: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(feat.astype(np.float32))[None, :, :, None].to(device)
        return model(x).cpu().numpy()[0]


@dataclass(frozen=True)
class ExclusiveBag:
    sample: int
    first: int
    stop: int
    k: int
    candidates: tuple[int, ...]  # crop-relative frames; never empty


def expand_frames(mask: np.ndarray, radius: int) -> np.ndarray:
    """Whether a marked frame is within an inclusive +/-radius neighborhood."""
    frames = np.arange(len(mask))
    sums = np.r_[0, np.cumsum(mask, dtype=int)]
    return sums[np.minimum(len(mask), frames + radius + 1)] > sums[np.maximum(0, frames - radius)]


def simultaneous(a: dict, b: dict) -> bool:
    """Equal quantized frames or overlapping bags alone do not prove a unison."""
    return (a["time"] == b["time"] and
            all(a.get(key) == b.get(key) for key in ("system", "moment")))


def exclusive_candidates(teacher: np.ndarray, target: np.ndarray, labels: list[dict],
                         windows: np.ndarray, guard: int = 2, threshold: float = 0.5,
                         radius: int | None = None) -> list[np.ndarray]:
    """Static, teacher/notation-only eligibility; never consult student activations.

    Competing dense bags protect their entire original uncertainty window, plus
    the guard. Only explicitly shared moments exempt that competitor. Teacher
    competition is between kick/snare/tom: hats/cymbals are incomplete in this
    notation tier and commonly accompany a legitimate hand/foot stroke.
    """
    n = len(teacher)
    written = target > 0
    unisons = {(k, other): expand_frames(written[:, k] & written[:, other], 1)
               for k in LEARNED for other in LEARNED if k != other}
    candidates = []
    for label, (first, stop, k) in zip(labels, windows):
        lo, hi = max(0, first - guard), min(n, stop + guard)
        blocked = np.zeros(hi - lo, bool)
        for other in LEARNED:
            if other == k:
                continue
            exempt = unisons[k, other][lo:hi].copy()
            competing = written[lo:hi, other] & ~written[lo:hi, k]
            for other_label, (a, b, cls) in zip(labels, windows):
                if cls != other:
                    continue
                a, b = max(lo, a), min(hi, b)
                if a >= b:
                    continue
                if simultaneous(label, other_label):
                    # Only the shared support, not neighboring unrelated strokes.
                    a, b = max(first, a), min(stop, b)
                    exempt[a - lo:b - lo] = True
                else:
                    competing[a - lo:b - lo] = True
            competing |= (teacher[lo:hi, other] >= threshold) & ~exempt
            blocked |= competing
        allowed = ~expand_frames(blocked, guard)[first - lo:stop - lo]
        frames = np.arange(first, stop)
        if radius is not None:
            allowed &= np.abs(frames - round(label["time"] * FPS)) <= radius
        candidates.append(frames[allowed])
    return candidates


class Song:
    def __init__(self, folder: Path, teacher_model: torch.nn.Module, device: str, dense: bool = False,
                 dense_fold: str | None = None, withheld_mask: bool = True,
                 dense_mode: str = "peak", dense_guard_frames: int = 2,
                 dense_teacher_threshold: float = 0.5, dense_radius: int | None = None,
                 dense_snare_weight: float = 1.0):
        meta = json.loads((folder / "truth.json").read_text())
        if vault.is_vault_song(meta):
            raise ValueError(f"sealed vault song cannot be loaded for training: {meta['slug']}")
        self.slug, self.channel = meta["slug"], meta["channel"]
        self.video_id = meta.get("video_id")
        self.folder = folder
        self.feat = np.load(folder / "feat.npy").astype(np.float32)
        cached = folder / "teacher0.npy"  # the original model on exactly these features
        if not cached.exists():
            np.save(cached, run(teacher_model, self.feat, device).astype(np.float16))
        self.teacher = np.load(cached).astype(np.float32)
        n = len(self.feat)
        self.target = np.zeros((n, 5), np.float32)
        self.truth_w = np.zeros((n, 5), np.float32)  # weight of the truth loss
        self.kd_w = np.ones((n, 5), np.float32)  # weight of following the original model
        inside = np.zeros(n, bool)
        for a, b in meta["intervals"]:
            inside[max(0, int(round(a * FPS))):min(n, int(round(b * FPS)) + 1)] = True
        self.inside = inside
        self.events = {}
        for k in LEARNED:
            frames = np.array(sorted({round(t * FPS) for t in meta["events"][CLASSES[k]]}), int)
            frames = frames[(frames >= 0) & (frames < n)]
            self.events[k] = frames
            near = np.zeros(n, bool)  # within a frame of a written stroke
            for d in (-1, 0, 1):
                near[np.clip(frames + d, 0, n - 1)] = True
            # A written stroke is a stroke; a frame nobody wrote anything at is silence only where
            # the original model is quiet too, since transcribers leave out ghost notes and the like.
            quiet = self.teacher[:, k] < THRESHOLDS[k] / 2
            self.truth_w[inside & ~near & quiet, k] = 1.0
            # Kick and tom are rarely left unwritten: where the original model hears one nobody
            # wrote, it is weakly taught silence. Snares stay unknown (ghost notes).
            if k != 1 and DOUBT:
                far = np.ones(n, bool)
                for d in range(-3, 4):
                    far[np.clip(frames + d, 0, n - 1)] = False
                self.truth_w[inside & far & ~quiet, k] = DOUBT
            self.target[frames, k] = 1.0
            self.truth_w[frames, k] = 1.0
            # Around a written stroke the original model is not followed: that is what is being fixed.
            self.kd_w[near, k] = 0.0
        # A head withheld by a partial legend is unknown, including under --doubt.
        # Keep verified positives and teacher distillation; mask only negative truth.
        for head in meta.get("withheld_heads", []) if withheld_mask else ():
            k = CLASSES.index(head["class"])
            frame = round(head["time"] * FPS)
            lo, hi = max(0, frame - 3), min(n, frame + 4)
            if lo < hi:
                negative = self.target[lo:hi, k] == 0
                self.truth_w[lo:hi, k][negative] = 0
        self.fill_frames = fill_frames(meta["events"], meta["intervals"])
        self.inside_frames = np.flatnonzero(inside)
        # Dense spans are positive bags, never accepted intervals or frame targets.
        self.dense_windows = np.empty((0, 3), dtype=int)  # [first, stop, class], stop exclusive
        self.dense_frames = np.empty(0, dtype=int)
        self.dense_candidates = []
        dense_path = folder / ("dense.json" if dense_fold is None else f"dense-fold{dense_fold}.json")
        if dense and dense_fold is not None and not dense_path.exists():
            raise FileNotFoundError(f"{dense_path}: run tools/finetune/dense.py first")
        if dense and dense_path.exists():
            labels = json.loads(dense_path.read_text())
            if dense_fold is not None and (labels.get("quality_fold") != dense_fold or
                                           labels.get("reference_scope") != "training-only"):
                raise ValueError(f"{dense_path}: expected training-only quality for fold {dense_fold}")
            if labels["fps"] != FPS or labels.get("positive_only") is not True:
                raise ValueError(f"{dense_path}: expected {FPS} fps, positive-only labels")
            windows, centres = [], []
            for label in labels["labels"]:
                k = CLASSES.index(label["class"])
                centre = round(label["time"] * labels["fps"])
                radius = label.get("tolerance_frames", labels["tolerance_frames"])
                first, stop = centre - radius, centre + radius + 1
                if k not in LEARNED or not isinstance(radius, int) or radius < 0 or not 0 <= first < stop <= n:
                    raise ValueError(f"{dense_path}: invalid dense label {label}")
                windows.append((first, stop, k))
                centres.append(centre)
            self.dense_windows = np.asarray(windows, dtype=int).reshape(-1, 3)
            self.dense_frames = np.asarray(centres, dtype=int)
            if dense_mode == "exclusive":
                self.dense_candidates = exclusive_candidates(
                    self.teacher, self.target, labels["labels"], self.dense_windows,
                    dense_guard_frames, dense_teacher_threshold, dense_radius)
                # Disabled snare labels still protect their support from kick/tom positives.
                if dense_snare_weight == 0:
                    self.dense_candidates = [np.empty(0, dtype=int) if k == 1 else frames
                                             for (_, _, k), frames in
                                             zip(self.dense_windows, self.dense_candidates)]
                usable = [bool(len(frames)) for frames in self.dense_candidates]
                self.dense_frames = self.dense_frames[usable]


def fill_frames(events: dict, intervals: list) -> np.ndarray:
    """Frames of the written strokes in fills: four or more kick/snare/tom moments 0.14 s apart or less, with a tom."""
    moments = sorted({(round(t, 3), c) for c in CLASSES[:3] for t in events[c]})
    times = sorted({t for t, _ in moments})
    toms = {t for t, c in moments if c == "tom"}
    out, run = [], []
    for t in times + [float("inf")]:
        if run and t - run[-1] > 0.14:
            if len(run) >= 4 and toms & set(run) and any(a <= run[0] <= b for a, b in intervals):
                out += run
            run = []
        run.append(t)
    return np.array([round(t * FPS) for t in out], int)


def load_songs(device: str, dense: bool = False, dense_fold: str | None = None,
               only: list[str] | None = None, withheld_mask: bool = True, **dense_options) -> list[Song]:
    seal = vault.Vault()
    teacher = original_model(device).eval()
    return [Song(d, teacher, device, dense=dense, dense_fold=dense_fold, withheld_mask=withheld_mask,
                 **dense_options)
            for d in sorted(CACHE.iterdir()) if (d / "truth.json").exists()
            and not seal.contains_song(json.loads((d / "truth.json").read_text()))
            and (not only or json.loads((d / "truth.json").read_text())["channel"] in only)]


def crop(song: Song, length: int, rng: np.random.Generator) -> slice:
    n = len(song.feat)
    r = rng.random()
    if r < 0.4 and len(song.fill_frames):
        centre = int(rng.choice(song.fill_frames))
    elif 0.4 <= r < 0.6 and len(song.dense_frames):
        centre = int(rng.choice(song.dense_frames))
        start = int(np.clip(centre - length // 2, 0, max(0, n - length)))
        return slice(start, start + length)
    elif r < 0.9 and len(song.inside_frames):
        centre = int(rng.choice(song.inside_frames))
    else:
        centre = int(rng.integers(n))
    start = int(np.clip(centre - rng.integers(length), 0, max(0, n - length)))
    return slice(start, start + length)


def augment(feat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A random level and a gentle spectral tilt, applied before the log compression."""
    linear = np.power(10.0, feat) - 1.0
    gain = 2 ** rng.uniform(-1, 1)
    tilt = 2 ** (rng.uniform(-0.5, 0.5) * np.linspace(-1, 1, feat.shape[1]))
    return np.log10(1.0 + linear * gain * tilt).astype(np.float32)


def batch(songs: list[Song], weights: np.ndarray, size: int, length: int, rng: np.random.Generator,
          device: str, dense_mode: str = "peak"):
    xs, ts, ms, ks, ys = [], [], [], [], []
    dense_windows = []  # [batch index, first, stop, class], relative to the crop
    for i in rng.choice(len(songs), size=size, p=weights):
        s = songs[i]
        window = crop(s, length, rng)
        feat = s.feat[window]
        if len(feat) < length:
            pad = length - len(feat)
            feat = np.pad(feat, ((0, pad), (0, 0)))
            pads = lambda a: np.pad(a[window], ((0, pad), (0, 0)))  # noqa: E731
        else:
            pads = lambda a: a[window]  # noqa: E731
        xs.append(augment(feat, rng))
        ys.append(pads(s.target))
        ms.append(pads(s.truth_w))
        kw = pads(s.kd_w).copy()
        for bag, (first, stop, k) in enumerate(s.dense_windows):
            # Never clip a positive bag at a crop edge or include padded frames.
            if window.start <= first and stop <= min(window.stop, len(s.feat)):
                lo, hi = int(first - window.start), int(stop - window.start)
                if dense_mode == "window":  # v3 reproduction
                    kw[lo:hi, k] = 0.0
                if dense_mode == "exclusive":
                    candidates = tuple(int(frame - window.start) for frame in s.dense_candidates[bag])
                    if candidates:
                        dense_windows.append(ExclusiveBag(len(xs) - 1, lo, hi, int(k), candidates))
                else:
                    dense_windows.append((len(xs) - 1, lo, hi, int(k)))
        ks.append(kw)
        ts.append(pads(s.teacher))
    to = lambda a: torch.from_numpy(np.stack(a)).to(device)  # noqa: E731
    return to(xs)[..., None], to(ys), to(ms), to(ks), to(ts), dense_windows


def loss_fn(out, target, truth_w, kd_w, teacher, distill: float,
            dense_windows=(), dense: float = 0.0, dense_mode: str = "peak",
            dense_snare_weight: float = 1.0) -> torch.Tensor:
    if dense_mode not in ("peak", "window", "exclusive"):
        raise ValueError(f"invalid dense mode: {dense_mode}")
    eps = 1e-6
    out = out.clamp(eps, 1 - eps)
    positives = []
    positive_weights = []
    if dense and len(dense_windows):
        if dense_mode in ("peak", "exclusive"):
            truth_w, kd_w = truth_w.clone(), kd_w.clone()
            # A single host synchronization for all argmax indices on MPS.
            if dense_mode == "exclusive":
                peaks = torch.stack([out[b.sample, list(b.candidates), b.k].detach().argmax()
                                     for b in dense_windows]).cpu().tolist()
            else:
                peaks = torch.stack([out[i, first:stop, k].detach().argmax()
                                     for i, first, stop, k in dense_windows]).cpu().tolist()
        for bag, window in enumerate(dense_windows):
            if dense_mode == "exclusive":
                i, k = window.sample, window.k
                peak = window.candidates[peaks[bag]]
                local = [frame for frame in window.candidates if abs(frame - peak) <= 1]
                # Blocked neighbors retain KD/negative truth even beside the chosen peak.
                kd_w[i, local, k] = 0
                truth_w[i, local, k] *= (target[i, local, k] > 0)
                positives.append(out[i, peak, k])
                positive_weights.append(dense_snare_weight if k == 1 else 1.0)
                continue
            i, first, stop, k = window
            if dense_mode == "peak":
                # Discrete, detached selection: only one positive gradient even on ties.
                peak = first + peaks[bag]
                lo, hi = max(first, peak - 1), min(stop, peak + 2)
                kd_w[i, lo:hi, k] = 0
                truth_w[i, lo:hi, k] *= (target[i, lo:hi, k] > 0)
                positives.append(out[i, peak, k])
            else:
                positives.append(out[i, first:stop, k].max())
    truth = F.binary_cross_entropy(out, target, weight=truth_w, reduction="sum") / truth_w.sum().clamp(min=1)
    kd = F.binary_cross_entropy(out, teacher.clamp(0, 1), weight=kd_w, reduction="sum") / kd_w.sum().clamp(min=1)
    loss = truth + distill * kd
    if positives:
        positives = torch.stack(positives)
        if dense_mode == "exclusive":
            positive_loss = F.binary_cross_entropy(positives, torch.ones_like(positives), reduction="none")
            positive_loss = positive_loss * positives.new_tensor(positive_weights)
            loss = loss + dense * positive_loss.mean()
        else:
            loss = loss + dense * F.binary_cross_entropy(positives, torch.ones_like(positives))
    return loss


def picked(act: np.ndarray, k: int) -> np.ndarray:
    import adtof_pytorch as adtof

    peaks = adtof.PeakPicker(thresholds=list(THRESHOLDS), fps=FPS).pick(act[None], labels=adtof.LABELS_5)[0]
    return np.array(peaks[adtof.LABELS_5[k]])


def validate(model: torch.nn.Module, songs: list[Song], device: str) -> dict[str, float]:
    """Plain ADTOF peak picking against the truth inside the intervals, run on the intervals with
    3 s of context: a quick proxy for the pipeline, logged only (no checkpoint is chosen by it)."""
    import sys

    sys.path.insert(0, str(ROOT / "tools" / "truth_video"))
    from align import match

    model.eval()
    tally = np.zeros((3, 3))
    for s in songs:
        edges = np.flatnonzero(np.diff(np.r_[0, s.inside.astype(int), 0]))
        for first, last in zip(edges[::2], edges[1::2]):
            lo, hi = max(0, first - 300), min(len(s.feat), last + 300)
            act = run(model, s.feat[lo:hi], device)
            for k in LEARNED:
                ref = s.events[k][(s.events[k] >= first) & (s.events[k] < last)] / FPS
                est = picked(act, k) + lo / FPS
                est = est[(est * FPS >= first) & (est * FPS < last)]
                tally[k] += (len(match(list(ref), list(est))), len(est), len(ref))
    return {CLASSES[k]: [round(tally[k][0] / max(1, tally[k][1]), 4), round(tally[k][0] / max(1, tally[k][2]), 4)] for k in LEARNED}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", choices=("0", "1", "2", "all"), default="0")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--distill", type=float, default=0.3)
    p.add_argument("--every", type=int, default=250)
    p.add_argument("--tag", default="v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--val-songs", type=int, default=12, help="held-out songs in the logged proxy score")
    p.add_argument("--doubt", type=float, default=0.0, help="weight of unwritten kicks/toms the original model hears")
    p.add_argument("--dense", type=float, nargs="?", const=0.3, default=0.0,
                   help="enable dense positives with weight W (default when enabled: 0.3; absent: disabled)")
    p.add_argument("--dense-mode", choices=("peak", "window", "exclusive"), default="peak",
                   help="peak/window: historical losses; exclusive: forbid competing kick/snare/tom frames")
    p.add_argument("--dense-guard-frames", type=int, default=2,
                   help="exclusive: exclude +/-N frames around competing notation/teacher activity (default: 2)")
    p.add_argument("--dense-teacher-threshold", type=float, default=0.5,
                   help="exclusive: other kick/snare/tom teacher activity >=T blocks a frame (default: 0.5)")
    p.add_argument("--dense-radius", type=int,
                   help="exclusive: additionally limit candidates to +/-N frames of the timing estimate; never widen bags")
    p.add_argument("--dense-snare-weight", type=float, default=1.0,
                   help="exclusive: multiply each snare bag's BCE by W; 0 disables snare positives (default: 1)")
    p.add_argument("--dense-labels", choices=("fold", "legacy"), default="fold",
                   help="fold-local quality labels, or existing dense.json for v3 reproduction")
    p.add_argument("--only", nargs="+", help="restrict training/held-out cohort (v3: original 14 channels)")
    p.add_argument("--withheld-mask", action=argparse.BooleanOptionalAction, default=True,
                   help="mask withheld-head negatives (disable only to reproduce historical v3)")
    a = p.parse_args(argv)
    if not np.isfinite(a.dense) or a.dense < 0:
        p.error("--dense must be finite and non-negative")
    if a.dense_guard_frames < 0 or (a.dense_radius is not None and a.dense_radius < 0):
        p.error("--dense-guard-frames and --dense-radius must be non-negative")
    if not np.isfinite(a.dense_teacher_threshold) or not 0 < a.dense_teacher_threshold <= 1:
        p.error("--dense-teacher-threshold must be finite and in (0, 1]")
    if not np.isfinite(a.dense_snare_weight) or a.dense_snare_weight < 0:
        p.error("--dense-snare-weight must be finite and non-negative")
    if a.dense_mode != "exclusive" and (a.dense_guard_frames != 2 or a.dense_teacher_threshold != 0.5
                                         or a.dense_radius is not None or a.dense_snare_weight != 1):
        p.error("dense guard/teacher/radius/snare controls require --dense-mode exclusive")
    return a


def dense_eligibility(songs: list[Song]) -> dict:
    """Static training-only coverage, so a conservative gate cannot silently remove a class."""
    counts = {c: {"labels": 0, "usable": 0, "skipped": 0, "candidate_frames": 0}
              for c in CLASSES[:3]}
    for song in songs:
        for (_, _, k), candidates in zip(song.dense_windows, song.dense_candidates):
            row = counts[CLASSES[k]]
            row["labels"] += 1
            row["usable"] += bool(len(candidates))
            row["skipped"] += not len(candidates)
            row["candidate_frames"] += len(candidates)
    return counts


def main() -> None:
    a = parse_args()
    device = a.device
    global DOUBT
    DOUBT = a.doubt
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    songs = load_songs(device, dense=a.dense > 0,
                       dense_fold=a.fold if a.dense_labels == "fold" else None,
                       only=a.only, withheld_mask=a.withheld_mask, dense_mode=a.dense_mode,
                       dense_guard_frames=a.dense_guard_frames,
                       dense_teacher_threshold=a.dense_teacher_threshold,
                       dense_radius=a.dense_radius, dense_snare_weight=a.dense_snare_weight)
    train, held = split_songs(songs, a.fold)
    eligibility = dense_eligibility(train) if a.dense and a.dense_mode == "exclusive" else None
    if eligibility is not None:
        print(json.dumps({"dense eligibility": eligibility}), flush=True)
    # Songs are drawn by how much truth they hold.
    weights = np.array([s.inside.sum() for s in train], float)
    weights /= weights.sum()
    model = original_model(device)
    probe = [held[i] for i in np.random.default_rng(1).permutation(len(held))[:a.val_songs]]
    print(f"fold {a.fold}: train {len(train)} songs, held out {len(held)}; before:",
          validate(model, probe, device) if probe else "-", flush=True)
    for module in model.modules():  # BatchNorm keeps the statistics it was trained with
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=a.lr)
    length = int(a.seconds * FPS)
    MODELS.mkdir(parents=True, exist_ok=True)
    name = f"{a.tag}-fold{a.fold}"
    log, started = [], time.time()
    dense_count = 0
    for step in range(1, a.steps + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        x, y, m, kw, t, dense_windows = batch(train, weights, a.batch, length, rng, device, a.dense_mode)
        dense_count += len(dense_windows)
        loss = loss_fn(model(x), y, m, kw, t, a.distill, dense_windows, a.dense, a.dense_mode,
                       a.dense_snare_weight)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.every == 0 or step == a.steps:
            score = validate(model, probe, device) if probe else {}
            log.append({"step": step, "loss": round(float(loss.detach()), 4), "val P/R": score, "minutes": round((time.time() - started) / 60, 1)})
            if a.dense:
                log[-1]["dense labels"] = dense_count
            print(json.dumps(log[-1], ensure_ascii=False), flush=True)
            torch.save(model.state_dict(), MODELS / f"{name}-{step}.pt")
    torch.save(model.state_dict(), MODELS / f"{name}.pt")
    metadata = {"args": vars(a), "held": [s.slug for s in held],
                "train": [s.slug for s in train], "log": log}
    if eligibility is not None:
        metadata["dense eligibility"] = eligibility
    (MODELS / f"{name}.json").write_text(json.dumps(metadata, indent=1))


if __name__ == "__main__":
    main()
