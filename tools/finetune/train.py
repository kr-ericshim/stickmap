"""Fine-tune ADTOF's Frame-RNN on the harvested transcription truth (cached by data.py).

Kick, snare and tom are learned from the truth inside the accepted intervals;
everywhere else, and for hi-hat and cymbal (which many channels never write),
the model is held to what the original model said (distillation), so it does
not forget what it was good at. Songs are split by channel into folds: a fold
is trained on the other channels and predicts its own songs, so every song is
scored by a model that never saw its channel.

    uv run python tools/finetune/train.py --fold 0           # train on folds 1,2; predict fold 0
    uv run python tools/finetune/train.py --fold all         # train on everything (deployment)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".eval-cache" / "ft"
MODELS = CACHE / "models"
CLASSES = ("kick", "snare", "tom", "hihat", "cymbal")
LEARNED = (0, 1, 2)  # classes taught by the truth
FPS = 100
THRESHOLDS = (0.22, 0.24, 0.32, 0.22, 0.30)
DOUBT = 0.0  # weight of "nobody wrote a kick/tom here" where the original model hears one (--doubt)
# Channels in three folds of similar tom counts, each with a mix of Korean pop covers and others.
FOLDS = (
    ("swartz", "allyeodrum", "drumclass", "mydrumsheet", "taktak"),
    ("drumshin", "drumplace", "nakamura", "chock"),
    ("yeol", "atthedrum", "jilju", "drummate", "drumscribe"),
)

# The same song transcribed by channels in different folds: never trained on while the other is scored.
SAME_SONG = ("love-me-or-leave-me", "나의-x", "last-christmas", "벚꽃엔딩")


def same_song(a: str, b: str) -> bool:
    return any(key in a and key in b for key in SAME_SONG)


def original_model(device: str) -> torch.nn.Module:
    import adtof_pytorch as adtof

    model = adtof.create_frame_rnn_model(adtof.calculate_n_bins())
    return adtof.load_pytorch_weights(model, adtof.get_default_weights_path(), strict=False).to(device)


def run(model: torch.nn.Module, feat: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(feat.astype(np.float32))[None, :, :, None].to(device)
        return model(x).cpu().numpy()[0]


class Song:
    def __init__(self, folder: Path, teacher_model: torch.nn.Module, device: str):
        meta = json.loads((folder / "truth.json").read_text())
        self.slug, self.channel = meta["slug"], meta["channel"]
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
        self.fill_frames = fill_frames(meta["events"], meta["intervals"])
        self.inside_frames = np.flatnonzero(inside)


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


def load_songs(device: str) -> list[Song]:
    teacher = original_model(device).eval()
    return [Song(d, teacher, device) for d in sorted(CACHE.iterdir()) if (d / "truth.json").exists()]


def crop(song: Song, length: int, rng: np.random.Generator) -> tuple[slice, ...]:
    n = len(song.feat)
    r = rng.random()
    if r < 0.4 and len(song.fill_frames):
        centre = int(rng.choice(song.fill_frames))
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


def batch(songs: list[Song], weights: np.ndarray, size: int, length: int, rng: np.random.Generator, device: str):
    xs, ts, ms, ks, ys = [], [], [], [], []
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
        ks.append(pads(s.kd_w))
        ts.append(pads(s.teacher))
    to = lambda a: torch.from_numpy(np.stack(a)).to(device)  # noqa: E731
    return to(xs)[..., None], to(ys), to(ms), to(ks), to(ts)


def loss_fn(out, target, truth_w, kd_w, teacher, distill: float) -> torch.Tensor:
    eps = 1e-6
    out = out.clamp(eps, 1 - eps)
    truth = F.binary_cross_entropy(out, target, weight=truth_w, reduction="sum") / truth_w.sum().clamp(min=1)
    kd = F.binary_cross_entropy(out, teacher.clamp(0, 1), weight=kd_w, reduction="sum") / kd_w.sum().clamp(min=1)
    return truth + distill * kd


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fold", default="0")
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
    a = p.parse_args()
    device = a.device
    global DOUBT
    DOUBT = a.doubt
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    songs = load_songs(device)
    if a.fold == "all":
        train, held = songs, []
    else:
        held_channels = FOLDS[int(a.fold)]
        held = [s for s in songs if s.channel in held_channels]
        train = [s for s in songs if s.channel not in held_channels and not any(same_song(s.slug, h.slug) for h in held)]
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
    for step in range(1, a.steps + 1):
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        x, y, m, kw, t = batch(train, weights, a.batch, length, rng, device)
        loss = loss_fn(model(x), y, m, kw, t, a.distill)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % a.every == 0 or step == a.steps:
            score = validate(model, probe, device) if probe else {}
            log.append({"step": step, "loss": round(float(loss), 4), "val P/R": score, "minutes": round((time.time() - started) / 60, 1)})
            print(json.dumps(log[-1], ensure_ascii=False), flush=True)
            torch.save(model.state_dict(), MODELS / f"{name}-{step}.pt")
    torch.save(model.state_dict(), MODELS / f"{name}.pt")
    (MODELS / f"{name}.json").write_text(json.dumps({"args": vars(a), "held": [s.slug for s in held], "log": log}, indent=1))


if __name__ == "__main__":
    main()
