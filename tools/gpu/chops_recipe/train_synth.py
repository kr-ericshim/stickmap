"""Fine-tune ADTOF as tools/finetune/train.py does, plus synthetic fast fills pasted into the crops.

Each training song gives a bank of its own written, isolated kick/snare/tom strokes (12-frame
snippets of linear magnitude, floor removed). Half the crops get a synthetic fill inside an
accepted interval: 4-16 strokes 40-90 ms apart, random drums, kick+tom/snare unisons, doubles,
each stroke a random snippet at a random gain, added to the crop's linear magnitude. The pasted
strokes are labelled exactly; hi-hat/cymbal distillation is switched off inside the pasted window.

    .venv/bin/python .eval-cache/chops/train_synth.py --fold 0 --tag v3 --doubt 0.3 --steps 1500 --every 500
    .venv/bin/python .eval-cache/chops/train_synth.py --fold 0 --tag v2s1 --seed 1 --synth 0   # control: same recipe, no synthesis

Models go to .eval-cache/chops/models/, never into tools/ or .eval-cache/ft/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v2frozen as train  # noqa: E402  (git HEAD tools/finetune/train.py, frozen: the recipe v2/v3 used)

train.MODELS = ROOT / ".eval-cache" / "chops" / "models"
FPS = train.FPS
SNIP = 12  # frames of a stroke snippet
FADE = 4
ISOLATE = 8  # frames: no other written stroke this close
P_SYNTH = 0.5
CLASS_P = (0.25, 0.30, 0.45)  # kick, snare, tom
P_UNISON_KICK = 0.25  # a kick under a tom/snare stroke (v3); --low-prior sets 0.08
P_UNISON_TOM = 0.10  # a tom with a kick stroke; --low-prior sets 0.05
P_DOUBLE = 0.15  # the next stroke repeats the drum; --low-prior sets 0.05
MASKFIX = False  # --maskfix: no kick/snare/tom distillation within ±MASK_FRAMES of any written kick/snare/tom stroke
MASK_FRAMES = 2
GAIN = (0.6, 1.4)
SPACING = (4.0, 9.0)  # frames between strokes
N_STROKES = (4, 16)


def bank_for(song: train.Song) -> dict[int, np.ndarray]:
    """Per class: snippets [count, SNIP, bins] of linear magnitude a written isolated stroke added."""
    lin = np.power(10.0, song.feat.astype(np.float32)) - 1.0
    n = len(lin)
    all_frames = np.sort(np.concatenate([song.events[k] for k in train.LEARNED]))
    hat = (song.teacher[:, 3] >= train.THRESHOLDS[3] / 2) | (song.teacher[:, 4] >= train.THRESHOLDS[4] / 2)
    fade = np.ones(SNIP, np.float32)
    fade[-FADE:] = np.linspace(1, 0, FADE + 1)[1:]
    bank = {}
    for k in train.LEARNED:
        frames = song.events[k]
        clean, loose = [], []
        for f in frames:
            if f < 5 or f + SNIP >= n or not song.inside[f]:
                continue
            others = np.abs(all_frames - f)
            if np.count_nonzero(others <= ISOLATE) > 1:  # itself plus another stroke
                continue
            if song.teacher[f - 1:f + 3, k].max() < train.THRESHOLDS[k]:  # the original model must hear it: a clean stroke
                continue
            (loose if hat[f - 2:f + 8].any() else clean).append(f)
        use = clean if len(clean) >= 8 else clean + loose
        snippets = []
        for f in use[:60]:
            floor = np.median(lin[f - 4:f - 1], axis=0)
            s = np.maximum(lin[f - 1:f - 1 + SNIP] - floor, 0) * fade[:, None]
            if s.max() > 0:
                snippets.append(s)
        if snippets:
            bank[k] = np.stack(snippets).astype(np.float32)
    return bank


def paste(song: train.Song, window: slice, lin: np.ndarray, target, truth_w, kd_w, rng: np.random.Generator) -> bool:
    """Add a synthetic fill to this crop in place; returns whether one was placed."""
    bank = song.bank
    if 2 not in bank or (0 not in bank and 1 not in bank):
        return False
    length = lin.shape[0]
    inside = song.inside[window]
    if len(inside) < length:
        inside = np.pad(inside, (0, length - len(inside)))
    count = int(rng.integers(N_STROKES[0], N_STROKES[1] + 1))
    spacing = float(rng.uniform(*SPACING))
    span = int(count * spacing) + SNIP + 4
    candidates = np.flatnonzero(inside)
    if not len(candidates) or span >= length:
        return False
    start = None
    for _ in range(8):
        c = int(rng.choice(candidates))
        if c + span < length and inside[c:c + span].all():
            start = c
            break
    if start is None:
        return False
    classes = [k for k in train.LEARNED if k in bank]
    probs = np.array([CLASS_P[k] for k in classes]); probs /= probs.sum()
    strokes = []  # (frame, class)
    previous = None
    pos = float(start + 2)
    for i in range(count):
        if previous is not None and rng.random() < P_DOUBLE:
            k = previous
        else:
            k = int(rng.choice(classes, p=probs))
        f = int(round(pos + rng.uniform(-0.5, 0.5)))
        strokes.append((f, k))
        if k in (1, 2) and 0 in bank and rng.random() < P_UNISON_KICK:
            strokes.append((f, 0))
        elif k == 0 and 2 in bank and rng.random() < P_UNISON_TOM:
            strokes.append((f, 2))
        previous = k
        pos += spacing * float(rng.uniform(0.85, 1.15))
    if not any(k == 2 for _, k in strokes):
        strokes[int(rng.integers(len(strokes)))] = (strokes[int(rng.integers(len(strokes)))][0], 2)
    lo, hi = start, min(length, int(pos) + SNIP + 2)
    kd_w[lo:hi, 3:5] = 0.0  # pasted strokes may carry hi-hat: the original model's hi-hat/cymbal is not followed here
    for f, k in strokes:
        snip = bank[k][int(rng.integers(len(bank[k])))] * float(rng.uniform(*GAIN))
        a, b = f - 1, min(length, f - 1 + SNIP)
        lin[a:b] += snip[:b - a]
        target[f, k] = 1.0
        truth_w[f, k] = 1.0
        for d in (-1, 1):
            g = f + d
            if 0 <= g < length and target[g, k] < 1:
                truth_w[g, k] = 0.0
        kd_w[max(0, f - 1):f + 2, k] = 0.0
    return True


def batch(songs, weights, size, length, rng, device):
    xs, ts, ms, ks, ys = [], [], [], [], []
    for i in rng.choice(len(songs), size=size, p=weights):
        s = songs[i]
        window = train.crop(s, length, rng)
        feat = s.feat[window].astype(np.float32)
        pad = length - len(feat)
        pads = (lambda a: np.pad(a[window], ((0, pad), (0, 0)))) if pad > 0 else (lambda a: a[window].copy())
        if pad > 0:
            feat = np.pad(feat, ((0, pad), (0, 0)))
        target, truth_w, kd_w, teacher = pads(s.target), pads(s.truth_w), pads(s.kd_w), pads(s.teacher)
        if SYNTH and rng.random() < P_SYNTH:
            lin = np.power(10.0, feat) - 1.0
            if paste(s, window, lin, target, truth_w, kd_w, rng):
                feat = np.log10(1.0 + lin).astype(np.float32)
        xs.append(train.augment(feat, rng))
        ys.append(target); ms.append(truth_w); ks.append(kd_w); ts.append(teacher)
    to = lambda a: torch.from_numpy(np.stack(a)).to(device)  # noqa: E731
    return to(xs)[..., None], to(ys), to(ms), to(ks), to(ts)


_load_songs = train.load_songs
SOLO_TRUTH = ROOT / ".eval-cache" / "chops" / "solo_truth"
SOLOS = False
SOLO_WEIGHT = 3.0  # solos are 5 % of the truth by length; drawn this many times more often


def maskfix(song):
    """train.py lifts the teacher only for the written stroke's own class; at a written tom the student is still
    held to the teacher's kick/snare. Lift kick/snare/tom distillation around every written K/S/T stroke (the
    truth weights are untouched: nothing asserts the other drums' absence)."""
    n = len(song.kd_w)
    frames = np.concatenate([song.events[k] for k in train.LEARNED])
    for d in range(-MASK_FRAMES, MASK_FRAMES + 1):
        song.kd_w[np.clip(frames + d, 0, n - 1), :3] = 0.0


def load_songs(device):
    songs = _load_songs(device)
    # .eval-cache/ft now also holds channels harvested later (moyin, golly, olliewoods, ...): the frozen recipe
    # would train every fold on them. Keep the original 14 so those songs stay never-trained.
    known = {c for fold in train.FOLDS for c in fold}
    dropped = sorted({s.channel for s in songs if s.channel not in known})
    songs = [s for s in songs if s.channel in known]
    print(f"cohort: {len(songs)} songs from the configured fold channels; left out: {' '.join(dropped) or 'none'}", flush=True)
    if MASKFIX:
        before = sum(float(s.kd_w[:, :3].sum()) for s in songs)
        for s in songs:
            maskfix(s)
        after = sum(float(s.kd_w[:, :3].sum()) for s in songs)
        print(f"maskfix: kick/snare/tom distillation weight {before:.0f} -> {after:.0f} frame-classes", flush=True)
    if SOLOS:
        teacher = train.original_model(device).eval()
        extra = [train.Song(d, teacher, device) for d in sorted(SOLO_TRUTH.iterdir()) if (d / "truth.json").exists()]
        extra = extra * int(SOLO_WEIGHT)  # the same Song objects listed again: drawn that many times more often
        print(f"solo videos as training truth: {len(extra)} songs, {sum(s.inside.sum() for s in extra) / 100:.0f} s inside", flush=True)
        songs += extra
        train.FOLDS = tuple(fold + (f"solo-{g}",) for fold, g in zip(train.FOLDS, "ABC"))
    for s in songs:
        s.bank = bank_for(s)
    sizes = [sum(len(v) for v in s.bank.values()) for s in songs]
    print(f"snippet banks: {sum(sizes)} snippets over {len(songs)} songs; songs with tom+kick/snare banks: "
          f"{sum(1 for s in songs if 2 in s.bank and (0 in s.bank or 1 in s.bank))}", flush=True)
    return songs


SYNTH = True
train.batch = batch
train.load_songs = load_songs

if __name__ == "__main__":
    if "--solos" in sys.argv:
        SOLOS = True
        sys.argv.remove("--solos")
    if "--maskfix" in sys.argv:
        MASKFIX = True
        sys.argv.remove("--maskfix")
    if "--low-prior" in sys.argv:
        P_UNISON_KICK, P_UNISON_TOM, P_DOUBLE = 0.08, 0.05, 0.05
        sys.argv.remove("--low-prior")
    if "--synth" in sys.argv:
        i = sys.argv.index("--synth")
        SYNTH = sys.argv[i + 1] not in ("0", "false", "no")
        del sys.argv[i:i + 2]
    train.main()
