"""E auxiliary branch: unchanged v2 strict batch + separate KD/CTC weak batch.

One whole-window weak crop per optimizer step (1:12 samples); lambda=.05,
preregistered alternative=.02. CPU CTC keeps the tested backend and autograd
copies gradients back to the CUDA forward. Neither RNG stream changes v2 crops.
"""
from __future__ import annotations

from collections import Counter, deque
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'finetune'))
from weak_order import WeakOrderWindow, weak_order_loss
from weak_set import load_fold


def finite_gradient(gradient):
    if not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError('nonfinite E gradient; stopping')
    return gradient


def weak_objective(outputs, teacher, windows, base_loss, *, lam=.05, distill=.3):
    """No frame positives; truth=0, KST KD off only inside coarse windows.

    Keep v2 BCE normalization for the KD-only term. HH/C and crop context
    retain teacher KD; this mask is determined solely by the source window.
    """
    target, truth = torch.zeros_like(outputs), torch.zeros_like(outputs)
    kd = torch.ones_like(outputs)
    for w in windows:
        kd[w.sample, w.start:w.stop, :3] = 0
    replay = base_loss(outputs, target, truth, kd, teacher, distill, (), 0.)
    order = weak_order_loss(outputs.to('cpu'), windows).to(outputs.device)
    total = replay + lam * order
    if not bool(torch.isfinite(total)):
        raise FloatingPointError('nonfinite E loss; stopping')
    return total, order, replay


class Auxiliary:
    def __init__(self, train, sidecar, seed, lam, device, teacher_model):
        self.train, self.lam, self.device = train, lam, device
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 20261003]))
        self.seed, self.step = seed, 0
        self.model = None
        self.counts = Counter()
        self.recent_hits = deque(maxlen=100)
        self.examples, self.features, self.teachers = [], {}, {}
        for slug, entries in sorted(sidecar['songs'].items()):
            if not entries:
                continue
            # load_fold already checked vault and full provenance; recheck at cache access.
            from weak_set import vault
            if vault.Vault().contains_song({'slug': slug}):
                raise ValueError('sealed weak song before feature load')
            folder = train.CACHE / slug
            feat = np.load(folder / 'feat.npy', allow_pickle=False).astype(np.float32)
            if any(e['feature_frames'] != len(feat) or e['window_stop_frame'] > len(feat) for e in entries):
                raise ValueError('weak feature length changed')
            self.features[slug] = feat
            cached = folder / 'teacher0.npy'
            if not cached.exists():
                # Teacher feature preparation must not consume the strict torch RNG.
                devices = [torch.cuda.current_device()] if str(device).startswith('cuda') else []
                with torch.random.fork_rng(devices=devices):
                    teacher = train.run(teacher_model, feat, device).astype(np.float16)
                np.save(cached, teacher)
            self.teachers[slug] = np.load(cached, allow_pickle=False).astype(np.float32)
            if self.teachers[slug].shape != (len(feat), 5):
                raise ValueError('weak teacher/feature length mismatch')
            self.examples.extend(entries)
        if not self.examples:
            raise ValueError('empty accepted weak set; E cannot train')

    def draw(self, length=800):
        e = self.examples[int(self.rng.integers(len(self.examples)))]
        slug, a, b = e['slug'], e['window_start_frame'], e['window_stop_frame']
        n = len(self.features[slug])
        if b - a > length or not 0 <= a < b <= n:
            raise ValueError('partial/invalid weak crop; stopping')
        # Uniform valid crop starts containing the WHOLE window, no target slicing.
        low, high = max(0, b - length), min(a, max(0, n - length))
        start = int(self.rng.integers(low, high + 1))
        stop = min(n, start + length)
        feat = self.features[slug][start:stop]
        teacher = self.teachers[slug][start:stop]
        if stop - start < length:
            pad = length - (stop - start)
            feat = np.pad(feat, ((0, pad), (0, 0)))
            teacher = np.pad(teacher, ((0, pad), (0, 0)))
        # Augmentation is the v2 implementation, with its own independent RNG.
        feat = self.train.augment(feat, self.rng)
        x = torch.from_numpy(feat[None, ..., None]).to(self.device)
        t = torch.from_numpy(teacher[None]).to(self.device)
        window = WeakOrderWindow(0, a - start, b - start, tuple(e['tokens']), e['confidence'], e['weight'])
        self.counts['windows'] += 1
        self.counts[f'song.{slug}'] += 1
        for token in e['tokens']:
            for k, cls in enumerate(('kick', 'snare', 'tom')):
                self.counts[f'moments.{cls}'] += bool(token & (1 << k))
        return x, t, [window], stop - start

    def loss(self, strict_out, *args, **kwargs):
        base = self.base_loss(strict_out, *args, **kwargs)
        self.step += 1
        x, teacher, windows, valid = self.draw(strict_out.shape[1])
        if windows[0].stop > valid:
            raise ValueError('weak window touches padded feature frames')
        # Preserve strict dropout RNG too. The weak branch gets a seed per step.
        devices = [torch.cuda.current_device()] if str(self.device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(20261003 + 100000 * self.seed + self.step)
            out = self.model(x)
        out.register_hook(finite_gradient)
        total, order, replay = weak_objective(out, teacher, windows, self.base_loss,
                                             lam=self.lam, distill=float(args[4]))
        # Valid padding has no KD or CTC gradient. Production crops are normally full length.
        if valid < out.shape[1]:
            target, truth, kd = torch.zeros_like(out), torch.zeros_like(out), torch.ones_like(out)
            kd[:, valid:] = 0
            for w in windows:
                kd[w.sample, w.start:w.stop, :3] = 0
            replay = self.base_loss(out, target, truth, kd, teacher, float(args[4]), (), 0.)
            total = replay + self.lam * order
        w = windows[0]
        self.recent_hits.append(bool((out[0, w.start:w.stop, :3].detach() >= .5).any()))
        if len(self.recent_hits) == 100 and not any(self.recent_hits):
            raise FloatingPointError('weak blank collapse: 100 consecutive all-blank greedy windows; stopping')
        if self.step % 100 == 0:
            print(json.dumps({'weak_step': self.step, 'normalized_ctc_loss': float(order.detach()),
                              'weak_kd_loss': float(replay.detach()), 'lambda': self.lam,
                              'weak_draws': dict(self.counts)}), flush=True)
        self.last = {'normalized_ctc_loss': float(order.detach()), 'weak_kd_loss': float(replay.detach())}
        result = base + total
        if not bool(torch.isfinite(result)):
            raise FloatingPointError('nonfinite E total loss; stopping')
        return result

    def summary(self):
        return {'lambda': self.lam, 'batch_size': 1, 'mix_ratio': '1:12 every optimizer step',
                'ctc_backend': 'cpu with autograd transfer', 'steps': self.step,
                'counts': dict(self.counts), 'last': getattr(self, 'last', {}),
                'kd_policy': 'source-window KST KD=0; context and HH/C v2 KD; no weak frame truth',
                'blank_stop': '100 consecutive greedy-blank windows'}


def install(train, plan, plan_path, fold, seed, root):
    config = plan['weak_order']
    data = load_fold(root / config['sidecars'][fold], plan, fold, root=root)
    original_load, original_model = train.load_songs, train.original_model
    auxiliary = None

    def load(device, **options):
        nonlocal auxiliary
        songs = original_load(device, **options)
        # Additional teacher initialization must not alter strict model initialization.
        devices = [torch.cuda.current_device()] if str(device).startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            teacher = original_model(device).eval()
            auxiliary = Auxiliary(train, data, seed, config['lambda'], device, teacher)
        auxiliary.base_loss = train.loss_fn
        train.loss_fn = auxiliary.loss
        return songs

    def model(device):
        result = original_model(device)
        if auxiliary is not None:
            auxiliary.model = result
            for parameter in result.parameters():
                if parameter.requires_grad:
                    parameter.register_hook(finite_gradient)
        return result

    train.load_songs, train.original_model = load, model
    return lambda: auxiliary.summary() if auxiliary else {}
