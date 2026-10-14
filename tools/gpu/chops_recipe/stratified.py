"""50/30/20 fast / original replay / slow crop-centre hierarchy.

Select bucket -> performance -> song -> available K/S/T -> position.
Exact written unisons are exposure diagnostics, never extra target choices.
The ordinary replay bucket uses v2's unchanged crop selection.
"""
from collections import Counter

import numpy as np

from chops_common import splits

CLASSES = ('kick', 'snare', 'tom')
FPS = 100


def centres(meta, n):
    moments = {}
    for cls in CLASSES:
        for t in meta['events'][cls]:
            f = round(t * FPS)
            if 0 <= f < n and any(a <= t <= b for a, b in meta['intervals']):
                moments.setdefault(round(t, 4), set()).add(cls)
    # Never infer speed across a rejected gap between accepted intervals.
    # Repeated/overlapping intervals contribute each centre only once.
    fast_times = set()
    for a, b in meta['intervals']:
        times = sorted(t for t in moments if a <= t <= b)
        if times:
            gaps = np.diff(times)
            fast = np.minimum(np.r_[np.inf, gaps], np.r_[gaps, np.inf]) < 0.09
            fast_times.update(t for t, yes in zip(times, fast) if yes)
    result = {'fast': {}, 'slow': {}}
    for t in sorted(moments):
        target = result['fast' if t in fast_times else 'slow']
        labels = moments[t]
        keys = sorted(labels)
        if len(labels) > 1:
            keys += ['+'.join(sorted(labels))]
        for key in keys:
            target.setdefault(key, set()).add(round(t * FPS))
    return {bucket: {key: np.array(sorted(frames), dtype=int) for key, frames in values.items()}
            for bucket, values in result.items()}


class HierarchicalSampler:
    def __init__(self, songs, metas, groups):
        self.songs = songs
        self.centres = [centres(metas[s.slug], len(s.feat)) for s in songs]
        self.buckets = {b: {} for b in ('fast', 'ordinary', 'slow')}
        self.group_for = groups
        self.draws = Counter()
        self.exposures = Counter()
        self.unison_exposures = Counter()
        self.seen = set()
        for i, song in enumerate(songs):
            for bucket in self.buckets:
                usable = (song.channel in splits.ORIGINAL_CHANNELS if bucket == 'ordinary'
                          else bool(self.centres[i][bucket]))
                if usable:
                    self.buckets[bucket].setdefault(groups[song.slug], []).append(i)
        self.available = [b for b, items in self.buckets.items() if items]
        if not self.available:
            raise ValueError('no verified crop centres or original replay songs')
        p = {'fast': 0.5, 'ordinary': 0.3, 'slow': 0.2}
        self.probabilities = np.array([p[b] for b in self.available])
        self.probabilities /= self.probabilities.sum()

    def draw(self, length, rng, replay_crop):
        bucket = str(rng.choice(self.available, p=self.probabilities))
        group = str(rng.choice(sorted(self.buckets[bucket])))
        i = int(rng.choice(self.buckets[bucket][group]))
        song = self.songs[i]
        if bucket == 'ordinary':
            key = 'replay'
            window = replay_crop(song, length, rng)
        else:
            locations = self.centres[i][bucket]
            key = str(rng.choice(sorted(k for k in locations if k in CLASSES)))
            centre = int(rng.choice(locations[key]))
            labels = [cls for cls in CLASSES if centre in locations.get(cls, ())]
            self.draws['centre.' + '+'.join(labels)] += 1
            start = int(np.clip(centre - length // 2, 0, max(0, len(song.feat) - length)))
            window = slice(start, start + length)
        self.draws[f'bucket.{bucket}'] += 1
        self.draws[f'group.{group}'] += 1
        self.draws[f'song.{song.slug}'] += 1
        self.draws['cohort.' + ('original' if song.channel in splits.ORIGINAL_CHANNELS else 'expanded')] += 1
        self.draws[f'target.{key}'] += 1
        # Unique labelled fast frames and total exposures, NOT inferred hits.
        for cls in CLASSES:
            frames = self.centres[i]['fast'].get(cls, np.empty(0, dtype=int))
            a, b = np.searchsorted(frames, [window.start, window.stop])
            self.exposures[cls] += int(b - a)
            self.seen.update((song.slug, cls, int(f)) for f in frames[a:b])
        for speed, locations in self.centres[i].items():
            for labels, frames in locations.items():
                if '+' in labels:
                    a, b = np.searchsorted(frames, [window.start, window.stop])
                    self.unison_exposures[f'{speed}.{labels}'] += int(b - a)
        return i, window

    def summary(self):
        return {'bucket_probabilities': dict(zip(self.available, self.probabilities.tolist())),
                'groups_per_bucket': {b: len(v) for b, v in self.buckets.items()},
                'draws': dict(self.draws), 'fast_exposures': dict(self.exposures),
                'unison_exposures': dict(self.unison_exposures),
                'unique_fast_seen': {c: sum(cls == c for _, cls, _ in self.seen) for c in CLASSES}}


def make_batch(base, sampler):
    def batch(songs, weights, size, length, rng, device, dense_mode='peak'):
        import torch
        xs, ys, ms, ks, ts = [], [], [], [], []
        for _ in range(size):
            i, window = sampler.draw(length, rng, base.crop)
            song = songs[i]
            pad = max(0, length - len(song.feat[window]))
            def pads(a):
                return np.pad(a[window], ((0, pad), (0, 0)))
            xs.append(base.augment(pads(song.feat), rng))
            ys.append(pads(song.target))
            ms.append(pads(song.truth_w))
            ks.append(pads(song.kd_w))
            ts.append(pads(song.teacher))
        def tensor(a):
            return torch.from_numpy(np.stack(a)).to(device)
        return tensor(xs)[..., None], tensor(ys), tensor(ms), tensor(ks), tensor(ts), []
    return batch
