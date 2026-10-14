"""Conservative K/S/T positive, audited negative and unknown masks.

Audit is optional kst_audit.json in canonical song/truth, never inferred from
teacher silence. Unknown K/S/T has neither BCE nor KD. H/H and cymbal retain
the original teacher replay, except explicitly withheld heads. Positive,
negative and KD losses each normalize by their own number of valid labels.
"""
import math

import numpy as np

CLASSES = ('kick', 'snare', 'tom', 'hihat', 'cymbal')
FPS = 100


def validate_audit(audit):
    if audit is None:
        return
    if audit.get('schema') != 1 or not all(isinstance(audit.get(k), str) and audit[k].strip()
                                         for k in ('auditor', 'evidence')):
        raise ValueError('kst_audit requires schema=1, auditor and evidence')
    for row in audit.get('complete_intervals', []):
        if row.get('classes') != list(CLASSES[:3]):
            raise ValueError('complete audit intervals must explicitly cover kick/snare/tom')
        _bounds(row)
    for row in audit.get('verified_negative', []):
        if row.get('class') not in CLASSES[:3]:
            raise ValueError('verified negative must name kick/snare/tom')
        _bounds(row)


def _bounds(row):
    a, b = row['start'], row['end']
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (a, b)) or not 0 <= a <= b:
        raise ValueError('audit interval must have finite 0 <= start <= end')
    return a, b


def masks(meta, n, audit=None):
    validate_audit(audit)
    target = np.zeros((n, 5), dtype=np.float32)
    known = np.zeros((n, 5), dtype=np.float32)
    kd = np.zeros((n, 5), dtype=np.float32)
    kd[:, 3:] = 1  # selective replay of untrained hat/cymbal heads
    inside = np.zeros(n, bool)
    for a, b in meta['intervals']:
        inside[max(0, round(a * FPS)):min(n, round(b * FPS) + 1)] = True
    near = np.zeros((n, 3), bool)
    for k, cls in enumerate(CLASSES[:3]):
        frames = np.array(sorted({round(t * FPS) for t in meta['events'][cls]
                                  if any(a <= t <= b for a, b in meta['intervals'])}), dtype=int)
        frames = frames[(frames >= 0) & (frames < n)]
        target[frames, k] = 1
        known[frames, k] = 1
        for delta in range(-2, 3):
            near[np.clip(frames + delta, 0, n - 1), k] = True
    if audit:
        rows = [(r, range(3)) for r in audit.get('complete_intervals', [])]
        rows += [(r, [CLASSES.index(r['class'])]) for r in audit.get('verified_negative', [])]
        for row, classes in rows:
            a, b = _bounds(row)
            if not any(lo <= a <= b <= hi for lo, hi in meta['intervals']):
                raise ValueError('audit must be contained in a strict accepted interval')
            first, stop = max(0, round(a * FPS)), min(n, round(b * FPS) + 1)
            for k in classes:
                if row in audit.get('verified_negative', []) and target[first:stop, k].any():
                    raise ValueError('verified negative conflicts with written positive')
                valid = ~near[first:stop, k]  # do not penalize +/-2-frame timing neighbourhood
                known[first:stop, k][valid] = 1
                kd[first:stop, k][valid] = 1
    # ALL K/S/T KD disappears near ANY written K/S/T onset, including competing
    # classes. This does not create a negative for an unwritten simultaneous hit.
    kd[np.any(near, axis=1), :3] = 0
    for head in meta.get('withheld_heads', []):
        k = CLASSES.index(head['class'])
        f = round(head['time'] * FPS)
        first, stop = max(0, f - 3), min(n, f + 4)
        if first < stop:
            unknown = target[first:stop, k] == 0
            known[first:stop, k][unknown] = 0
            kd[first:stop, k][unknown] = 0
    return target, known, kd


def loss_fn(out, target, truth_w, kd_w, teacher, distill, dense_windows=(),
            dense=0., dense_mode='peak', dense_snare_weight=1.):
    import torch.nn.functional as F
    if dense or dense_windows:
        raise ValueError('three-state recipe does not admit dense positive bags')
    out = out.clamp(1e-6, 1 - 1e-6)
    positive = truth_w * (target > 0)
    negative = truth_w * (target == 0)
    def mean_bce(y, weight):
        return F.binary_cross_entropy(out, y, weight=weight, reduction='sum') / weight.sum().clamp(min=1)
    return (mean_bce(target, positive) + mean_bce(target, negative) +
            distill * mean_bce(teacher.clamp(0, 1), kd_w))
