"""Bring back toms the snare-echo rule drops when they are real unisons (snare+tom, kick+tom).

ADTOF often hears a snare+floor-tom unison as both classes at once, and models._toms then
drops the tom as the snare's echo. A frozen gradient-boosted classifier (JSON trees, NumPy
only) looks at each tom peak the pipeline left out and adds the ones it calls real.
Measured on 420 fresh non-vault songs (2026-10-14): tom F1 80.7 -> 82.4, precision flat.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .cymbal_veto import predict_proba
from .models import ADTOF_CLASSES, ADTOF_FPS, TOM_BLEND, Hit, picked_frames

MODEL_PATH = Path(__file__).parent / 'data' / 'tom_rescue.json'
MODEL_SHA256 = '9236c3c9061494cf42ccf1c112d6dee909cf5042059e5a5af564fcb41e56c163'
KICK, SNARE, TOM = (ADTOF_CLASSES.index(name) for name in ('kick', 'snare', 'tom'))
CANDIDATE_THRESHOLD = 0.15  # tom peaks of the stem/mix blend this strong are candidates
CLEAR = 0.03  # seconds; a candidate this close to a kept tom is the same stroke


def load_model():
    payload = MODEL_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MODEL_SHA256:
        raise ValueError('model SHA-256 mismatch')
    return json.loads(payload)


def _family(inst):
    return 'tom' if inst.startswith('tom') else 'snare' if inst == 'side_stick' else inst


def _features(f, act, mix, response, kept, song_stats):
    n = len(act)
    idx = np.clip(np.arange(f - 3, f + 7), 0, n - 1)
    x = [act[idx].ravel(), mix[idx].ravel()]
    x.append(response[np.clip(np.arange(f - 2, f + 4), 0, len(response) - 1)].ravel())
    t = f / ADTOF_FPS
    context = []
    for inst in ('tom', 'snare', 'kick'):
        times = kept[inst]
        d = np.abs(times - t) if len(times) else np.array([9.0])
        context += [d.min(), (d < 1.0).sum(), (d < .25).sum()]
    x.append(np.array(context))
    x.append(np.array(song_stats))
    blend = (act[f] + mix[f]) / 2
    x.append(np.array([blend[TOM], blend[SNARE], blend[KICK],
                       blend[TOM] / (blend[SNARE] + 1e-3), blend[TOM] / (blend[KICK] + 1e-3)]))
    return np.concatenate(x)


def candidates(hits, activations, mix, response, thresholds):
    """Tom peaks the pipeline left out, with their feature rows."""
    frames = min(len(activations), len(mix))
    act, mix = activations[:frames], mix[:frames]
    kept = {inst: np.array(sorted(h.time for h in hits if _family(h.inst) == inst)) for inst in ('tom', 'snare', 'kick')}
    blend = (act + mix) / 2
    snare_frames = (kept['snare'] * ADTOF_FPS).astype(int)
    snare_frames = snare_frames[snare_frames < frames]
    tom_frames = (kept['tom'] * ADTOF_FPS).astype(int)
    tom_frames = tom_frames[tom_frames < frames]
    # How much this kit's snare bleeds into the tom output, and how its toms usually sound.
    song_stats = [
        np.median(blend[snare_frames, TOM] / (blend[snare_frames, SNARE] + 1e-3)) if len(snare_frames) else 0,
        np.percentile(blend[snare_frames, TOM], 90) if len(snare_frames) else 0,
        np.median(blend[tom_frames, TOM]) if len(tom_frames) else 0,
        len(tom_frames) / max(1, len(snare_frames)),
    ]
    low = list(thresholds)
    low[TOM] = CANDIDATE_THRESHOLD
    found = [f for f in picked_frames(blend, low)['tom']
             if not len(kept['tom']) or np.abs(kept['tom'] - f / ADTOF_FPS).min() > CLEAR]
    rows = np.array([_features(f, act, mix, response, kept, song_stats) for f in found])
    return found, rows, blend


def rescue(hits, activations, mix, response, thresholds, log=print):
    """Fail closed: return the hits unchanged if the model or features fail."""
    try:
        model = load_model()
        found, rows, blend = candidates(hits, activations, mix, response, thresholds)
        if not found:
            log('tom rescue: 0 of 0 candidates')
            return hits
        keep = predict_proba(model, rows)[:, 1] >= model['threshold']
        added = [Hit(float(f) / ADTOF_FPS, 'tom', float(blend[f, TOM])) for f, k in zip(found, keep) if k]
    except Exception as error:
        log(f'tom rescue: skipped ({error})')
        return hits
    log(f'tom rescue: {len(added)} of {len(found)} candidates -> tom')
    return sorted(hits + added, key=lambda h: h.time)
