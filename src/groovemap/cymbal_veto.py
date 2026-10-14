"""Frozen W5 cymbal veto: JSON trees and NumPy only at prediction time."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .cymbals import ride_run_indices

MODEL_PATH = Path(__file__).parent / 'data' / 'cymbal_veto.json'
MODEL_SHA256 = 'e79dc8bbcc62b3f737ebd1c083d8381ffc9a0556f6fa98d6c109957b43edd5b3'


def load_model():
    payload = MODEL_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != MODEL_SHA256:
        raise ValueError('model SHA-256 mismatch')
    return json.loads(payload)


def predict_proba(model, features):
    """Traverse numerical HGB trees, whose leaf values include learning rate."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != model['n_features']:
        raise ValueError('invalid veto feature shape')
    raw = np.full(len(x), model['baseline'], dtype=np.float64)
    for tree in model['trees']:
        nodes = {key: np.asarray(value) for key, value in tree.items()}
        position = np.zeros(len(x), dtype=np.intp)
        for _ in range(len(nodes['value'])):
            active = np.flatnonzero(~nodes['is_leaf'][position].astype(bool))
            if not len(active):
                break
            p = position[active]
            value = x[active, nodes['feature_idx'][p]]
            left = np.where(np.isnan(value), nodes['missing_go_to_left'][p],
                            value <= nodes['num_threshold'][p]).astype(bool)
            position[active] = np.where(left, nodes['left'][p], nodes['right'][p])
        else:
            raise ValueError('invalid veto tree')
        raw += nodes['value'][position]
    # Stable logistic, preserving sklearn's binary probability order.
    positive = raw >= 0
    prob = np.empty_like(raw)
    prob[positive] = 1 / (1 + np.exp(-raw[positive]))
    exp = np.exp(raw[~positive])
    prob[~positive] = exp / (1 + exp)
    return np.column_stack((1 - prob, prob))


def adtof_input(stem):
    from adtof_pytorch.audio import process_audio_file

    features, _ = process_audio_file(str(stem))
    return validate_input(features[:, :, 0].astype(np.float16))


def validate_input(features):
    if (features.ndim != 2 or features.shape[1] != 84 or not len(features)
            or features.dtype != np.float16 or not np.isfinite(features).all()):
        raise ValueError('invalid W5 ADTOF input')
    return features


def context_features(features, times):
    """Exact recipe.py windows/vec(context), including Python round and clipping."""
    validate_input(features)
    if not len(times):
        return np.empty((0, 336), dtype=np.float32)
    ix = np.asarray([round(float(t) * 100) for t in times])[:, None] + np.arange(-5, 15)[None, :]
    x = np.asarray(features[np.clip(ix, 0, len(features) - 1)], np.float32) - np.asarray(features, np.float32).mean(0)
    return np.concatenate([x[:, :5].mean(1), x[:, 5:10].mean(1),
                           x[:, 10:].mean(1), x.std(1)], 1)


def vote_runs(original, relabeled, probabilities, threshold):
    """Full-song member votes; indices refer to the original unfiltered hit list."""
    runs = ride_run_indices(original)
    vetoed = [run for run in runs if np.mean(np.asarray(probabilities)[run]) >= threshold]
    indices = {i for run in vetoed for i in run}
    return ([replace(hit, inst='cymbal') if i in indices else hit
             for i, hit in enumerate(relabeled)], len(vetoed), len(runs))


def apply_veto(original, relabeled, get_features, log=print):
    """Fail closed: keep timing-rule labels if assets, extraction or inference fail."""
    try:
        model = load_model()  # Check integrity before any feature work.
        if not ride_run_indices(original):
            log('cymbal veto: 0 of 0 ride runs -> cymbal')
            return relabeled
        indices = sorted((i for i, h in enumerate(original) if h.inst == 'cymbal'),
                         key=lambda i: original[i].time)
        x = context_features(get_features(), [original[i].time for i in indices])
        probabilities = np.zeros(len(original), dtype=np.float64)
        probabilities[indices] = predict_proba(model, x)[:, 1]
        result, k, n = vote_runs(original, relabeled, probabilities, model['threshold'])
    except Exception as error:
        log(f'cymbal veto: skipped ({error})')
        return relabeled
    log(f'cymbal veto: {k} of {n} ride runs -> cymbal')
    return result
