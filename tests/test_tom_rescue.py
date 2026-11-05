import hashlib

import numpy as np

from stickmap import tom_rescue
from stickmap.models import DEFAULT_THRESHOLDS, Hit


def test_model_asset_matches_hash():
    model = tom_rescue.load_model()
    assert hashlib.sha256(tom_rescue.MODEL_PATH.read_bytes()).hexdigest() == tom_rescue.MODEL_SHA256
    assert model["threshold"] == 0.5 and model["n_features"] > 0


def _song(frames=400):
    act = np.zeros((frames, 5)); mix = np.zeros((frames, 5)); response = np.zeros((frames, 2))
    for f in range(50, 350, 50):  # snares with a strong tom in the same frame: a unison
        act[f, 1] = mix[f, 1] = .8
        act[f, 2] = mix[f, 2] = .65
        response[f] = (1.0, .9)
    return act, mix, response


def test_candidates_are_tom_peaks_the_pipeline_left_out():
    act, mix, response = _song()
    hits = [Hit(f / 100, "snare", .8) for f in range(50, 350, 50)]
    found, rows, _ = tom_rescue.candidates(hits, act, mix, response, list(DEFAULT_THRESHOLDS))
    assert found == list(range(50, 350, 50))
    assert rows.shape == (len(found), tom_rescue.load_model()["n_features"])
    kept = hits + [Hit(.5, "tom", .65)]
    assert 50 not in tom_rescue.candidates(kept, act, mix, response, list(DEFAULT_THRESHOLDS))[0]


def test_rescue_only_adds_toms_and_fails_closed(monkeypatch):
    act, mix, response = _song()
    hits = [Hit(f / 100, "snare", .8) for f in range(50, 350, 50)]
    out = tom_rescue.rescue(hits, act, mix, response, list(DEFAULT_THRESHOLDS), log=lambda _: None)
    assert [h for h in out if h.inst != "tom"] == hits
    assert all(h.inst == "tom" for h in out if h not in hits)
    monkeypatch.setattr(tom_rescue, "MODEL_SHA256", "0" * 64)
    assert tom_rescue.rescue(hits, act, mix, response, list(DEFAULT_THRESHOLDS), log=lambda _: None) == hits
