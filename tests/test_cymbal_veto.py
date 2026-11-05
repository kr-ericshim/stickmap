import json

import numpy as np
import pytest

from stickmap import cymbal_veto as veto, library, pipeline
from stickmap.cymbals import ride_runs
from stickmap.score import Hit
from stickmap.server import OptionsPatch


def sklearn_model(payload):
    pytest.importorskip('sklearn')
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.ensemble._hist_gradient_boosting.common import PREDICTOR_RECORD_DTYPE
    from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
    # Fit initializes the public estimator's loss/link, then replace its trees.
    x = np.zeros((40, payload['n_features']))
    model = HistGradientBoostingClassifier(max_iter=1).fit(x, np.arange(40) % 2)
    model._baseline_prediction = np.array([[payload['baseline']]])
    model._predictors = []
    for tree in payload['trees']:
        nodes = np.zeros(len(tree['value']), dtype=PREDICTOR_RECORD_DTYPE)
        for key, values in tree.items():
            nodes[key] = values
        model._predictors.append([TreePredictor(nodes, np.zeros((0, 8), np.uint32),
                                               np.zeros((0, 8), np.uint32))])
    return model


def test_shipped_predictor_against_sklearn():
    payload = veto.load_model()
    model = sklearn_model(payload)
    x = np.random.default_rng(412).normal(size=(4096, 336))
    x[::31, ::7] = np.nan
    np.testing.assert_allclose(veto.predict_proba(payload, x), model.predict_proba(x), atol=1e-12, rtol=0)


def test_exporter_round_trip():
    pytest.importorskip('sklearn')
    from sklearn.ensemble import HistGradientBoostingClassifier
    from veto_export import export_model
    rng = np.random.default_rng(31)
    x = rng.normal(size=(160, 4))
    x[::5, 0] = np.nan
    y = (x[:, 1] + x[:, 2] > 0).astype(int)
    model = HistGradientBoostingClassifier(max_iter=4, min_samples_leaf=3).fit(x, y)
    payload = json.loads(json.dumps(export_model(model, 'test')))
    np.testing.assert_allclose(veto.predict_proba(payload, x), model.predict_proba(x), atol=1e-12, rtol=0)


@pytest.mark.parametrize('probability, expected', [(0.65, 'cymbal'), (0.649999, 'ride'), (0.9, 'cymbal')])
def test_run_vote_boundary_isolated_and_existing_ride(probability, expected):
    hits = [Hit(t, 'cymbal', 1) for t in [0, .02, .5, 3]] + [Hit(4, 'ride', 1)]
    result, k, n = veto.vote_runs(hits, ride_runs(hits), [probability] * 5, .65)
    assert [h.inst for h in result] == [expected] * 3 + ['cymbal', 'ride']
    assert (k, n) == (int(expected == 'cymbal'), 1)


def test_all_members_vote_and_order_preserved():
    hits = [Hit(t, 'cymbal', 1) for t in [.5, 0, .02, 3, 4, 4.5]]
    result, k, n = veto.vote_runs(hits, ride_runs(hits), [.99, .5, .5, 1, .2, .2], .65)
    assert [h.inst for h in result] == ['cymbal'] * 4 + ['ride'] * 2
    assert [h.time for h in result] == [h.time for h in hits]
    assert (k, n) == (1, 2)


def test_hash_mismatch_skips_before_features(tmp_path, monkeypatch):
    path = tmp_path / 'broken.json'
    path.write_text('{}')
    monkeypatch.setattr(veto, 'MODEL_PATH', path)
    hits = [Hit(t, 'cymbal', 1) for t in [0, .5]]
    relabeled = ride_runs(hits)
    log = []
    def forbidden():
        pytest.fail('features must not be computed')
    assert veto.apply_veto(hits, relabeled, forbidden, log.append) is relabeled
    assert 'SHA-256 mismatch' in log[0]


def test_feature_failure_skips():
    hits = [Hit(t, 'cymbal', 1) for t in [0, .5]]
    relabeled = ride_runs(hits)
    log = []
    assert veto.apply_veto(hits, relabeled, lambda: np.zeros((5, 84)), log.append) is relabeled
    assert 'skipped' in log[0]


def test_exact_feature_recipe_edges_rounding():
    f = np.random.default_rng(42).normal(size=(37, 84)).astype(np.float16)
    times = [-.1, 0, .005, .015, .025, .36, .5]
    ix = np.asarray([round(float(t)*100) for t in times])[:, None] + np.arange(-5, 15)[None, :]
    x = np.asarray(f[np.clip(ix, 0, len(f)-1)], np.float32) - np.asarray(f, np.float32).mean(0)
    expected = np.concatenate([x[:, :5].mean(1), x[:, 5:10].mean(1), x[:, 10:].mean(1), x.std(1)], 1)
    np.testing.assert_array_equal(veto.context_features(f, times), expected)


def test_defaults_and_legacy(tmp_path):
    assert pipeline.Options().cymbal_veto is True
    assert OptionsPatch().cymbal_veto is True
    assert library.load_options(tmp_path)['cymbal_veto'] is True
    (tmp_path / 'options.json').write_text('{"grid":8}')
    assert library.load_options(tmp_path)['cymbal_veto'] is True
    library.save_options(tmp_path, pipeline.Options(cymbal_veto=False))
    assert library.load_options(tmp_path)['cymbal_veto'] is False


@pytest.mark.parametrize('value', [0, 1, 'true', 'false', None, [], {}])
def test_strict_option(value):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        OptionsPatch(cymbal_veto=value)


@pytest.mark.parametrize('ride_enabled,veto_enabled', [(False, False), (False, True), (True, False), (True, True)])
def test_pipeline_opt_in_cache_and_confidence_order(tmp_path, monkeypatch, ride_enabled, veto_enabled):
    from stickmap import dynamics, fingerprint
    from stickmap.fetch import SongMeta

    np.save(tmp_path / 'activations.mix.npy', np.zeros((1, 5)))
    (tmp_path / 'beats.json').write_text(json.dumps({'beats': [i*.5 for i in range(16)], 'downbeats': [0]}))
    original = [Hit(1, 'cymbal', .7), Hit(1.5, 'cymbal', .8)]
    monkeypatch.setattr(pipeline.models, 'pick_device', lambda _: 'cpu')
    monkeypatch.setattr(pipeline, 'detect_hits', lambda *args: (original, np.zeros((1, 5)), np.empty(0)))
    calls = []
    def separate(audio, target, **kwargs):
        calls.append('stem')
        target.write_bytes(b'fixture')
    def features(stem):
        calls.append('features')
        return np.zeros((200, 84), dtype=np.float16)
    monkeypatch.setattr(pipeline.models, 'separate_drums', separate)
    monkeypatch.setattr(veto, 'adtof_input', features)
    monkeypatch.setattr(veto, 'predict_proba', lambda model, x: np.tile([.1, .9], (len(x), 1)))
    def confidence(hits, *args):
        expected = 'ride' if ride_enabled and not veto_enabled else 'cymbal'
        assert [h.inst for h in hits] == [expected, expected]
        return hits
    monkeypatch.setattr(dynamics, 'annotate_confidence', confidence)
    opts = pipeline.Options(onset_input='mix', pdf=False, ride_runs=ride_enabled, cymbal_veto=veto_enabled)
    logs = []
    for _ in range(2):
        pipeline.run(tmp_path, SongMeta('file', 'test.wav', 'Veto'), opts, log=logs.append)
    active = ride_enabled and veto_enabled
    assert calls == (['stem', 'features'] if active else [])
    cache = pipeline.cache_paths(tmp_path, opts)['adtof_input']
    assert cache.exists() is active
    if active:
        stamps = library.cache_stamps(tmp_path)
        assert stamps[cache.name] == fingerprint.cache_stamp(cache.name, fingerprint.snapshot())
        assert logs.count('cymbal veto: 1 of 1 ride runs -> cymbal') == 2
