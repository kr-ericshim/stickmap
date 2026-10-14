"""Offline trusted-joblib exporter; never imported by the application."""
import hashlib
import json
from pathlib import Path

FIELDS = ('feature_idx', 'num_threshold', 'missing_go_to_left', 'left', 'right', 'is_leaf', 'value')


def export_model(model, source_sha256, threshold=0.65):
    if list(model.classes_) != [0, 1] or model.n_trees_per_iteration_ != 1:
        raise ValueError('Only binary 0=ride, 1=crash models supported')
    trees = []
    for iteration in model._predictors:
        nodes = iteration[0].nodes
        if nodes['is_categorical'].any():
            raise ValueError('Categorical splits are unsupported')
        trees.append({name: nodes[name].tolist() for name in FIELDS})
    return dict(format_version=1, baseline=float(model._baseline_prediction[0, 0]),
                trees=trees, classes_=model.classes_.tolist(), threshold=threshold,
                vote='mean', n_features=int(model.n_features_in_),
                feature_recipe_version='w5-f16-f32-context-v1',
                source_joblib_sha256=source_sha256)


if __name__ == '__main__':
    import argparse
    import joblib
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    frozen = json.loads((args.source.parent / 'FROZEN.json').read_text())['sha256']
    assert source_hash == frozen['scratch/sx_diag/w5/model.joblib']
    threshold = json.loads((args.source.parent / 'threshold.json').read_text())['threshold']
    payload = export_model(joblib.load(args.source), source_hash, threshold)
    args.output.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False) + '\n')
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())
