"""Offline paired cluster bootstrap of TP/estimate/reference integer counts."""
import numpy as np

CLASSES = ('kick', 'snare', 'tom')
SUBSETS = ('all', 'fast')
DIFFERENCES = (('B', 'A'), ('C', 'B'), ('D', 'C'), ('D', 'A'))
RESAMPLES = 10000
SEED = 20261002


def count_array(counts):
    return np.array([[[counts.get(f'{s}.{c}.{key}', 0) for key in ('tp', 'est', 'ref')]
                      for c in CLASSES] for s in SUBSETS], dtype=np.int64)


def prf(counts):
    counts = np.asarray(counts, dtype=float)
    tp, est, ref = np.moveaxis(counts, -1, 0)
    def ratio(a, b):
        return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)
    return np.stack([ratio(tp, est), ratio(tp, ref), ratio(2 * tp, est + ref)], axis=-1)


def _number(v):
    return float(v) if np.isfinite(v) else None


def _interval(point, samples):
    finite = samples[np.isfinite(samples)]
    return {'difference': _number(point),
            'ci95': [_number(v) for v in np.quantile(finite, [0.025, 0.975])] if len(finite) else [None, None],
            'valid_resamples': int(len(finite))}


def paired_bootstrap(arms, *, resamples=RESAMPLES, seed=SEED):
    """arms[arm][performance_group] -> subset/class/tp-est-ref array.

    Every arm must have EXACTLY the same group roster. One shared multinomial
    draw resamples entire groups for all arms, classes and contrasts. Re-pool
    counts before computing ratios; do not average group F scores.
    """
    if resamples <= 0:
        raise ValueError('positive bootstrap resample count required')
    if set(arms) != {'A', 'B', 'C', 'D'}:
        raise ValueError('paired bootstrap requires exactly A/B/C/D')
    groups = sorted(arms['A'])
    if not groups or any(set(arms[a]) != set(groups) for a in arms):
        raise ValueError('paired arms must have identical nonempty performance groups')
    arrays = {}
    for arm, values in arms.items():
        array = np.array([values[g] for g in groups])
        if (array.shape != (len(groups), 2, 3, 3) or not np.issubdtype(array.dtype, np.integer)
                or np.any(array < 0) or np.any(array[..., 0] > array[..., 1]) or
                np.any(array[..., 0] > array[..., 2])):
            raise ValueError('bootstrap needs valid integer TP/estimate/reference counts')
        arrays[arm] = array
    if any(not np.array_equal(arrays[a][..., 2], arrays['A'][..., 2]) for a in arrays):
        raise ValueError('paired reference counts differ across arms')
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=resamples)
    points = {a: prf(v.sum(axis=0)) for a, v in arrays.items()}
    samples = {a: prf((draws @ v.reshape(len(groups), -1)).reshape(resamples, 2, 3, 3))
               for a, v in arrays.items()}
    differences = {}
    for candidate, baseline in DIFFERENCES:
        name = candidate + '-' + baseline
        differences[name] = {}
        for i, subset in enumerate(SUBSETS):
            result = {}
            for j, cls in enumerate(CLASSES):
                result[cls] = {metric: _interval(points[candidate][i, j, k] - points[baseline][i, j, k],
                                                 samples[candidate][:, i, j, k] - samples[baseline][:, i, j, k])
                               for k, metric in enumerate(('P', 'R', 'F'))}
            result['macro_F'] = _interval(np.mean(points[candidate][i, :, 2] - points[baseline][i, :, 2]),
                                          np.mean(samples[candidate][:, i, :, 2] - samples[baseline][:, i, :, 2], axis=1))
            differences[name][subset] = result
    per_group = {}
    leave_one_out = {}
    for candidate, baseline in DIFFERENCES:
        key = candidate + '-' + baseline
        per_group[key], leave_one_out[key] = {}, {}
        for index, group in enumerate(groups):
            delta = prf(arrays[candidate][index]) - prf(arrays[baseline][index])
            per_group[key][group] = {s: [_number(v) for v in delta[i, :, 2]] for i, s in enumerate(SUBSETS)}
            if len(groups) > 1:
                delta = (prf(arrays[candidate].sum(0) - arrays[candidate][index]) -
                         prf(arrays[baseline].sum(0) - arrays[baseline][index]))
                leave_one_out[key][group] = {s: _number(np.mean(delta[i, :, 2])) for i, s in enumerate(SUBSETS)}
    return {'resamples': resamples, 'seed': seed, 'groups': groups, 'units': 'fraction (multiply by 100 for percentage points)',
            'differences': differences, 'per_group_F_differences': per_group,
            'leave_one_group_out_macro_F': leave_one_out}


def sufficiency(fast_moments, groups, fast_groups=None):
    return {'independent_groups': groups, 'recommended_groups': [20, 30],
            'group_target_met': groups >= 20,
            'classes': {c: {'fast_moments': int(fast_moments.get(c, 0)),
                            'independent_fast_groups': int(fast_groups.get(c, 0)) if fast_groups is not None else None,
                            'first_look_500': fast_moments.get(c, 0) >= 500,
                            'final_minimum_1000': fast_moments.get(c, 0) >= 1000,
                            'final_target_2000': fast_moments.get(c, 0) >= 2000}
                        for c in CLASSES}}
