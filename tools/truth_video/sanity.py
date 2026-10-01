"""Cheap per-song audio sanity flags that do not use the detector.

A drumless backing track separates into a near-silent drum stem whose attack
list is noise; a mis-fetched or truncated audio has far fewer attacks than
the notation has moments. Both produced accepted-looking truth before. These
flags are diagnostic: the alignment gates still decide every system.
"""
import json
from pathlib import Path
import numpy as np
import bands

THRESHOLDS = dict(silent_rms_db=-40., silent_active_fraction=.05, active_block_db=-40., attack_deficit_ratio=.5)


def stem_stats(song):
    path = bands.stem_path(song)
    if path is None:
        return None
    y, sr = bands.load_mono(path)
    block = int(sr*.1); n = len(y)//block
    if n == 0:
        return dict(source=path.name, seconds=len(y)/sr, rms_db=-120., active_fraction=0.)
    rms = np.sqrt((y[:n*block].reshape(n, block)**2).mean(1))
    db = 20*np.log10(rms+1e-9)
    return dict(source=path.name, seconds=round(len(y)/sr, 2), rms_db=round(float(20*np.log10(np.sqrt((y**2).mean())+1e-9)), 1),
                active_fraction=round(float((db > THRESHOLDS['active_block_db']).mean()), 3),
                p90_block_db=round(float(np.percentile(db, 90)), 1))


def attack_coverage(song, systems):
    """Attacks inside the notated systems' intervals against their notated moments."""
    from align import moments
    attacks = np.unique(np.load(Path(song)/'attacks.htdemucs.npy').astype(float))
    notated = inside = 0
    for s in systems:
        x, _ = moments(s)
        if len(x) < 6:
            continue
        lo, hi = s['interval']
        notated += len(x); inside += int(((attacks >= lo-.65) & (attacks <= hi+.65)).sum())
    return dict(attacks_total=int(len(attacks)), attacks_in_notated_intervals=inside, notated_moments=notated,
                ratio=round(inside/notated, 3) if notated else None)


def check(song, systems):
    stem = stem_stats(song)
    coverage = attack_coverage(song, systems)
    flags = []
    if stem is None:
        flags.append('no drum stem')
    elif stem['rms_db'] < THRESHOLDS['silent_rms_db'] or stem['active_fraction'] < THRESHOLDS['silent_active_fraction']:
        flags.append('drum stem nearly silent (drumless or non-drum audio)')
    if coverage['ratio'] is not None and coverage['ratio'] < THRESHOLDS['attack_deficit_ratio']:
        flags.append('audio attacks far fewer than notated moments')
    return dict(stem=stem, attacks=coverage, thresholds=THRESHOLDS, flags=flags)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('song', type=Path); p.add_argument('strip', type=Path, nargs='?')
    a = p.parse_args()
    systems = json.loads(a.strip.read_text()) if a.strip else []
    print(json.dumps(check(a.song, systems), indent=1))
