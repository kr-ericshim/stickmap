"""Score the current detector against ground truth read from a transcription video (see scan.py, truth.py)."""
import sys, json
from pathlib import Path
from collections import Counter
import numpy as np
from align import match, build_truth

def events(strip, dx, dt):
    """Timed notes: when the cursor reaches each head (dx pixels from its centre), plus the video's lag dt."""
    out = []
    for s in strip:
        x, t = np.array(s["cursor_x"], dtype=float), np.array(s["cursor_t"])
        if not len(x): continue
        for n in s["notes"]:
            if x[0] - 30 <= n["x"] + dx <= x[-1] + 30:
                out.append({**n, "time": float(np.interp(n["x"] + dx, x, t)) + dt})
    return sorted(out, key=lambda e: e["time"])

def fit(strip, attacks):
    """The pixel and time offsets at which the most notes fall on an attack in the audio (within 20 ms).

    Against every attack, whatever the drum: the detector's own labels would
    bias the fit on exactly the songs it gets wrong.
    """
    best = (-1, 0, 0)
    for dx in range(-40, 41, 2):
        ref = sorted({round(e["time"], 3) for e in events(strip, dx, 0)})
        for dt in np.arange(-0.4, 0.401, 0.01):
            n = len(match([t + dt for t in ref], attacks, 0.02))
            if n > best[0]: best = (n, dx, float(dt))
    return best

def confusion(truth, hits, tol=0.05):
    """For each notated drum: what the detector has at that moment."""
    times = np.array([h.time for h in hits])
    table = {}
    for e in truth:
        near = [family(hits[i].inst) for i in np.where(np.abs(times - e["time"]) <= tol)[0]]
        label = e["inst"] + ("(ghost)" if e["ghost"] and e["inst"] == "snare" else "")
        row = table.setdefault(label, Counter())
        row[family(e["inst"]) if family(e["inst"]) in near else (Counter(x for x in near if x in ("kick", "snare", "tom")).most_common(1) or [("hat/cym only" if near else "nothing", 0)])[0][0]] += 1
    return table

def detect(song, sensitivity=1.0, stage="full"):
    from groovemap import models
    from groovemap.fills import fill_snares, tom_runs
    from groovemap.toms import tom_sizes
    from groovemap.score import Hit
    act, mix = np.load(song / "activations.drums-htdemucs.npy"), np.load(song / "activations.mix.npy")
    thr = [t / sensitivity for t in models.DEFAULT_THRESHOLDS]
    if stage == "adtof":
        import adtof_pytorch as adtof
        peaks = adtof.PeakPicker(thresholds=thr, fps=100).pick(act[None], labels=adtof.LABELS_5)[0]
        return sorted([Hit(float(t), inst, 1.0) for lab, inst in zip(adtof.LABELS_5, models.ADTOF_CLASSES) for t in peaks[lab]], key=lambda h: h.time)
    snare, attacks = np.load(song / "snare.htdemucs.npy"), np.load(song / "attacks.htdemucs.npy")
    hits = models.pick_hits(act, thr, mix=mix, snare=snare)
    hits = fill_snares(tom_runs(hits, act, mix, attacks, thr), act, snare, attacks)
    if stage == 'pre-toms':
        return hits
    stem = song / "drums.htdemucs.wav"
    if not stem.exists():  # compacted harvest songs keep the stem as Opus
        stem = stem.with_suffix(".opus")
    return tom_sizes(stem, hits)

def family(inst):
    if inst.startswith('tom'): return 'tom'
    if inst in ('ride', 'ride_bell', 'bell', 'crash', 'cymbal_unknown'): return 'cymbal'
    if inst == 'side_stick': return 'snare'
    return inst


def score(truth, hits, label='', intervals=None):
    """Score only accepted systems; rejected gaps are not detector false positives."""
    if not truth:
        print(f'{label}: no usable ground truth; precision/recall unavailable')
        return {}
    if intervals is None:
        intervals = [[truth[0]['time'], truth[-1]['time']]]
    inside = lambda t: any(a-.05 <= t <= b+.05 for a,b in intervals)
    out, line = {}, f'{label:24s}|'
    for c in ('kick','snare','tom','hihat','cymbal','hat/cymbal'):
        belongs=lambda inst: family(inst) in ('hihat','cymbal') if c=='hat/cymbal' else family(inst)==c
        ref = [e for e in truth if belongs(e['inst'])]
        # Multiple fine cymbal labels at one moment are one family event.
        ref = list({round(e['time'],5):e for e in ref}.values())
        est = [h for h in hits if belongs(h.inst) and inside(h.time)]
        pairs = match([e['time'] for e in ref], [h.time for h in est])
        line += f' {c} P {len(pairs)/max(1,len(est)):.2f} R {len(pairs)/max(1,len(ref)):.2f} ({len(pairs)}/{len(est)} est, {len(ref)} ref) |'
        out[c] = (pairs,ref,est)
    pairs,ref,est=out['snare'];got={i for i,_ in pairs}
    ghosts=[i for i,e in enumerate(ref) if e['ghost']]
    if ghosts:
        line+=f' ghost R {sum(i in got for i in ghosts)}/{len(ghosts)}'
    print(line)
    return out


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('song',type=Path);p.add_argument('strip',type=Path)
    p.add_argument('--alignment-only',action='store_true')
    p.add_argument('--reuse-truth',action='store_true')
    p.add_argument('--tier',choices=['strict','scoring'],default='strict')
    a=p.parse_args();folder=a.strip.parent
    if a.reuse_truth:
        suffix='.scoring' if a.tier=='scoring' else ''
        truth=json.loads((folder/f'truth{suffix}.json').read_text())
        report=json.loads((folder/f'alignment{suffix}.json').read_text())
    else:
        truth,report=build_truth(a.song,folder)
    if not a.alignment_only and truth:
        hits=detect(a.song)
        score(truth,detect(a.song,stage='adtof'),'ADTOF picker only',report['intervals'])
        out=score(truth,hits,'current detector',report['intervals'])
        for sensitivity in (1.7,3.0):
            score(truth,detect(a.song,sensitivity),f'current, sensitivity {sensitivity}',report['intervals'])
        for label,row in sorted(confusion(truth,hits).items()):
            print(f'   notated {label:13s} -> detector has: '+', '.join(f'{k} {v}' for k,v in row.most_common()))
        pairs,ref,est=out['snare']
        print('   snare (truth ghost?, ours ghost?):',dict(Counter((ref[i]['ghost'],est[j].ghost) for i,j in pairs)))
        pairs,ref,est=out['tom']
        print('   tom (truth staff position -> our size):',dict(Counter((ref[i]['pos'],est[j].inst) for i,j in pairs)))
        delta=np.array([est[j].time-ref[i]['time'] for c in ('kick','snare') for pairs,ref,est in [out[c]] for i,j in pairs])
        if len(delta):
            print(f'   timing: median {1000*np.median(delta):+.0f} ms, IQR {1000*np.percentile(delta,25):+.0f}..{1000*np.percentile(delta,75):+.0f}')
    elif not truth:
        score([],[],'current detector')
    print('   fine labels/techniques are not separately scored here; heuristic ride_bell outputs are diagnosed in correspondence.*.json')
