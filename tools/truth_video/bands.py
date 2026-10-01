"""Band-limited onset strength from the drum stem: instrument-aware phase evidence.

The notation is trusted for WHICH drums play; the audio decides WHEN. A notated
kick must coincide with a low-band onset and a notated snare with a mid-band
broadband onset. That agreement is computed from the separated drum stem with a
plain STFT log-magnitude flux; it never consults the detector's drum labels, so
the exported truth stays independent of the model it evaluates.
"""
import json
from pathlib import Path
import subprocess
import numpy as np

VERSION = 2
HOP_SECONDS = .005
N_FFT = 2048
# Kick fundamentals sit below ~110 Hz; a snare's body/wires and stick attack
# dominate 250-2500 Hz where a kick's log-flux is small. Hats live above 5 kHz.
BANDS = {'low': (35, 110), 'mid': (250, 2500), 'high': (5000, 16000)}
ORDER = ('low', 'mid', 'high')
INSTRUMENT_BAND = {"kick": "low", "snare": "mid"}  # see RULES
CACHE = 'onsets.bands.htdemucs.npy'


def stem_path(song):
    song = Path(song)
    for name in ('drums.htdemucs.wav', 'drums.htdemucs.opus'):
        if (song/name).is_file():
            return song/name
    return None


def load_mono(path, sr=44100):
    path = Path(path)
    if path.suffix == '.wav':
        import soundfile as sf
        y, rate = sf.read(str(path), dtype='float32', always_2d=True)
        return y.mean(1), rate
    run = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(path), '-f', 'f32le', '-ac', '1', '-ar', str(sr), '-'],
                         capture_output=True, check=True)
    return np.frombuffer(run.stdout, np.float32).copy(), sr


def onset_strength(y, sr, hop_seconds=HOP_SECONDS, n_fft=N_FFT, bands=BANDS):
    """Per-band positive log-magnitude flux at 5 ms hops, each band scaled by its 99th percentile."""
    y = np.asarray(y, np.float32)
    peak = float(np.percentile(np.abs(y), 99.9)) if len(y) else 0
    if peak > 0:
        y = y/peak
    hop = int(round(sr*hop_seconds))
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft-len(y)))
    frames = np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop]
    window = np.hanning(n_fft).astype(np.float32)
    # Spectrum scaling: a full-scale sine reads as its amplitude, so the log
    # compression point (1/100 of full scale) is the same for every song.
    window *= 2/window.sum()
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    selection = {name: (freqs >= lo) & (freqs < hi) for name, (lo, hi) in bands.items()}
    flux = {name: np.zeros(len(frames), np.float32) for name in bands}
    previous = None
    block = 4096
    for start in range(0, len(frames), block):
        mag = np.log1p(100*np.abs(np.fft.rfft(frames[start:start+block]*window, axis=1)))
        if previous is None:
            previous = mag[:1]
        stacked = np.concatenate([previous, mag])
        diff = np.maximum(stacked[1:]-stacked[:-1], 0)
        for name, sel in selection.items():
            flux[name][start:start+len(mag)] = diff[:, sel].sum(1)
        previous = mag[-1:]
    t = (np.arange(len(frames))*hop+n_fft/2)/sr
    out = {'t': t.astype(np.float32)}
    for name in bands:
        scale = float(np.percentile(flux[name], 99))
        out[name] = flux[name]/scale if scale > 0 else flux[name]
    return out


def peaks(env, t, distance_seconds=.04):
    """The flux kept only at local maxima: a slow low-end swell is one onset, not a tail of onsets."""
    from scipy.signal import find_peaks
    step = float(t[1]-t[0]) if len(t) > 1 else HOP_SECONDS
    idx, _ = find_peaks(env, distance=max(1, int(round(distance_seconds/step))))
    out = np.zeros_like(env)
    out[idx] = env[idx]
    return out


def compute(song):
    path = stem_path(song)
    if path is None:
        return None
    y, sr = load_mono(path)
    return onset_strength(y, sr)


TOLERANCE = .04


def with_peaks(out, tol=TOLERANCE):
    """Add peak envelopes and +-tol sliding maxima so per-note lookups are O(1)."""
    from scipy.ndimage import maximum_filter1d
    t = out['t']
    k = int(round(tol/float(t[1]-t[0]))) if len(t) > 1 else 0
    for name in ORDER:
        out[name+'_peaks'] = peaks(out[name], t)
        for key in (name, name+'_peaks'):
            out[key+'_wmax'] = maximum_filter1d(out[key], 2*k+1, mode='nearest')
    out['tolerance'] = tol
    return out


def load(song, cache=True):
    """Cached band onsets for a harvested song; None when no drum stem exists."""
    song = Path(song)
    p = song/CACHE
    if p.exists():
        try:
            data = np.load(p)
            if data.shape[0] == len(ORDER)+2 and int(data[0, 0]) == VERSION:
                return with_peaks({'t': data[1], **{name: data[i+2] for i, name in enumerate(ORDER)}})
        except (ValueError, OSError):
            pass
    out = compute(song)
    if out is None:
        return None
    if cache:
        try:
            data = np.vstack([np.full_like(out['t'], VERSION), out['t'], *[out[name] for name in ORDER]]).astype(np.float32)
            tmp = p.with_name(p.stem+'.tmp.npy')
            np.save(tmp, data)
            tmp.rename(p)
        except OSError:
            pass
    return with_peaks(out)


def window_max(env, t, times, tol=.04):
    idx = np.searchsorted(t, np.asarray(times, float))
    k = int(round(tol/float(t[1]-t[0]))) if len(t) > 1 else 0
    out = np.zeros(len(idx))
    for n, i in enumerate(idx):
        lo, hi = max(0, i-k), min(len(env), i+k+1)
        if hi > lo:
            out[n] = env[lo:hi].max()
    return out


# A kick needs a low-band onset peak that is not dwarfed by the mid band (a
# snare alone leaks little below 110 Hz); a snare needs a mid-band peak that is
# not dwarfed by the low band (a kick's click alone stays below its own low
# flux). Unisons pass both. Thresholds are fractions of each band's 99th
# percentile flux for the song.
RULES = {'kick': dict(band='low', other='mid', peak=.3, ratio=.8),
         'snare': dict(band='mid', other='low', peak=.4, ratio=.8)}


def evidence(bands, times, inst, tol=TOLERANCE):
    rule = RULES[inst]
    if bands.get('tolerance') == tol and rule['band']+'_wmax' in bands:
        idx = np.clip(np.searchsorted(bands['t'], np.asarray(times, float)), 0, len(bands['t'])-1)
        own = bands[rule['band']+'_wmax'][idx]
        peak = bands[rule['band']+'_peaks_wmax'][idx]
        other = bands[rule['other']+'_wmax'][idx]
    else:
        own = window_max(bands[rule['band']], bands['t'], times, tol)
        peak = window_max(bands[rule['band']+'_peaks'], bands['t'], times, tol)
        other = window_max(bands[rule['other']], bands['t'], times, tol)
    return (peak > rule['peak']) & (own >= rule['ratio']*other)


def agreement(bands, times, insts, tol=TOLERANCE):
    """Fraction of notated kicks on kick-like onsets and snares on snare-like onsets.

    `insts` holds one set of instrument names per moment. Moments without a
    kick or snare carry no phase evidence and are not counted.
    """
    times = np.asarray(times, float)
    result = {}
    hits = total = 0
    for inst in RULES:
        mask = np.array([inst in q for q in insts], bool) if len(insts) else np.zeros(0, bool)
        n = int(mask.sum())
        if n:
            h = int(evidence(bands, times[mask], inst, tol).sum())
            result[inst] = h/n
            hits += h; total += n
        else:
            result[inst] = None
        result[f'{inst}_count'] = n
    result['pooled'] = hits/total if total else None
    result['count'] = total
    return result


def attack_agreement(times, attacks, tol=TOLERANCE):
    """Fraction of times within tol of any attack (nearest-neighbour, class-agnostic)."""
    times = np.asarray(times, float)
    if not len(times) or attacks is None or not len(attacks):
        return 0.
    attacks = np.asarray(attacks, float)
    i = np.clip(np.searchsorted(attacks, times), 1, len(attacks)-1)
    nearest = np.minimum(abs(attacks[i]-times), abs(attacks[i-1]-times))
    return float((nearest <= tol).mean())


MIN_INSTRUMENT_NOTES = 4   # an instrument's margin counts only with this many notated moments
MIN_GAIN = .15             # one instrument must beat the alias by this much
MAX_DEFICIT_NOTES = 2      # an instrument contradicts the alignment when it trails an alias by more than MIN_GAIN and more than this many moments


def compare(here, there):
    """Per-instrument margins of an alignment over a shifted copy of itself.

    Pooled numbers dilute the instrument that carries the evidence (four-on-
    the-floor kicks agree with themselves one beat later; the snares do not),
    so each instrument is judged on its own: a shift is beaten when some
    instrument gains at least MIN_GAIN and none trails it by more than
    MIN_GAIN and more than MAX_DEFICIT_NOTES moments.
    """
    margins = {}
    for inst in RULES:
        n = here[f'{inst}_count']
        if n >= MIN_INSTRUMENT_NOTES and here[inst] is not None and there[inst] is not None:
            margins[inst] = here[inst]-there[inst]
    if not margins:
        return dict(margins={}, gain=None, contradiction=False)
    allowed = {inst: max(MIN_GAIN, MAX_DEFICIT_NOTES/here[f'{inst}_count']) for inst in margins}
    # A deficit on one instrument is a contradiction unless another instrument
    # compensates by at least its own allowance (kicks are neutral or noisy in
    # many grooves; a snare row that flips from 100% to 0% is not).
    deficit = any(margins[i] < -allowed[i] for i in margins)
    compensated = any(margins[i] >= allowed[i] for i in margins)
    return dict(margins=margins, gain=max(margins.values()), contradiction=deficit and not compensated)


def phase_check(bands, times, insts, attacks=None, minimum_shift=.09, maximum_shift=.45, step=.01, alias_ratio=.75, **kw):
    """Compare the alignment with every reachable time shift that is a rhythmic alias.

    A shift competes only when the shifted moments still land on attacks about
    as well as the unshifted ones (`alias_ratio`, floor 50%): that is exactly
    the case the class-agnostic gates cannot separate. `margin` is the gain of
    the best instrument over the hardest competing shift; `contradicted_by` is
    a shift that puts clearly more notated kicks or snares on their onset band.
    """
    times = np.asarray(times, float)
    here = agreement(bands, times, insts, **kw)
    base = attack_agreement(times, attacks) if attacks is not None else None
    weakest = None; contradiction = None; candidates = 0
    for shift in np.arange(-maximum_shift, maximum_shift+1e-9, step):
        if abs(shift) < minimum_shift:
            continue
        if base is not None and attack_agreement(times+shift, attacks) < max(.5, alias_ratio*base):
            continue
        candidates += 1
        there = agreement(bands, times+shift, insts, **kw)
        c = compare(here, there)
        if c['gain'] is None:
            continue
        entry = dict(shift=float(round(shift, 3)), pooled=there['pooled'], **c)
        if weakest is None or c['gain'] < weakest['gain']:
            weakest = entry
        if c['contradiction'] and (contradiction is None or min(c['margins'].values()) < min(contradiction['margins'].values())):
            contradiction = entry
    return dict(pooled=here['pooled'], kick=here['kick'], snare=here['snare'], count=here['count'],
                kick_count=here['kick_count'], snare_count=here['snare_count'], alias_candidates=candidates,
                best_shift=weakest['shift'] if weakest else None, best_shift_pooled=weakest['pooled'] if weakest else None,
                best_shift_margins=weakest['margins'] if weakest else None,
                margin=weakest['gain'] if weakest else None,
                contradicted_by=contradiction['shift'] if contradiction else None,
                contradiction_margins=contradiction['margins'] if contradiction else None)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('song', type=Path); p.add_argument('--truth', type=Path)
    a = p.parse_args()
    bands = load(a.song)
    if bands is None:
        raise SystemExit('no drum stem')
    if a.truth:
        notes = json.loads(a.truth.read_text())
        times = np.array([n['time'] for n in notes]); insts = [{n['inst']} for n in notes]
        print(json.dumps(phase_check(bands, times, insts), indent=1))
