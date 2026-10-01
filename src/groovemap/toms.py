"""Which tom each tom hit was played on: high, mid or low.

A kit's toms are tuned well apart, so the pitch a drum rings at right after
each hit falls into a few groups per song. Only how the groups lie against
each other names them, never where they lie in hertz: kits are tuned anywhere
from a gospel kit's low, close toms to a jazz kit's high ones. The lowest group
is the floor tom, the highest rack tom is written high and the one below it
mid. In a fast fill the strokes ring into each other: what was already
sounding and any kick or snare that lands with the tom are taken off before
the pitch is read.

Now and then the detector hears a tom in something no drum head rang in, with
next to nothing below a few hundred hertz. Those few are taken out. A song full
of such "toms" is played on a kit tuned that high (a dry, high snare and toms),
and they are left alone.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls
from scipy.signal import find_peaks

from .score import Hit

HIGH, MID, LOW = "tom_high", "tom", "tom_low"
_LEAD = 0.008  # seconds after the onset: the stick attack has no pitch yet
_RING = (0.06, 0.14)  # seconds listened to: up to the next tom, within these limits
_CLEAR = 0.015  # seconds before the onset where the "before" window ends
_CARRY = 0.7  # how much of what rang before is taken off
_BAND = (55.0, 420.0)  # Hz: where a tom's pitch is looked for
# A drum head rings at its pitch and at modes 1.5-1.6 times higher, which in a short
# window can come out louder: the pitch is the lowest peak at least this share of the loudest.
_LOWEST = 0.3
_FIT = (45.0, 2000.0)  # Hz: where a kick or snare landing with the tom is matched and removed
_ALONE = 0.10  # seconds from any tom or the other drum: a kick or snare to learn its sound from
_HEAD = (60.0, 450.0)  # Hz: where a struck drum head rings
_HEARD = (60.0, 12000.0)  # Hz: the whole of a hit's sound
_HEADLESS = 0.1  # a "tom" with less than this share of its sound where a head rings was not played on a drum
_KIT = (24, 0.15)  # this many headless "toms", and this share of the song's toms: the kit is tuned that high
_SPREAD = 0.9  # semitones: how far one drum's pitch wanders between hits
_APART = 2.5  # semitones: two toms are tuned at least this far apart
# Semitones: the two lowest groups are one floor tom when this close, or within the
# second distance when the lower one has under _WEAK as many hits as the upper.
_SAME_FLOOR = (3.0, 7.0)
_WEAK = 0.5
_APART_HIGH = 5.0  # semitones: a top group this far above the next and as weak is not a rack tom


class _Stem:
    """Spectra of short stretches of the drum stem, fine enough to tell toms 10 Hz apart."""

    def __init__(self, drums_wav: Path) -> None:
        self.audio, self.sr = sf.read(drums_wav, dtype="float32", always_2d=True)
        self.size = 1 << int(np.ceil(np.log2(self.sr / 2.7)))
        self.freqs = np.fft.rfftfreq(self.size, 1 / self.sr)

    def spectrum(self, start: float, length: float) -> np.ndarray:
        a = round(start * self.sr)
        b = a + round(length * self.sr)
        piece = np.zeros((b - a, self.audio.shape[1]), dtype=np.float32)
        lo, hi = max(0, a), min(len(self.audio), b)
        if hi > lo:
            piece[lo - a:hi - a] = self.audio[lo:hi]
        power = np.abs(np.fft.rfft(piece * np.hanning(len(piece))[:, None], n=self.size, axis=0)) ** 2
        return np.sqrt(power.mean(axis=1))

    def struck(self, time: float, length: float = 0.12) -> np.ndarray:
        """What the hit at `time` added: the spectrum after it, less most of what rang before."""
        before = self.spectrum(time - length - _CLEAR, length)
        return np.maximum(self.spectrum(time + _LEAD, length) - _CARRY * before, 0)


def _other_drums(stem: _Stem, hits: list[Hit]) -> np.ndarray:
    """The song's typical kick and snare spectra, one per column, from hits that stand alone."""
    sounds = []
    for inst, other in (("kick", "snare"), ("snare", "kick")):
        busy = np.array([hit.time for hit in hits if hit.inst in (MID, other)])
        alone = [
            hit for hit in hits
            if hit.inst == inst and (not len(busy) or np.abs(busy - hit.time).min() > _ALONE)
        ]
        spectra = np.array([stem.struck(hit.time) for hit in sorted(alone, key=lambda h: -h.strength)[:40]])
        if len(spectra):
            spectra /= np.maximum(np.linalg.norm(spectra, axis=1, keepdims=True), 1e-10)
            sounds.append(np.median(spectra, axis=0))
    return np.array(sounds).T if sounds else np.zeros((len(stem.freqs), 0))


def _pitch(stem: _Stem, time: float, next_tom: float | None, others: np.ndarray) -> float | None:
    """The frequency the tom hit at `time` rings at."""
    length = _RING[1] if next_tom is None else min(_RING[1], max(_RING[0], next_tom - time - 0.012))
    struck = stem.struck(time, length)
    if not struck.any():
        return None
    if others.shape[1]:  # take off as much kick and snare as fits
        fit = (stem.freqs >= _FIT[0]) & (stem.freqs <= _FIT[1])
        weight = 1 / np.sqrt(np.maximum(struck[fit], max(struck.max() * 0.03, 1e-10)))
        amounts = nnls(others[fit] * weight[:, None], struck[fit] * weight)[0]
        struck = np.maximum(struck - others @ amounts, 0)
    band = (stem.freqs >= _BAND[0]) & (stem.freqs <= _BAND[1])
    smooth = gaussian_filter1d(struck, 1)  # over the whole spectrum, so a peak at the band's edge is one
    if not smooth[band].any():
        return None
    loudest = smooth[band].max()
    peaks, _ = find_peaks(smooth, height=_LOWEST * loudest)
    peaks = peaks[band[peaks]]
    return float(stem.freqs[peaks[0]] if len(peaks) else stem.freqs[band][np.argmax(smooth[band])])


def _headless(stem: _Stem, time: float) -> bool:
    """Whether the hit at `time` has next to nothing where a drum head rings."""
    struck = stem.struck(time, _RING[0])
    heard = struck[(stem.freqs >= _HEARD[0]) & (stem.freqs < _HEARD[1])].sum()
    head = struck[(stem.freqs >= _HEAD[0]) & (stem.freqs < _HEAD[1])].sum()
    return bool(heard > 0 and head < _HEADLESS * heard)


def _without_strays(stem: _Stem, hits: list[Hit]) -> list[Hit]:
    """The hits, without the few "toms" that no drum head rang in.

    On an ordinary kit these are the detector's mistakes (on MDB all ten were;
    on the J-Rod transcription eight of nine). A song full of them is played on
    a kit tuned high enough to ring above the band, and nothing is taken out.
    """
    toms = [i for i, hit in enumerate(hits) if hit.inst == MID]
    strays = {i for i in toms if _headless(stem, hits[i].time)}
    if not strays or (len(strays) >= _KIT[0] and len(strays) >= _KIT[1] * len(toms)):
        return hits
    return [hit for i, hit in enumerate(hits) if i not in strays]


def _groups(pitches: list[float]) -> list[float]:
    """The pitches the song's toms ring at, lowest first: the peaks of where the hits crowd.

    A peak needs a few hits behind it; in a song with only a few toms every
    peak counts. Two low peaks close together, or a weak one just under a
    stronger one, are one floor tom whose pitch wobbles. A weak peak well
    above all the others is not a rack tom (a snare or a ringing mode).
    """
    semitones = 12 * np.log2(pitches)
    grid = np.arange(semitones.min() - 3 * _SPREAD, semitones.max() + 3 * _SPREAD + 0.075, 0.075)
    density = np.exp(-0.5 * ((grid[:, None] - semitones[None, :]) / _SPREAD) ** 2).sum(axis=1)
    peaks, _ = find_peaks(density, distance=round(_APART / 0.075))
    enough = max(3, 0.01 * len(pitches))
    supported = [p for p in peaks if np.sum(np.abs(semitones - grid[p]) < 1.5 * _SPREAD) >= enough] or list(peaks)
    centres = np.sort(grid[sorted(supported, key=lambda p: -density[p])[:5]]) if supported else np.array([np.median(semitones)])

    def support(centres: np.ndarray) -> np.ndarray:
        return np.bincount(np.abs(semitones[:, None] - centres).argmin(axis=1), minlength=len(centres))

    counts = support(centres)
    while len(centres) >= 2:
        gap = centres[1] - centres[0]
        if not (gap < _SAME_FLOOR[0] or (gap < _SAME_FLOOR[1] and counts[0] < _WEAK * counts[1])):
            break
        centres = np.r_[np.average(centres[:2], weights=counts[:2]), centres[2:]]
        counts = support(centres)
    while len(centres) >= 3:
        gap = centres[-1] - centres[-2]
        if not (gap > _APART_HIGH and counts[-1] < _WEAK * counts[:-1].max()):
            break
        centres = centres[:-1]
        counts = support(centres)
    return [float(2 ** (c / 12)) for c in centres]


def _sizes(count: int) -> list[str]:
    """The size of each of `count` toms, lowest first.

    As drummers write them: the lowest is a floor tom, one rack tom is written
    high, and with more the smallest is high and the next mid. A five-tom kit
    has two floor toms; any other larger kit has its extra toms low.
    """
    if count <= 2:
        return [LOW, HIGH][:count]
    if count == 5:
        return [LOW, LOW, MID, HIGH, HIGH]
    return [LOW] * (count - 2) + [MID, HIGH]


def tom_sizes(drums_wav: Path, hits: list[Hit]) -> list[Hit]:
    """The hits, with each "tom" renamed to the tom it was played on; stray ones are dropped."""
    if not any(hit.inst == MID for hit in hits):
        return hits
    stem = _Stem(drums_wav)
    hits = _without_strays(stem, hits)
    toms = [i for i, hit in enumerate(hits) if hit.inst == MID]
    others = _other_drums(stem, hits)
    times = [hits[i].time for i in toms]
    pitches = {
        # The window stops at the next tom, unless that one is struck together with this one.
        i: _pitch(stem, time, next((later for later in times[n + 1:] if later > time + 0.025), None), others)
        for n, (i, time) in enumerate(zip(toms, times))
    }
    known = [pitch for pitch in pitches.values() if pitch]
    centres = _groups(known) if known else []
    sizes = _sizes(len(centres))
    sized = list(hits)
    for i, pitch in pitches.items():
        if not pitch:
            continue  # stays a plain (mid) tom
        nearest = min(range(len(centres)), key=lambda k: abs(np.log2(pitch / centres[k])))
        sized[i] = replace(hits[i], inst=sizes[nearest])
    return sized
