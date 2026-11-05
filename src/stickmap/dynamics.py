"""How hard each snare stroke was played, measured against the song's own kit.

ADTOF's activation says how sure the model is, not how loud the drum was, and
it hardly responds to ghost notes. Here the drum stem is explained as a sum of
this kit's own sounds: one short spectrogram patch per drum, taken from the
hits ADTOF is sure of, plus a few free patches that soak up bleed and ringing
(NMFD, a convolutive non-negative matrix factorisation). How much of the snare
patch each moment needs grows linearly with loudness, so a ghost note shows as
a small but distinct peak.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from collections.abc import Sequence

import numpy as np
import soundfile as sf

from .models import ADTOF_FPS, picked_frames
from .score import Hit


def annotate_confidence(hits: Sequence[Hit], activations: np.ndarray, response: np.ndarray | None) -> list[Hit]:
    """Read existing snare identity/share evidence; never recompute loudness.

    A near-silent response coinciding with a tom is especially unreliable.
    Confidence expresses support for a soft-snare label, not its volume.
    """
    out = []
    toms = np.asarray([h.time for h in hits if h.inst.startswith("tom")])
    for hit in hits:
        if hit.inst != "snare":
            out.append(hit)
            continue
        f = round(hit.time * ADTOF_FPS)
        a = max(0, f - 2)
        b = min(len(activations), f + 3)
        identity = float(activations[a:b, 1].max()) if b > a else 0.0
        share = 0.0
        if response is not None and response.ndim == 2 and response.shape[1] >= 2 and a < len(response):
            segment = response[a:min(b, len(response))]
            if len(segment):
                share = float(segment[int(np.argmax(segment[:, 0])), 1])
        confidence = max(identity, share)
        if len(toms) and np.min(np.abs(toms - hit.time)) < 0.025 and share < 0.5:
            confidence *= 0.4
        out.append(replace(hit, confidence=float(np.clip(confidence, 0, 1))))
    return out


def ghost_states(hits: Sequence[Hit]) -> dict[tuple[float, str], str]:
    """Only clearly separated soft hits earn parentheses; no percentile quota.

    Both loudness and its mapped velocity must be far below this song's typical
    snare. A broad continuous distribution does not itself prove two clusters:
    marginal soft candidates remain unknown. Detector strength/velocity survive.
    """
    snares = [h for h in hits if h.inst == "snare"]
    if not snares:
        return {}
    typical = float(np.median([h.strength for h in snares]))
    p90 = float(np.percentile([h.strength for h in snares], 90))
    velocities = [40 + 70 * min(1, h.strength / p90) if p90 > 0 else 40 for h in snares]
    velocity_limit = 40 + (float(np.median(velocities)) - 40) / 3
    states = {}
    toms = [h.time for h in hits if h.inst.startswith("tom")]
    for hit, velocity in zip(snares, velocities):
        soft = hit.strength < typical / 3 and velocity <= velocity_limit
        trusted = hit.confidence is not None and hit.confidence >= 0.65
        # Small manually labelled examples can carry an explicit ghost flag.
        if hit.confidence is None and len(snares) < 16:
            trusted = hit.ghost and not any(abs(t - hit.time) < 0.025 for t in toms)
        states[hit.time, hit.inst] = "ghost" if soft and trusted else "unknown" if soft or hit.ghost else "normal"
    return states

_PATCH = 10  # frames per drum sound: 100 ms
_FREE = 3  # patches left free to explain what the drum sounds do not
_ITERATIONS = 60
_MIN_SNARES = 4


def _spectrogram(drums_wav: Path) -> np.ndarray:
    """Mel magnitudes, shape [bands, frames], one frame per ADTOF frame."""
    import librosa

    audio, sr = sf.read(drums_wav, dtype="float32", always_2d=True)
    mel = librosa.feature.melspectrogram(
        y=audio.mean(axis=1), sr=sr, n_fft=2048, hop_length=sr // ADTOF_FPS, n_mels=128, fmin=30, fmax=16000, power=1.0
    )
    return mel.astype(np.float32) + 1e-6


def _patches(spectrogram: np.ndarray, peaks: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """The typical onset of each drum, shape [bands, drums, _PATCH], and which drums were found.

    The median over its hits, preferring hits no other drum coincides with;
    what was already sounding just before each hit is taken off first.
    """
    frames = spectrogram.shape[1]
    patches, names = [], []
    for inst, own in peaks.items():
        if not len(own):
            continue
        # By instrument, not by frame: a kick on the very frame of a snare still counts as company.
        others = np.concatenate([found for other, found in peaks.items() if other != inst])
        alone = own[np.abs(own[:, None] - others[None, :]).min(axis=1) > 3] if len(others) else own
        use = alone if len(alone) >= 6 else own
        use = use[(use >= 5) & (use - 1 + _PATCH <= frames)]
        if len(use) < 3:
            continue
        onsets = np.stack([spectrogram[:, f - 1:f - 1 + _PATCH] for f in use])
        before = np.stack([spectrogram[:, f - 4:f - 1].mean(axis=1) for f in use])[:, :, None]
        onsets = np.maximum(onsets - before, 0) + 1e-6
        onsets /= onsets.max(axis=(1, 2), keepdims=True)
        patches.append(np.median(onsets, axis=0))
        names.append(inst)
    if not patches:
        return np.zeros((len(spectrogram), 0, _PATCH), dtype=np.float32), names
    return np.stack(patches, axis=1), names


def _shift(rows: np.ndarray, frames: int) -> np.ndarray:
    """Move every row later (positive) or earlier (negative), filling with zeros."""
    if frames == 0:
        return rows
    moved = np.zeros_like(rows)
    if frames > 0:
        moved[:, frames:] = rows[:, :-frames]
    else:
        moved[:, :frames] = rows[:, -frames:]
    return moved


def _decompose(spectrogram: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """How much of each patch sounds at each frame, shape [patches + _FREE, frames].

    Multiplicative updates for the KL divergence; the given patches stay as
    they are, the free ones are learned along the way.
    """
    rng = np.random.default_rng(0)
    bands, frames = spectrogram.shape
    known = fixed.shape[1]
    free = rng.random((bands, _FREE, _PATCH)).astype(np.float32) * 0.1 + 1e-3
    patches = np.concatenate([fixed.astype(np.float32), free], axis=1)
    gains = rng.random((patches.shape[1], frames)).astype(np.float32) * 0.1 + 1e-3
    ones = np.ones_like(spectrogram)

    def ratio() -> np.ndarray:
        model = sum(patches[:, :, t] @ _shift(gains, t) for t in range(_PATCH))
        return spectrogram / (model + 1e-6)

    for _ in range(_ITERATIONS):
        fit = ratio()
        gains *= sum(patches[:, :, t].T @ _shift(fit, -t) for t in range(_PATCH)) / (
            # Near the end of the recording a patch has fewer frames left to explain.
            sum(patches[:, :, t].T @ _shift(ones, -t) for t in range(_PATCH)) + 1e-9
        )
        fit = ratio()
        for t in range(_PATCH):
            moved = _shift(gains, t)
            patches[:, known:, t] *= ((fit @ moved.T) / (ones @ moved.T + 1e-9))[:, known:]
        # Keep the free patches at unit size so their loudness stays in the gains.
        size = patches[:, known:, :].sum(axis=(0, 2)) + 1e-9
        patches[:, known:, :] /= size[None, :, None]
        gains[known:] *= size[:, None]
    return gains


def snare_response(drums_wav: Path, activations: np.ndarray) -> np.ndarray:
    """Per ADTOF frame: the snare's loudness relative to the song's typical
    snare hit (1.0), and its share of all the drums sounding at that frame.

    Shape [frames, 2]; empty when the song has too few clear snare hits to
    learn its sound from.
    """
    empty = np.zeros((0, 2), dtype=np.float32)
    peaks = picked_frames(activations)
    if len(peaks["snare"]) < _MIN_SNARES:
        return empty
    spectrogram = _spectrogram(drums_wav)
    frames = min(spectrogram.shape[1], len(activations))
    spectrogram = spectrogram[:, :frames]
    peaks = {inst: found[found < frames] for inst, found in peaks.items()}
    patches, names = _patches(spectrogram, peaks)
    if "snare" not in names:
        return empty
    gains = _decompose(spectrogram, patches)
    # An onset spreads over a frame or two, depending on where it falls in the window.
    smooth = np.stack([np.convolve(row, [0.25, 0.5, 0.25], mode="same") for row in gains[:len(names)]])
    snare = smooth[names.index("snare")]
    typical = float(np.median([snare[max(0, f - 2):f + 3].max() for f in peaks["snare"]]))
    return np.stack([snare / (typical + 1e-9), snare / (smooth.sum(axis=0) + 1e-9)], axis=1).astype(np.float32)
