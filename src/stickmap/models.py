"""Stages 2-4: the three neural models (Demucs, beat_this, ADTOF).

Audio is always read with soundfile and handed over as arrays: the file
loaders bundled with demucs/beat_this go through torchcodec, which cannot
find Homebrew's FFmpeg libraries.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .score import Hit

# ADTOF output columns, in the order of adtof_pytorch.LABELS_5 = [35, 38, 47, 42, 49].
ADTOF_CLASSES = ("kick", "snare", "tom", "hihat", "cymbal")
ADTOF_FPS = 100
# Peak thresholds the ADTOF authors tuned, in ADTOF_CLASSES order.
DEFAULT_THRESHOLDS = (0.22, 0.24, 0.32, 0.22, 0.30)


def pick_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def separate_drums(audio_wav: Path, drums_wav: Path, *, device: str, model: str = "htdemucs") -> None:
    from demucs.api import Separator

    audio, sr = sf.read(audio_wav, dtype="float32", always_2d=True)
    separator = Separator(model=model, device=device, progress=False)
    _, stems = separator.separate_tensor(torch.from_numpy(audio.T.copy()), sr)
    sf.write(drums_wav, stems["drums"].cpu().numpy().T, separator.samplerate)


def track_beats(audio_wav: Path, *, device: str) -> tuple[list[float], list[float]]:
    """Beat and downbeat times in seconds, tracked on the full mix."""
    from beat_this.inference import Audio2Beats

    audio, sr = sf.read(audio_wav, dtype="float32", always_2d=True)
    beats, downbeats = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)(audio, sr)
    return [float(b) for b in beats], [float(d) for d in downbeats]


# ADTOF fine-tuned on drum transcription videos (tools/finetune): it finds more toms and kicks in
# fast fills. It learned from drum stems only, so the full mix keeps the original weights.
STEM_WEIGHTS = Path(__file__).with_name("adtof-stickmap-v2.pt")


def drum_activations(wav: Path, *, device: str, weights: Path | None = None) -> np.ndarray:
    """ADTOF frame activations, shape [frames, 5] at ADTOF_FPS, columns as ADTOF_CLASSES.

    `weights` replaces the original model's weights (a state dict of the same network).
    """
    import adtof_pytorch as adtof

    model = adtof.create_frame_rnn_model(adtof.calculate_n_bins())
    model = adtof.load_pytorch_weights(model, adtof.get_default_weights_path(), strict=False)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval().to(device)
    features = adtof.load_audio_for_model(str(wav)).to(device)
    with torch.no_grad():
        try:
            return model(features).cpu().numpy()[0]
        except RuntimeError as error:
            # cuDNN's RNN kernels refuse very long inputs (e.g. a 12-minute song on an RTX 5080);
            # PyTorch's own RNN kernels on the same GPU handle them.
            if "CUDNN_STATUS_NOT_SUPPORTED" not in str(error):
                raise
            with torch.backends.cudnn.flags(enabled=False):
                return model(features).cpu().numpy()[0]


def stem_activations(drums_wav: Path, *, device: str) -> np.ndarray:
    """ADTOF frame activations of the drum stem, from the fine-tuned model."""
    return drum_activations(drums_wav, device=device, weights=STEM_WEIGHTS)


# ADTOF's picker measures each peak against the average of the 100 ms before it,
# which the previous stroke of a fast double or a roll is still part of. A peak
# this soon after a hit on the same drum counts on its own when it clears the
# threshold and the activation dipped by half the threshold in between.
_FOLLOW_FRAMES = 12


def _with_fast_strokes(activation: np.ndarray, picked: list[int], threshold: float) -> list[int]:
    frames = sorted(picked)
    taken = set(frames)
    kept, previous = [], None
    for f in range(len(activation)):
        if f in taken:
            kept.append(f)
            previous = f
        elif (
            previous is not None and 2 < f - previous <= _FOLLOW_FRAMES
            and activation[f] >= threshold and activation[f] > activation[f - 1]
            and activation[f] == activation[f - 2:f + 3].max()
            and not any(g in taken for g in (f + 1, f + 2))
            and activation[f] - activation[previous:f].min() >= threshold / 2
        ):
            kept.append(f)
            previous = f
    return kept


def picked_frames(activations: np.ndarray, thresholds: list[float] | None = None) -> dict[str, np.ndarray]:
    """Peak-picked frame numbers per instrument, in ADTOF_CLASSES order."""
    import adtof_pytorch as adtof

    thresholds = list(thresholds or DEFAULT_THRESHOLDS)
    picker = adtof.PeakPicker(thresholds=thresholds, fps=ADTOF_FPS)
    peaks = picker.pick(activations[None], labels=adtof.LABELS_5)[0]
    last = len(activations) - 1
    frames = {}
    for column, (label, inst) in enumerate(zip(adtof.LABELS_5, ADTOF_CLASSES)):
        picked = [min(int(round(t * ADTOF_FPS)), last) for t in peaks[label]]
        frames[inst] = np.array(_with_fast_strokes(activations[:, column], picked, thresholds[column]), dtype=int)
    return frames


# With the stem and the mix averaged, a tom needs a little more than ADTOF's own threshold.
TOM_BLEND = 0.36 / 0.32
# A snare stroke under this share of the song's typical snare loudness is a ghost note.
GHOST_RATIO = 0.5
# A snare ADTOF did not pick is added when it is at least this loud (against the typical
# snare), takes this share of what the drums play at that moment, has this much
# activation, and no picked hit lies within these frames before or after it.
_RESCUE_LOUDNESS, _RESCUE_SHARE, _RESCUE_ACTIVATION, _RESCUE_CLEAR = 0.05, 0.4, 0.01, (10, 3)


def _toms(activations: np.ndarray, mix: np.ndarray, snares: np.ndarray, thresholds: list[float]) -> list[Hit]:
    """Toms from the stem and the mix together, without those that are a snare's echo in the tom output."""
    frames = min(len(activations), len(mix))
    blend = (activations[:frames] + mix[:frames]) / 2
    snare, tom = ADTOF_CLASSES.index("snare"), ADTOF_CLASSES.index("tom")
    raised = [t * TOM_BLEND if i == tom else t for i, t in enumerate(thresholds)]
    hits = []
    for f in picked_frames(blend, raised)["tom"]:
        on_snare = len(snares) and np.abs(snares - f).min() <= 3
        if not (on_snare and blend[max(0, f - 3):f + 4, snare].max() > blend[f, tom]):
            hits.append(Hit(float(f) / ADTOF_FPS, "tom", float(blend[f, tom])))
    return hits


def _snares(activations: np.ndarray, response: np.ndarray, peaks: dict[str, np.ndarray]) -> list[Hit]:
    """Snares with their loudness as strength, plus the quiet ones ADTOF did not pick."""
    frames = min(len(response), len(activations))
    loudness, share = response[:frames, 0], response[:frames, 1]
    column = ADTOF_CLASSES.index("snare")

    def hit(f: int, strength: float) -> Hit:
        return Hit(float(f) / ADTOF_FPS, "snare", strength, ghost=strength < GHOST_RATIO)

    hits = [hit(f, float(loudness[max(0, f - 2):f + 3].max()) if f < frames else 1.0) for f in peaks["snare"]]
    picked = np.unique(np.concatenate(list(peaks.values())))
    after, before = _RESCUE_CLEAR
    added = -before - 1
    for f in range(2, frames - 2):
        # A peak of the loudness; where it stays level, the first frame stands for the whole stretch.
        if loudness[f] < _RESCUE_LOUDNESS or loudness[f] <= loudness[f - 1] or loudness[f] < loudness[f - 2:f + 3].max():
            continue
        if share[f] < _RESCUE_SHARE or activations[f - 2:f + 3, column].max() < _RESCUE_ACTIVATION:
            continue
        distance = f - picked
        if f - added > before and not np.any((distance >= -before) & (distance <= after)):
            hits.append(hit(f, float(loudness[f])))
            added = f
    return hits


def pick_hits(
    activations: np.ndarray, thresholds: list[float] | None = None, *,
    mix: np.ndarray | None = None, snare: np.ndarray | None = None,
) -> list[Hit]:
    """Peak-pick activations into hits; `strength` is the activation at the peak.

    `mix` (ADTOF activations of the full mix) steadies the toms. `snare` (see
    dynamics.snare_response) makes a snare's strength its loudness, marks the
    ghost notes and adds quiet snares the activation is too weak to pick.
    """
    thresholds = list(thresholds or DEFAULT_THRESHOLDS)
    peaks = picked_frames(activations, thresholds)
    refined = {"tom": mix is not None, "snare": snare is not None and len(snare) > 0}
    hits = [
        Hit(float(f) / ADTOF_FPS, inst, float(activations[f, column]))
        for column, inst in enumerate(ADTOF_CLASSES) if not refined.get(inst) for f in peaks[inst]
    ]
    if refined["tom"]:
        hits += _toms(activations, mix, peaks["snare"], thresholds)
    if refined["snare"]:
        hits += _snares(activations, snare, peaks)
    return sorted(hits, key=lambda h: h.time)
