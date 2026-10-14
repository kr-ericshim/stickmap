"""Order-only K/S/T supervision; no note timestamps or frame targets.

Seven nonempty instrument subsets are CTC tokens, with the empty subset as
blank. Independent sigmoid outputs define a normalized distribution over all
eight subsets. See WEAK_ORDER.md for the complete-notation assumption: this
is sequence supervision, NOT a positive-only loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Literal, Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


# Bit positions match train.CLASSES[:3], not a new learned output head.
TOKEN_NAMES = ("blank", "K", "S", "K+S", "T", "K+T", "S+T", "K+S+T")
_FAMILY_BITS = {"K": 1, "kick": 1, "S": 2, "snare": 2,
                "T": 4, "tom": 4, "tom_high": 4, "tom_mid": 4,
                "tom_low": 4, "tom_floor": 4}


def encode_moments(moments: Iterable[Iterable[str]]) -> tuple[int, ...]:
    """Encode already verified simultaneous groups, preserving moment order.

    Merge tom positions WITHIN a moment only; repeated tom moments survive.
    Callers remove ties and hat/cymbal-only moments first. Unknown names and
    empty groups fail closed instead of silently deleting notation. No timing
    proximity heuristic is used to infer simultaneity.
    """
    tokens = []
    for names in moments:
        if isinstance(names, str):
            raise ValueError("each moment must be a group of instrument names")
        token = 0
        for name in names:
            if name not in _FAMILY_BITS:
                raise ValueError(f"unrecognized K/S/T instrument: {name!r}")
            token |= _FAMILY_BITS[name]
        if not token:
            raise ValueError("empty moment: remove non-K/S/T moments explicitly")
        tokens.append(token)
    return tuple(tokens)


@dataclass(frozen=True)
class WeakOrderWindow:
    """One complete sequence inside a crop-relative [start, stop) window.

    `sample` indexes [batch, time, channels]. `tokens` contain IDs 1..7, never
    blank. Empty tokens explicitly mean verified no K/S/T events; missing
    annotation must instead be represented by no window. `confidence` is in
    [0, 1]; `weight` is nonnegative. Both scale loss without renormalization.
    """

    sample: int
    start: int
    stop: int
    tokens: tuple[int, ...]
    confidence: float = 1.0
    weight: float = 1.0


def subset_log_probs(outputs: torch.Tensor, *, from_logits: bool = False) -> torch.Tensor:
    """[..., 3 or 5] -> [..., 8] log probabilities, in TOKEN_NAMES order.

    The default consumes Frame-RNN sigmoid probabilities. Logits use stable
    logsigmoid; probabilities are clamped at 1e-6 for saturated model outputs.
    Half/bfloat16 accumulate in float32; float64 is preserved for gradcheck.
    Hi-hat/cymbal columns are ignored and receive no weak-loss gradients.
    """
    if outputs.ndim < 1 or outputs.shape[-1] not in (3, 5):
        raise ValueError("expected K/S/T or K/S/T/HH/C channels in that order")
    if not outputs.is_floating_point():
        raise ValueError("outputs must be floating point")
    x = outputs[..., :3]
    if not bool(torch.isfinite(x).all()):
        raise ValueError("K/S/T outputs must be finite")
    if x.dtype not in (torch.float32, torch.float64):
        x = x.float()
    if from_logits:
        yes, no = F.logsigmoid(x), F.logsigmoid(-x)
    else:
        if bool(((x < 0) | (x > 1)).any()):
            raise ValueError("probabilities must be in [0, 1]; use from_logits=True for logits")
        x = x.clamp(1e-6, 1 - 1e-6)
        yes, no = x.log(), torch.log1p(-x)
    ids = torch.arange(8, device=x.device)
    bits = (ids[:, None] & ids.new_tensor([1, 2, 4])) != 0
    # No softmax over K/S/T: simultaneous instruments are independent Bernoullis.
    return torch.where(bits, yes.unsqueeze(-2), no.unsqueeze(-2)).sum(-1)


def weak_order_loss(
    outputs: torch.Tensor,
    windows: Sequence[WeakOrderWindow],
    *,
    from_logits: bool = False,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:
    """Batched CTC over complete, variable-length coarse windows.

    Per-window value = confidence * weight * CTC_NLL / max(1, #moments).
    `mean` divides by NUMBER OF WINDOWS, not sum of weights: lowering every
    confidence really lowers the auxiliary loss. `none` preserves input order.
    Empty batches return differentiable zero (empty vector for `none`).

    Repeated equal tokens need an intervening blank: T >= L + repeats.
    Invalid bounds, impossible alignments and nonfinite losses raise instead
    of being hidden by zero_infinity. The caller must reject partial crops,
    strict overlap, vault/held-out sources and unverified notation BEFORE
    constructing windows; this numerical module does no file/data access.
    """
    if outputs.ndim != 3 or outputs.shape[-1] not in (3, 5):
        raise ValueError("outputs must have shape [batch, frames, 3 or 5]")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError(f"invalid reduction: {reduction}")
    # Connect an empty batch to autograd without summing unused model values.
    if not outputs.is_floating_point():
        raise ValueError("outputs must be floating point")
    zero = outputs[..., :3].reshape(-1)[:0].sum()
    if not windows:
        return zero.expand(0) if reduction == "none" else zero

    chunks, targets, lengths, target_lengths, scales = [], [], [], [], []
    for index, window in enumerate(windows):
        for name in ("sample", "start", "stop"):
            value = getattr(window, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"window {index}: {name} must be an integer")
        if not (0 <= window.sample < outputs.shape[0] and
                0 <= window.start < window.stop <= outputs.shape[1]):
            raise ValueError(f"window {index}: invalid bounds (never clip partial windows)")
        tokens = tuple(window.tokens)
        if any(isinstance(t, bool) or not isinstance(t, Integral) or not 1 <= t <= 7
               for t in tokens):
            raise ValueError(f"window {index}: tokens must be integers 1..7, without blank")
        repeats = sum(a == b for a, b in zip(tokens, tokens[1:]))
        length = window.stop - window.start
        if length < len(tokens) + repeats:
            raise ValueError(f"window {index}: impossible CTC alignment; need "
                             f"{len(tokens) + repeats} frames including repeat blanks")
        if not math.isfinite(window.confidence) or not 0 <= window.confidence <= 1:
            raise ValueError(f"window {index}: confidence must be finite in [0, 1]")
        if not math.isfinite(window.weight) or window.weight < 0:
            raise ValueError(f"window {index}: weight must be finite and nonnegative")
        chunks.append(subset_log_probs(outputs[window.sample, window.start:window.stop],
                                       from_logits=from_logits))
        targets.extend(tokens)
        lengths.append(length)
        target_lengths.append(len(tokens))
        scales.append(window.confidence * window.weight / max(1, len(tokens)))

    # Uniform padding is normalized, ignored by input_lengths, and has no model gradient.
    log_probs = pad_sequence(chunks, padding_value=-math.log(8)).contiguous()
    with torch.autocast(device_type=outputs.device.type, enabled=False):
        nll = F.ctc_loss(log_probs, torch.tensor(targets, dtype=torch.long),
                         torch.tensor(lengths, dtype=torch.long),
                         torch.tensor(target_lengths, dtype=torch.long),
                         blank=0, reduction="none", zero_infinity=False)
    if not bool(torch.isfinite(nll).all()):
        raise FloatingPointError("nonfinite weak-order CTC loss; do not silently skip")
    losses = nll * nll.new_tensor(scales)
    if not bool(torch.isfinite(losses).all()):
        raise FloatingPointError("nonfinite weighted weak-order loss")
    if reduction == "none":
        return losses
    return losses.sum() if reduction == "sum" else losses.mean()
