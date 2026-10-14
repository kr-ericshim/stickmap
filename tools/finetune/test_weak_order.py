"""CPU-only synthetic tests. No training data, model downloads, or vault access."""

from dataclasses import replace
from itertools import groupby, product
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from weak_order import WeakOrderWindow, encode_moments, subset_log_probs, weak_order_loss


@pytest.fixture(autouse=True)
def cpu_threads_and_rng():
    threads = torch.get_num_threads()
    with torch.random.fork_rng(devices=[]):
        torch.set_num_threads(1)
        torch.manual_seed(27)
        yield
    torch.set_num_threads(threads)


def collapse(path):
    return tuple(token for token, _ in groupby(path) if token)


def impulses(tokens, positions, length=50, magnitude=10.0):
    logits = torch.full((1, length, 3), -magnitude, dtype=torch.float64)
    for token, frame in zip(tokens, positions):
        for k in range(3):
            if token & (1 << k):
                logits[0, frame, k] = magnitude
    return logits


def test_grouping_toms_unisons_and_repeats():
    assert encode_moments([['kick', 'snare'], ['tom_high', 'tom_low'],
                           ['tom_mid'], ['K', 'S', 'T']]) == (3, 4, 4, 7)
    for malformed in ([[]], [['unknown']], [['hihat']], ['kick']):
        with pytest.raises(ValueError):
            encode_moments(malformed)


def test_joint_distribution_and_logits_probability_equivalence():
    logits = torch.randn(2, 17, 5, dtype=torch.float64)
    actual = subset_log_probs(logits, from_logits=True)
    torch.testing.assert_close(actual.exp().sum(-1), torch.ones(2, 17, dtype=torch.float64))
    torch.testing.assert_close(actual, subset_log_probs(logits.sigmoid()))
    torch.testing.assert_close(actual[..., 0], torch.nn.functional.logsigmoid(-logits[..., :3]).sum(-1))


def test_adtof_sigmoid_path_matches_logits_loss_and_gradients():
    logits = torch.randn(2, 12, 5, dtype=torch.float64, requires_grad=True)
    windows = [WeakOrderWindow(0, 1, 11, (3, 4, 4)), WeakOrderWindow(1, 0, 7, (2, 1))]
    direct = weak_order_loss(logits, windows, from_logits=True)
    sigmoid = weak_order_loss(logits.sigmoid(), windows)
    torch.testing.assert_close(sigmoid, direct)
    direct_grad, = torch.autograd.grad(direct, logits)
    sigmoid_grad, = torch.autograd.grad(sigmoid, logits)
    torch.testing.assert_close(sigmoid_grad, direct_grad)
    assert torch.isfinite(sigmoid_grad).all()


@pytest.mark.parametrize('target', [(), (1,), (1, 2), (3, 4), (2, 2)])
def test_ctc_matches_exhaustive_path_probability(target):
    logits = torch.randn(1, 3, 3, dtype=torch.float64)
    probabilities = subset_log_probs(logits, from_logits=True).exp()[0]
    probability = sum(torch.stack([probabilities[t, token] for t, token in enumerate(path)]).prod()
                      for path in product(range(8), repeat=3) if collapse(path) == target)
    expected = -probability.log() / max(1, len(target))
    actual = weak_order_loss(logits, [WeakOrderWindow(0, 0, 3, target)], from_logits=True)
    torch.testing.assert_close(actual, expected)


def test_correct_order_preferred_to_swapped_same_counts():
    tokens = (1, 2, 4, 3, 4)
    logits = impulses(tokens, [8, 14, 20, 26, 32])
    correct = WeakOrderWindow(0, 0, 50, tokens)
    swapped = replace(correct, tokens=(4, 2, 1, 3, 4))
    good = weak_order_loss(logits, [correct], from_logits=True)
    bad = weak_order_loss(logits, [swapped], from_logits=True)
    assert good < .01
    assert bad > good + 2


def test_moderate_shift_and_local_warp_within_window():
    # At 100 Hz, shifts up to +/-60 ms and uneven IOIs, without target time hints.
    tokens = (1, 2, 4, 3)
    window = WeakOrderWindow(0, 0, 50, tokens)
    values = [weak_order_loss(impulses(tokens, positions), [window], from_logits=True)
              for positions in ([12, 18, 24, 30], [6, 12, 18, 24],
                                [18, 24, 30, 36], [9, 17, 27, 34])]
    for value in values[1:]:
        torch.testing.assert_close(value, values[0], rtol=0, atol=1e-5)


def test_unison_requires_both_instruments_in_one_token():
    window = WeakOrderWindow(0, 0, 50, (3, 4))
    unison = weak_order_loss(impulses((3, 4), (10, 25)), [window], from_logits=True)
    serial = weak_order_loss(impulses((1, 2, 4), (10, 16, 25)), [window], from_logits=True)
    missing = weak_order_loss(impulses((1, 4), (10, 25)), [window], from_logits=True)
    assert serial > unison + 2
    assert missing > unison + 2


def test_repeated_toms_require_two_runs_separated_by_blank():
    window = WeakOrderWindow(0, 0, 50, (4, 4))
    two = weak_order_loss(impulses((4, 4), (10, 20)), [window], from_logits=True)
    one = weak_order_loss(impulses((4,), (10,)), [window], from_logits=True)
    assert one > two + 2
    with pytest.raises(ValueError, match='repeat blanks'):
        weak_order_loss(torch.zeros(1, 2, 3), [replace(window, stop=2)], from_logits=True)


def test_ragged_batch_weights_padding_and_gradient_isolation():
    logits = torch.randn(3, 24, 5, dtype=torch.float64, requires_grad=True)
    windows = [WeakOrderWindow(0, 3, 18, (1, 3, 4), confidence=.4, weight=.5),
               WeakOrderWindow(1, 5, 12, (2, 2)),
               WeakOrderWindow(0, 19, 24, (4,), weight=0)]
    losses = weak_order_loss(logits, windows, from_logits=True, reduction='none')
    separate = torch.stack([weak_order_loss(logits, [w], from_logits=True) for w in windows])
    torch.testing.assert_close(losses, separate)
    unweighted = weak_order_loss(logits, [replace(windows[0], confidence=1, weight=1)], from_logits=True)
    torch.testing.assert_close(losses[0], .2 * unweighted)
    torch.testing.assert_close(weak_order_loss(logits, windows, from_logits=True), losses.mean())
    losses.sum().backward()
    allowed = torch.zeros_like(logits, dtype=torch.bool)
    allowed[0, 3:18, :3] = True
    allowed[1, 5:12, :3] = True
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[allowed].abs().sum() > 0
    assert torch.count_nonzero(logits.grad[~allowed]) == 0


def test_empty_windows_empty_target_and_zero_confidence():
    probabilities = torch.full((1, 8, 5), .1, requires_grad=True)
    empty = weak_order_loss(probabilities, [])
    assert empty == 0
    assert weak_order_loss(probabilities, [], reduction='none').shape == (0,)
    zero = weak_order_loss(probabilities, [WeakOrderWindow(0, 0, 8, (1,), confidence=0)])
    (empty + zero).backward()
    assert torch.count_nonzero(probabilities.grad) == 0
    blank = weak_order_loss(probabilities, [WeakOrderWindow(0, 0, 8, ())])
    torch.testing.assert_close(blank, -24 * torch.log(torch.tensor(.9)))


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_extreme_logits_finite_loss_and_gradients(dtype):
    logits = torch.full((2, 96, 5), -80., dtype=dtype, requires_grad=True)
    windows = [WeakOrderWindow(0, 0, 96, (1, 2, 4, 3) * 5),
               WeakOrderWindow(1, 3, 80, (4, 4, 4, 7))]
    loss = weak_order_loss(logits, windows, from_logits=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[..., :3].abs().sum() > 0
    assert torch.count_nonzero(logits.grad[..., 3:]) == 0


def test_saturated_probabilities_and_numerical_gradient():
    probabilities = torch.tensor([[[0., 1., 0.], [1., 0., 1.], [0., 0., 0.]]], requires_grad=True)
    loss = weak_order_loss(probabilities, [WeakOrderWindow(0, 0, 3, (1, 4))])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(probabilities.grad).all()
    logits = torch.randn(1, 4, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: weak_order_loss(x, [WeakOrderWindow(0, 0, 4, (3, 4))], from_logits=True),
        (logits,), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('changes', [dict(start=-1), dict(stop=99), dict(start=8, stop=8),
                                     dict(sample=1), dict(start=.5), dict(tokens=(0,)),
                                     dict(tokens=(8,)), dict(tokens=(1.5,)),
                                     dict(confidence=float('nan')), dict(confidence=1.1),
                                     dict(weight=-1), dict(weight=float('inf'))])
def test_invalid_windows_fail_closed(changes):
    with pytest.raises(ValueError):
        weak_order_loss(torch.full((1, 10, 3), .3),
                        [replace(WeakOrderWindow(0, 0, 10, (1, 2)), **changes)])


def test_invalid_outputs_fail_closed():
    window = [WeakOrderWindow(0, 0, 10, (1,))]
    for output in (torch.zeros(1, 10, 4), torch.ones(1, 10, 3, dtype=torch.long),
                   torch.full((1, 10, 3), float('nan')), torch.full((1, 10, 3), 1.1)):
        with pytest.raises(ValueError):
            weak_order_loss(output, window)


def synthetic_batch(seed, batch_size=24):
    """Unknown hit locations; three acoustic signatures plus background noise.

    Latent frame tokens are returned ONLY for evaluation, never to the loss.
    Random orders prevent a time/position-only classifier from solving the task.
    """
    rng = torch.Generator().manual_seed(seed)
    x = .03 * torch.randn(batch_size, 36, 3, generator=rng)
    latent = torch.zeros(batch_size, 36, dtype=torch.long)
    signatures = torch.tensor([[1., .15, -.3], [-.25, 1., .2], [.1, -.35, 1.]])
    windows = []
    for b in range(batch_size):
        tokens = torch.randint(1, 8, (5,), generator=rng).tolist()
        positions = torch.arange(5) * 6 + torch.randint(1, 5, (5,), generator=rng)
        for token, frame in zip(tokens, positions):
            latent[b, frame] = token
            for k in range(3):
                if token & (1 << k):
                    x[b, frame] += signatures[k]
        windows.append(WeakOrderWindow(b, 0, 36, tuple(tokens)))
    return x, windows, latent


def test_tiny_cpu_model_learns_identity_from_order_only():
    model = torch.nn.Linear(3, 3)  # No position input, no access to latent timing.
    # An instrument-agnostic energy prior, identical for all three outputs.
    # This models fine-tuning an onset-capable network, not learning onsets
    # from scratch. No class-specific initialization or timing supervision.
    torch.nn.init.constant_(model.weight, 2.)
    torch.nn.init.constant_(model.bias, -2.)
    optimizer = torch.optim.Adam(model.parameters(), lr=.10)
    x, windows, _ = synthetic_batch(123)
    initial = weak_order_loss(model(x), windows, from_logits=True).item()
    with torch.no_grad():
        test_x, _, test_latent = synthetic_batch(9999, batch_size=48)
        before = subset_log_probs(model(test_x), from_logits=True).argmax(-1)
        initial_identity = (before[test_latent > 0] == test_latent[test_latent > 0]).float().mean().item()
    assert initial_identity < .3
    for step in range(140):
        x, windows, _ = synthetic_batch(1000 + step)
        loss = weak_order_loss(model(x), windows, from_logits=True)
        optimizer.zero_grad()
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    x, windows, latent = synthetic_batch(9999, batch_size=48)
    with torch.no_grad():
        logits = model(x)
        final = weak_order_loss(logits, windows, from_logits=True).item()
        predicted = subset_log_probs(logits, from_logits=True).argmax(-1)
        identity_accuracy = (predicted[latent > 0] == latent[latent > 0]).float().mean().item()
        sequence_accuracy = sum(collapse(p.tolist()) == w.tokens
                                for p, w in zip(predicted, windows)) / len(windows)
        blank_accuracy = (predicted[latent == 0] == 0).float().mean().item()
    print(f'CPU toy: loss {initial:.4f} -> {final:.4f}; held-out hit identity '
          f'{identity_accuracy:.1%}, exact sequences {sequence_accuracy:.1%}, blank {blank_accuracy:.1%}')
    assert final < initial * .3
    assert identity_accuracy >= .95
    assert sequence_accuracy >= .90
    assert blank_accuracy >= .99
