"""Counting level, phase and variable bar lengths are distinct decisions."""
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from stickmap.metre import Signature, parse_signatures
from stickmap.pipeline import Options
from stickmap.server import OptionsPatch
from stickmap.pulse import build_pulse, mixed_bars, _follow
from stickmap.signature_quantize import quantize_signature
from stickmap.musicxml import to_musicxml
from stickmap.score import Hit, Score


def mixed_pattern(lengths=(7,) * 8 + (6,) * 8, period=.283, pickup=.1):
    size = sum(lengths)
    times = pickup + np.arange(size + 1) * period
    a = np.zeros((int((times[-1] + .02) * 100), 5))
    hits = []
    offset = 0
    for bar, n in enumerate(lengths):
        for i in range(n):
            t = times[offset+i]
            events = [(3, 'hihat', .25)]
            if i == 0:
                events += [(0, 'kick', 1), (4, 'cymbal', .8)]
            if i == n // 2:
                events += [(1, 'snare', .8)]
            if i == n - 2:
                events += [(0, 'kick', .4)]
            for col, inst, strength in events:
                a[round(t*100), col] += strength
                hits.append(Hit(t, inst, strength))
            # Dense 32nds: many temporal votes, but no new counting level.
            if bar % 4 == 3 and i >= n-2:
                for j in (1, 2, 3):
                    onset = t + j * period / 4
                    a[round(onset*100), 1 + j % 2] += .7
                    hits.append(Hit(onset, 'snare' if j % 2 == 0 else 'tom', .7))
        offset += n
    return gaussian_filter1d(a, .7, axis=0), times, hits


@pytest.mark.parametrize('value', ['7/8+6/8', '5/4+4/4', '7/8+6/8+5/8', '1/16+3/16+7/16+15/16'])
def test_mixed_validation(value):
    assert Options(time_signature=value).time_signature == value
    assert OptionsPatch(time_signature=value).time_signature == value
    assert len(parse_signatures(value)) == len(value.split('+'))
    with pytest.raises(ValueError):
        Signature.parse(value)


@pytest.mark.parametrize('value', ['7/8+6/4', '7/8+', '+7/8', '7/8++6/8', '7/8+33/8',
                                  '7/8+6/8+5/8+4/8+3/8', '7/8 +6/8', '7/8+06/8'])
def test_invalid_mixed_validation(value):
    for cls in (Options, OptionsPatch):
        with pytest.raises(ValueError):
            cls(time_signature=value)


def test_dense_chops_preserve_tracker_eighth_level():
    a, times, hits = mixed_pattern()
    attacks = [h.time for h in hits]
    pulse = build_pulse(a, attacks, times[::2], [], Signature(7, 8),
                        signatures=parse_signatures('7/8+6/8'))
    assert pulse.period == pytest.approx(.283, abs=.01)
    assert np.quantile(np.abs(np.diff(pulse.times)-.283), .99) < .01
    assert pulse.period == np.median(np.diff(pulse.times))


@pytest.mark.parametrize('lengths,denominator,period', [((7,)*8+(6,)*8, 8, .283),
                                                       ((5,)*8+(4,)*8, 4, .5),
                                                       ((7,)*8+(6,)*8+(5,)*8, 8, .283)])
def test_mixed_dp_lengths_and_downbeats(lengths, denominator, period):
    a, times, hits = mixed_pattern(lengths, period)
    signatures = tuple(Signature(n, denominator) for n in dict.fromkeys(lengths))
    actual, decoded = mixed_bars(times, a, signatures, [h.time for h in hits])
    assert decoded == list(lengths)
    assert np.array(actual)[np.r_[0, np.cumsum(decoded)]] == pytest.approx(times[np.r_[0, np.cumsum(lengths)]], abs=.02)


def test_mixed_score_shift_notes_xml_and_roundtrip():
    a, times, hits = mixed_pattern()
    args = dict(activations=a, attacks=[h.time for h in hits], time_signature='7/8+6/8')
    score = quantize_signature(hits, times, [], **args)
    shifted = quantize_signature(hits, times, [], beat_shift=.25, **args)
    assert [b.beats for b in score.bars] == [7]*8+[6]*8
    assert [b.beats for b in shifted.bars] == [b.beats for b in score.bars]
    assert [b.start-a.start for a,b in zip(score.bars,shifted.bars)] == pytest.approx([.283/4]*16, abs=.001)
    assert score.pulse['meters_used'] == ['7/8','6/8']
    assert score.tempo == pytest.approx(106, abs=1)
    assert Score.from_json(score.to_json()) == score
    assert all(0 <= note.tick < bar.ticks for bar in score.bars for note in bar.notes)
    xml = ET.fromstring(to_musicxml(score))
    assert [node.findtext('beats') for node in xml.findall('.//time')] == ['7','6']


def test_silent_intro_and_dense_subdivisions_do_not_shrink_follower():
    period = .283
    # The first 25 seconds contain no onset evidence. A dense fill still has
    # the same denominator pulse as the groove on either side of it.
    times = 25 + np.arange(1200) * period / 4
    weights = np.tile([1, .65, .8, .65], 300)
    grid = np.asarray(_follow(times, weights, period, times[-1] + period))
    gaps = np.diff(grid)
    assert np.median(gaps) == pytest.approx(period, abs=.002)
    assert np.quantile(np.abs(gaps-period), .99) < .01


def test_pickup_is_covered_without_changing_first_full_downbeat():
    a, times, hits = mixed_pattern(pickup=.7)
    a[13, 3] = .4
    hits.append(Hit(.13, 'hihat', .4))
    grid = np.r_[times[0] - np.arange(3, 0, -1)*.283, times]
    actual, lengths = mixed_bars(grid, a, parse_signatures('7/8+6/8'), [h.time for h in hits])
    starts = np.asarray(actual)[np.r_[0, np.cumsum(lengths)]]
    assert actual[0] <= .13
    assert np.min(abs(starts - .7)) < .02
    assert lengths[-16:] == [7]*8 + [6]*8


def test_jumpy_tracker_cannot_put_half_notes_in_an_eighth_grid():
    a, times, hits = mixed_pattern()
    indices = np.r_[0, np.cumsum(np.resize([1, 1, 2], 70))]
    tracker = times[indices[indices < len(times)]]
    pulse = build_pulse(a, [h.time for h in hits], tracker, [], Signature(7, 8),
                        signatures=parse_signatures('7/8+6/8'))
    assert pulse.period == pytest.approx(.283, abs=.002)
    assert np.max(abs(np.diff(pulse.times) - .283)) < .01


@pytest.mark.parametrize('numerator', [6, 7])
def test_sustained_32nds_do_not_define_the_denominator(numerator):
    a, times, hits = mixed_pattern((numerator,)*16)
    dense = np.arange(times[0], times[-1]-.02, .283/4)
    for i, time in enumerate(dense):
        a[round(time*100), 1 + i % 2] += .8
    pulse = build_pulse(a, dense, times[::2], [], Signature(numerator, 8))
    assert pulse.period == pytest.approx(.283, abs=.01)
    assert np.quantile(abs(np.diff(pulse.times) - .283), .95) < .01


def test_placed_changes_start_where_the_user_heard_them():
    from stickmap.signature_quantize import placed_bars

    grid = [i * 0.5 for i in range(30)]
    times, lengths = placed_bars(grid, 7, [(7.0, Signature(6, 8))])
    assert lengths[:3] == [7, 7, 6] and set(lengths[2:]) == {6}
    assert times[14] == 7.0 and len(times) == sum(lengths) + 1
    # Off a bar line, the remainder becomes one short bar instead of moving the change.
    assert placed_bars(grid, 7, [(6.0, Signature(6, 8))])[1][:3] == [7, 5, 6]
    assert set(placed_bars(grid, 7, [(0.0, Signature(6, 8))])[1]) == {6}


def test_meter_changes_are_validated():
    from stickmap.metre import parse_meter_changes

    assert parse_meter_changes([{"time": 216.0, "time_signature": "6/8"}], "7/8") == [(216.0, Signature(6, 8))]
    for bad in ([{"time": 1.0, "time_signature": "3/4"}], [{"time": -1, "time_signature": "6/8"}],
                [{"time": 1.0}], [{"time": True, "time_signature": "6/8"}]):
        with pytest.raises(ValueError):
            parse_meter_changes(bad, "7/8")
