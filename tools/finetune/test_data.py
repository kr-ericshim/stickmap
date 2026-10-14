"""Partial-legend uncertainty uses the accepted export's exact moment mapping."""
from data import withheld_heads
import pytest


def test_withheld_heads_use_fitted_times_and_classes_without_becoming_truth():
    strip = [dict(system=3, notes=[dict(inst='tom_pos_3', exportable=False),
                                   dict(inst='kick', exportable=True),
                                   dict(inst='unknown', exportable=False),
                                   dict(inst='crash', exportable=False)])]
    report = dict(systems=[dict(system=3, usable=True, times=[1.23456, 2.0],
                               moment_groups=[[0, 1], [2, 3]])])
    heads = withheld_heads(strip, report)
    assert [(q['time'], q['class']) for q in heads] == [
        (1.2346, 'tom'), (2., 'cymbal'), (2., 'kick'), (2., 'snare'), (2., 'tom')]
    assert all(q['system'] == 3 and q['head'] != 1 for q in heads)
    report['systems'][0]['usable'] = False
    assert withheld_heads(strip, report) == []


def test_withheld_time_group_mismatch_fails_instead_of_silently_missing_masks():
    strip = [dict(system=1, notes=[dict(inst='tom', exportable=False)])]
    report = dict(systems=[dict(system=1, usable=True, times=[1.], moment_groups=[])])
    with pytest.raises(ValueError, match='mismatch'):
        withheld_heads(strip, report)
