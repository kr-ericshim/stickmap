"""Repeated page-boundary observations must not inflate positive counts."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from align import unique_events


def test_identical_boundary_positive_is_counted_once_with_provenance():
    a=dict(time=105.111111,inst='kick',pos=1,system=189,ghost=False)
    b={**a,'system':194}
    notes=unique_events([a,b])
    assert len(notes)==1
    assert notes[0]['observed_systems']==[189,194]
    assert 'observed_systems' not in a  # input diagnostics remain inspectable


def test_strict_observation_wins_without_copying_scoring_technique_flags():
    a=dict(time=1.,inst='snare',pos=5,system=1,tier='scoring',ghost=True)
    b={**a,'system':2,'tier':'strict','ghost':False}
    note=unique_events([a,b])[0]
    assert note['tier']=='strict' and note['system']==2 and not note['ghost']
    assert note['observed_systems']==[1,2]


def test_nearby_hits_and_distinct_instruments_or_positions_are_preserved():
    a=dict(time=1.,inst='tom',pos=3,system=1)
    notes=[a,{**a,'time':1.000001},{**a,'pos':7},{**a,'inst':'snare'}]
    assert unique_events(notes)==sorted(notes,key=lambda n:n['time'])
