"""Native Shorts geometry, small ordinary heads, source selection and gates."""
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from layouts import detect_shorts, ink, shorts_paper_top, region_cursor
from media import video_path
import pilot_new
from pilot_shorts import fast_counts, target_tile
from strip import label_heads

FIXTURES=Path(__file__).parent/'fixtures'
PROFILE=json.loads((HERE/'legends/olliewoods_shorts.json').read_text())


def test_native_page_boundary_excludes_performance_and_faded_end_cards():
    page=cv2.imread(str(FIXTURES/'shorts-weckl-page-native.png'))
    frame=np.full((1280,720,3),(150,30,210),np.uint8)
    frame[527:]=page
    class Capture:
        def __init__(self,*args): self.fraction=0
        def get(self,key): return 1000
        def set(self,key,value): self.fraction=value/1000
        def read(self): return True,frame.copy() if self.fraction<.95 else (frame*.25).astype(np.uint8)
        def release(self): pass
    assert shorts_paper_top(frame)==527
    with patch('layouts.cv2.VideoCapture',Capture): geometry=detect_shorts('local')
    assert (geometry['top'],geometry['bottom'],geometry['left'],geometry['right'])==(527,1280,0,720)
    assert geometry['absent_samples'] and geometry['prefer_staff_line_span']
    assert min(geometry['native_spacing'])>10
    box=region_cursor(page,10.6,geometry['cursor_style'])
    assert box and box[0]==pytest.approx(129.5) and 4<=box[3]<=8
    assert box[2]-box[1]>30
    with pytest.raises(ValueError,match='vertical'): shorts_paper_top(frame[:720])
    with pytest.raises(ValueError,match='white Shorts page'): shorts_paper_top(np.zeros_like(frame))


def test_native_small_ordinary_heads_survive_without_reading_sticking_or_bar_numbers():
    gray=ink(cv2.imread(str(FIXTURES/'shorts-kathe-row-native.png')))
    notes,checks=label_heads(gray,PROFILE)
    assert not checks['reasons'] and 9.5<checks['spacing']<10
    assert len(notes)>=12 and sum(n['small'] for n in notes)>=9
    assert {n['pos'] for n in notes}=={1,3,5,7}
    assert all(n['shape']=='filled' and not n['grace'] for n in notes)
    assert not any(n['ghost'] or n['accent'] or n['flam'] or n['roll'] for n in notes)
    legacy,_=label_heads(gray,{**PROFILE,'keep_small_filled':False})
    assert len(legacy)<len(notes)/2


def test_native_translucent_cursor_is_not_just_the_coloured_note_or_sticking():
    page=cv2.imread(str(FIXTURES/'shorts-kathe-cursor-native.png'))
    box=region_cursor(page,9.66,'orange_line')
    assert box and box[0]==pytest.approx(581) and box[3]<=4
    assert box[1]<70 and box[2]>110
    # A coloured note/sticking alone has no sustained six-space line.
    assert region_cursor(page[70:120],9.66,'orange_line') is None


def test_native_roni_still_rejects_below_eight_pixels():
    notes,checks=label_heads(ink(cv2.imread(str(FIXTURES/'shorts-roni-row-native.png'))),PROFILE)
    assert notes and 7.8<checks['spacing']<8
    assert 'native staff spacing below 8 px' in checks['reasons']


def test_shorts_profile_withholds_unaudited_instrument_names_and_techniques():
    assert PROFILE['enabled'] and PROFILE['strict_clef_pair']
    assert not PROFILE['verified_flags'] and not PROFILE.get('open_verified')
    assert not PROFILE.get('clef_omitted_verified') and not PROFILE.get('scrolling_verified')
    assert {k for k,r in PROFILE['rules'].items() if r['verified']}=={
        'filled:1','filled:3','filled:5','filled:6','filled:7','filled:8'}


def test_webm_is_a_native_source_and_multiple_sources_fail_closed(tmp_path,monkeypatch):
    folder=tmp_path/'ollieshorts'/'clip';folder.mkdir(parents=True)
    (folder/'video.webm').write_bytes(b'local')
    assert video_path(folder).name=='video.webm'
    monkeypatch.setattr(pilot_new,'PILOTS',tmp_path)
    assert pilot_new.folders(['olliewoods_shorts'])==[folder]
    (folder/'video.mp4').write_bytes(b'other')
    with pytest.raises(ValueError,match='found 2'): video_path(folder)
    assert pilot_new.folders(['ollieshorts'])==[folder]


def test_title_pipe_is_not_a_fabricated_duration(tmp_path):
    folder=tmp_path/'ollieshorts'/'clip';folder.mkdir(parents=True)
    (folder/'video.webm').write_bytes(b'local')
    title='Tobines Baianá | Drum Transcription'
    (folder/'title.txt').write_text(title)
    class Capture:
        def __init__(self,*args):pass
        def get(self,key):return 30 if key==5 else 1005
        def release(self):pass
    with patch('cv2.VideoCapture',Capture): meta=pilot_new.metadata(folder)
    assert meta['title']==title and meta['seconds']==33.5
    assert meta['channel']=='olliewoods_shorts'


def test_fast_counts_do_not_bridge_rejected_systems_or_count_unisons_as_doubles():
    isolated=[dict(system=0,time=1.,inst='kick'),dict(system=1,time=1.05,inst='snare')]
    assert not any(fast_counts(isolated)['core_neighbour'].values())
    chord=[dict(system=0,time=1.,inst='kick'),dict(system=0,time=1.,inst='tom'),
           dict(system=0,time=1.,inst='tom'),dict(system=0,time=1.05,inst='snare')]
    result=fast_counts(chord)
    assert result['core_neighbour']==dict(kick=1,snare=1,tom=1)
    assert not any(result['same_class_neighbour'].values())


def test_audit_crop_keeps_high_ledger_target_and_native_fixtures_have_provenance():
    frame=np.full((150,200,3),255,np.uint8)
    # Target at position 12 is two full spaces above the top staff line.
    cv2.circle(frame,(100,30),3,(0,0,0),-1)
    image=target_tile(frame,100,30,[50,60,70,80,90],10)
    assert (image.min(2)<50).sum()>10
    for row in json.loads((FIXTURES/'shorts-provenance.json').read_text()):
        assert not row['resized'] and row['native_size']==[720,1280]
        assert hashlib.sha256((FIXTURES/row['fixture']).read_bytes()).hexdigest()==row['sha256']
