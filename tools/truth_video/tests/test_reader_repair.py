"""Native regressions for grey scrolling staffs, dense beams and partial visits."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from heads import staffs,staff_checks
from layouts import detect
from harvest_scroll import write
from harvest_flags import ghost_score
from strip import label_heads,read

FIXTURES=Path(__file__).parent/'fixtures'


@pytest.mark.parametrize('channel,spacing', [('golly',13.105),('olliewoods',11.703)])
def test_dense_beam_is_not_the_top_staff_line(channel,spacing):
    gray=cv2.imread(str(FIXTURES/f'{channel}-dense-beams.png'),0)
    legend=json.loads((HERE/'legends'/f'{channel}.json').read_text())
    legacy={**legend,'prefer_staff_line_span':False}
    _,bad=label_heads(gray,legacy)
    assert 'uneven staff lines' in bad['reasons']
    notes,good=label_heads(gray,legend)
    assert good['spacing']==pytest.approx(spacing,abs=.01)
    assert good['clef_confirmed'] and not good['reasons']
    assert any(n['exportable'] for n in notes)


def test_span_ranking_keeps_native_eight_pixel_gate():
    gray=np.full((130,600),255,np.uint8)
    for y in (55,62,69,76,83):gray[y,:]=170
    for y in (20,27,34):gray[y:y+3,80:500]=0
    found=staffs(gray,prefer_line_span=True)
    assert len(found)==1 and found[0][1]==7
    assert 'native staff spacing below 8 px' in staff_checks(gray,*found[0],[])['reasons']


def test_dailychops_limits_staff_search_to_light_top_notation():
    frame=np.full((720,1280,3),255,np.uint8)
    for y in (80,89,98,107,116):frame[y,400:]=150
    for y in (400,420,440,460,480):frame[y,:]=110
    class Capture:
        def __init__(self,*a):pass
        def get(self,key):return 120
        def set(self,*a):pass
        def read(self):return True,frame.copy()
        def release(self):pass
    with patch('layouts.cv2.VideoCapture',Capture):
        geometry=detect('fixture','dailychops')
    assert geometry['bottom']<200
    assert all(8.9<x<9.1 for x in geometry['native_spacing'])
    assert 400<=geometry['motion_left']<450


def test_native_scroll_preserves_grey_lines_and_rejects_discontinuity(tmp_path):
    data=np.load(FIXTURES/'dailychops-scroll-native.npz')
    frames=data['frames'];left=int(data['motion_left'])
    class Capture:
        def __init__(self,*a):self.i=0
        def get(self,key):return 5 if key==5 else len(frames)
        def read(self):
            if self.i==len(frames):return False,None
            frame=frames[self.i];self.i+=1;return True,frame
        def release(self):pass
    geometry=dict(layout='dailychops',top=0,bottom=frames.shape[1],left=0,
                  right=frames.shape[2],native_spacing=[8.9],motion_left=left)
    with patch('harvest_scroll.cv2.VideoCapture',Capture):
        scan=write('fixture',tmp_path/'good',geometry)
    row=scan['records'][0]
    assert not row['scroll_reasons']
    assert 8.8<row['native_space']<9.1
    assert len(row['scroll_offsets'])==len(frames)
    gray=cv2.imread(str(tmp_path/'good/system_000.png'),0)
    assert len(staffs(gray))==1
    # Real source intensities survive; this is neither binary repainting nor upscaling.
    assert ((gray>110)&(gray<230)).sum()>100
    frames=frames.copy()
    frames[10]=frames[7]  # demonstrably rightward source discontinuity
    with patch('harvest_scroll.cv2.VideoCapture',Capture):
        scan=write('fixture',tmp_path/'bad',geometry)
    assert 'scroll translation ambiguous or rightward discontinuity' in scan['records'][0]['scroll_reasons']
    assert not scan['records'][0]['cursor_x']


def test_partial_page_visit_withholds_unvisited_heads_without_changing_prior_gate(tmp_path):
    scan=json.loads((FIXTURES/'moyin-partial-row.json').read_text())
    (tmp_path/'scan.json').write_text(json.dumps(scan))
    gray=cv2.imread(str(FIXTURES/'moyin-partial-row.png'),0)
    cv2.imwrite(str(tmp_path/'system_000.png'),gray)
    legend=json.loads((HERE/'legends/moyin.json').read_text())
    profile=tmp_path/'legend.json'
    profile.write_text(json.dumps({**legend,'trim_partial_cursor':False}))
    full=read(tmp_path,profile)[0]
    profile.write_text(json.dumps(legend));partial=read(tmp_path,profile)[0]
    assert len(partial['notes'])<len(full['notes'])
    assert len(partial['notes'])+len(partial['unvisited_heads'])==len(full['notes'])
    assert partial['cursor_coverage']==full['cursor_coverage']<.35
    assert all(partial['read_x'][0]<=n['x']<=partial['read_x'][1] for n in partial['notes'])
    assert len(partial['cursor_x'])>=6


def test_daily_ghost_requires_two_parentheses_and_a_calibrated_position():
    cfg=json.loads((HERE/'legends/dailychops.json').read_text())['ghost_template']
    gray=cv2.imread(str(HERE/cfg['path']),0)
    n=dict(inst='tom',shape='filled',pos=7,x=cfg['anchor'][0],y=cfg['anchor'][1])
    assert ghost_score(gray,n,cfg['reference_spacing'],cfg)>.99
    for a,b in cfg['side_columns']:
        missing=gray.copy();missing[:,a:b]=np.median(missing[:,a:b],axis=1).astype(np.uint8)[:,None]
        assert ghost_score(missing,n,cfg['reference_spacing'],cfg)==0
    assert ghost_score(gray,{**n,'pos':5},cfg['reference_spacing'],cfg)==0
    assert ghost_score(gray,{**n,'shape':'cross'},cfg['reference_spacing'],cfg)==0
