"""Opt-in native ROI, dark-paper lines, gray cursor and no-cursor regressions."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from layouts import ink, detect_region, region_cursor
from heads import staffs, staff_checks


def test_white_strip_preserves_native_gray_lines_without_borrowing_overlay_policy():
    frame=np.full((180,500,3),15,np.uint8)
    for y in (65,80,95,110,125): frame[y:y+2]=115
    # Printed percussion clef. No geometric acceptance exception is used.
    frame[80:111,15:19]=245;frame[80:111,25:29]=245
    gray=ink(frame,'white_strip')
    found=staffs(gray,prefer_line_span=True)
    assert len(found)==1
    lines,space=found[0]
    assert space==pytest.approx(15)
    checks=staff_checks(gray,lines,space,[],strict_clef=True)
    assert checks['clef_confirmed'] and not checks['reasons']
    assert all(np.all(ink(frame,'white_overlay')[y:y+2,100:200]==255)
               for y in (65,80,95,110,125))
    # Saturated purple paper and pink/red cursors cannot become score ink.
    colors=np.array([[[60,10,90],[20,20,230],[235,235,235]]],np.uint8)
    assert ink(colors,'white_strip').tolist()==[[255,255,20]]


def test_legacy_ink_is_pixel_identical_to_numpy_channel_reductions():
    frame=np.random.default_rng(12).integers(0,256,(80,100,3),dtype=np.uint8)
    lo,hi=frame.min(2),frame.max(2)
    expected=np.where((hi>170)&(lo>105),hi,lo).astype(np.uint8)
    np.testing.assert_array_equal(ink(frame),expected)


def test_inspected_region_uses_native_pixels_and_late_notation():
    frame=np.full((300,600,3),255,np.uint8)
    for y in (220,232,244,256,268):frame[y]=0
    class Capture:
        def __init__(self,*args):self.i=0
        def get(self,key):return 1000
        def set(self,key,value):self.i=value
        def read(self):return True,frame.copy() if self.i>=750 else np.full_like(frame,255)
        def release(self):pass
    profile=dict(layout='kr_strip',score_region=[.1,.6,.9,1],cursor_style='none')
    with patch('layouts.cv2.VideoCapture',Capture):d=detect_region('local',profile)
    assert (d['left'],d['top'],d['right'],d['bottom'])==(60,180,540,300)
    assert all(s==pytest.approx(12) for s in d['native_spacing'])
    with patch('layouts.cv2.VideoCapture',Capture):
        with pytest.raises(ValueError):detect_region('local',{**profile,'score_region':[0,.9,1,.8]})


def test_gray_box_is_opt_in_and_static_scores_ignore_colored_icons():
    frame=np.full((180,600,3),255,np.uint8)
    frame[20:160,210:245]=185
    for y in (60,75,90,105,120):frame[y:y+2]=0
    box=region_cursor(frame,15,'gray_box')
    assert box and box[0]==pytest.approx(227)
    frame[20:160,210:245]=255;frame[20:45,210:245]=185
    assert region_cursor(frame,15,'gray_box') is None
    frame[0:180,100:110]=[20,20,230]
    assert region_cursor(frame,15,'none') is None


def test_native_dim_line_and_translucent_red_are_not_lost_to_bright_cursor_policy():
    purple=np.full((360,800,3),[40,10,25],np.uint8)
    purple[:,400:402]=[80,40,55]
    box=region_cursor(purple,17,'dim_line')
    assert box and box[0]==pytest.approx(400.5)
    purple[:,400:402]=[40,10,25];purple[160:230,400:402]=[80,40,55]
    assert region_cursor(purple,17,'dim_line') is None
    dark=np.full((280,800,3),12,np.uint8)
    dark[15:265,500:540]=[8,5,95]
    box=region_cursor(dark,17,'red_block')
    assert box and box[0]==pytest.approx(519.5)
    dark[15:265,500:540]=12;dark[15:265,250:520]=[8,5,95]
    assert region_cursor(dark,17,'red_block') is None


@pytest.mark.parametrize('name,style,x',[
    ('new-timbuell-red-native.png','red_block',1082.5),
    ('new-baquetazo-green-native.png','dim_line',1720.),
    ('new-baquetazo-purple-native.png','dim_line',864.),
])
def test_real_dark_pilot_cursors(name,style,x):
    frame=cv2.imread(str(Path(__file__).parent/'fixtures'/name))
    box=region_cursor(frame,17,style)
    assert box and box[0]==pytest.approx(x)
    assert box[2]-box[1]>=len(frame)*.7


def test_native_gray_box_is_not_merged_with_anti_aliased_staff_lines():
    frame=cv2.imread(str(Path(__file__).parent/'fixtures/new-gwon-gray-native.png'))
    box=region_cursor(frame,14,'gray_box')
    assert box and box[0]==pytest.approx(1013.5)
    assert box[1]>80 and box[2]>240 and 45<=box[3]<=60


def test_new_legends_never_enable_unverified_articulations_or_cymbals():
    for channel in ('timbuell','alexblack','baquetazo','copydrum','gwon','nyenty','yeom','bukey'):
        profile=json.loads((HERE/f'legends/{channel}.json').read_text())
        assert profile['harvest_profile'] and profile['strict_clef_pair']
        assert not profile.get('clef_omitted_verified')
        assert not profile.get('scroll_clef_omitted_verified')
        assert not profile.get('open_verified')
        assert all(not rule.get('verified') for key,rule in profile['rules'].items()
                   if key.startswith(('cross:','circled:','diamond:')))
    bukey=json.loads((HERE/'legends/bukey.json').read_text())
    assert bukey['cursor_style']=='none'
