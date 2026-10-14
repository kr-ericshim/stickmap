"""Native xiangsong paper/cursor/ghost regressions and sealed export scope."""
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
from layouts import detect, ink, region_cursor, shorts_paper_top
from strip import label_heads
import pilot_new

FIXTURES=Path(__file__).parent/'fixtures'
PROFILE=json.loads((HERE/'legends/xiangsong.json').read_text())
PROVENANCE=json.loads((FIXTURES/'xiangsong-provenance.json').read_text())


def test_dense_native_top_row_does_not_replace_the_paper_boundary():
    page=cv2.imread(str(FIXTURES/'xiangsong-dense-page-native.png'))
    frame=np.full((1920,1080,3),30,np.uint8);frame[1085:]=page
    assert shorts_paper_top(frame)>1085  # Dense ink defeated the older survey.
    assert shorts_paper_top(frame,edge_margins=True)==1085
    class Capture:
        def __init__(self,*args):pass
        def get(self,key):return 1755
        def set(self,*args):pass
        def read(self):return True,frame.copy()
        def release(self):pass
    with patch('layouts.cv2.VideoCapture',Capture): geometry=detect('local','xiangsong')
    assert (geometry['top'],geometry['bottom'])==(1085,1920)
    assert geometry['layout']=='xiangsong' and geometry['cursor_style']=='blue_box'
    assert min(geometry['native_spacing'])>8 and geometry['prefer_staff_line_span']


def test_native_blue_plate_is_unique_and_blue_note_ink_alone_is_insufficient():
    page=cv2.imread(str(FIXTURES/'xiangsong-dense-page-native.png'))
    box=region_cursor(page,9.8,'blue_box')
    assert box==pytest.approx([878.5,23,112,34])
    assert region_cursor(page[60:90,850:910],9.8,'blue_box') is None
    plate=page[20:115,850:910]
    twice=np.full((120,180,3),255,np.uint8)
    twice[:len(plate),:60]=plate;twice[:len(plate),100:160]=plate
    assert region_cursor(twice,9.8,'blue_box') is None


@pytest.mark.parametrize('tag',('ghost','plain'))
def test_native_snare_parentheses_and_plain_neighbour_controls(tag):
    row=next(r for r in PROVENANCE if r['fixture']==f'xiangsong-{tag}-row-native.png')
    frame=cv2.imread(str(FIXTURES/row['fixture']))
    notes,checks=label_heads(ink(frame),PROFILE)
    assert not checks['reasons'] and checks['spacing']>8
    n=min(notes,key=lambda n:abs(n['x']-row['target']['x'])+abs(n['y']-row['target']['y']))
    assert abs(n['x']-row['target']['x'])<1 and n['pos']==5
    assert n['exportable'] and n['inst']=='snare' and n['ghost']==row['expected_ghost']
    assert not any(n['ghost'] for n in notes if n['inst']!='snare' or n['pos']!=5)
    if tag=='ghost':
        assert n['ghost_confidence']>=.95
        # Keep the same head but remove either actual parenthesis. A note head
        # and a nearby stem must not independently manufacture a ghost flag.
        for side in (-1,1):
            gray=ink(frame);x=round(n['x']);y=round(n['y']);sp=checks['spacing']
            lo,hi=sorted((round(x+side*sp*.6),round(x+side*sp*1.5)))
            gray[max(0,round(y-sp)):round(y+sp),lo:hi]=255
            from harvest_flags import ghost_score
            assert not any(ghost_score(gray,n,sp,cfg)>=cfg['threshold'] for cfg in PROFILE['ghost_templates'])


def test_native_size_clef_and_partial_label_gates_are_preserved():
    assert PROFILE['test_only'] and PROFILE['strict_clef_pair']
    assert not PROFILE.get('clef_omitted_verified') and not PROFILE.get('scrolling_verified')
    assert PROFILE['verified_flags']==['ghost']
    assert {k for k,r in PROFILE['rules'].items() if r['verified']}=={
        'filled:1','filled:3','filled:5','filled:6','filled:7'}
    # The observed slash at p8 and all physical cymbal identities stay withheld.
    assert not PROFILE['rules']['filled:8']['verified']
    assert not any(r['verified'] for r in PROFILE['rules'].values() if r['inst']=='hihat')
    small=cv2.imread(str(FIXTURES/'shorts-roni-row-native.png'))
    _,checks=label_heads(ink(small),PROFILE)
    assert 'native staff spacing below 8 px' in checks['reasons']


def test_embedded_audio_pilot_metadata_uses_na_list_and_remains_test_only(tmp_path,monkeypatch):
    folder=tmp_path/'.eval-cache/pilot_new/xiangsong/local';folder.mkdir(parents=True)
    (folder/'video.mp4').write_bytes(b'local')
    lists=tmp_path/'truth/harvest/lists';lists.mkdir(parents=True)
    (lists/'xiangsong.txt').write_text('local|NA|xiangsong reel\n')
    monkeypatch.setattr(pilot_new,'ROOT',tmp_path)
    class Capture:
        def __init__(self,*args):pass
        def get(self,key):return 30 if key==5 else 969
        def release(self):pass
    with patch('cv2.VideoCapture',Capture): meta=pilot_new.metadata(folder)
    assert meta['seconds']==32.3 and meta['title']=='xiangsong reel'
    assert meta['channel']=='xiangsong' and meta['test_only']
    assert not (folder/'audio.m4a').exists()


def test_sealed_config_cannot_enter_detector_fetch_or_vault_training():
    root=HERE.parents[1]
    config=json.loads((root/'tools/harvest/channels.json').read_text())
    entry=next(r for r in config['channels'] if r['name']=='xiangsong')
    assert entry['listing']=='file' and entry['url_template']=='https://www.instagram.com/reel/{id}/'
    assert entry['video_format']=='bv*[width<=1080]/b' and entry['sleep']==25
    assert entry['layout']=='xiangsong' and entry['fetch_enabled'] is False
    assert 'xiangsong' not in json.loads((root/'truth/harvest/vault.json').read_text())['train_channels']


def test_native_fixtures_are_unscaled_and_have_matching_provenance():
    for r in PROVENANCE:
        path=FIXTURES/r['fixture'];image=cv2.imread(str(path))
        x0,y0,x1,y1=r['crop']
        assert not r['resized'] and image.shape[:2]==(y1-y0,x1-x0)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
