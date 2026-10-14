"""Native polarity/guard regressions; no acoustic caches or models are used."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from batch import region_scan_signature
from heads import staffs
from harvest_scan import write
from layouts import channel_profile, detect_region, ink, select_score_variant, region_cursor
from reader_window import run
from strip import label_heads, read, stable_strip_runs

FIXTURES=Path(__file__).parent/'fixtures'
VARIANTS=[dict(foreground='white_strip',cursor_style='chromatic'),
          dict(foreground='dark_strip',cursor_style='chromatic')]


@pytest.mark.parametrize('video,channel,spacing,anchors',[
    ('6CLJH8YJtFM','timbuell',17.26,[(183.5,1,'kick'),(126.5,5,'snare'),(240.1,3,'tom'),(296.9,6,'tom')]),
    ('e2DLkb2lhnM','alexblack',19.14,[(130.7,1,'kick'),(245.4,5,'snare')]),
    ('LAdCiQAb-C0','baquetazo',15.09,[(87.5,1,'kick'),(87.7,5,'snare'),(465.6,7,'tom'),(627.4,6,'tom')]),
    ('YdNS8gjCX60','alexblack',20.18,[(206.1,1,'kick'),(274.2,5,'snare')]),
])
def test_native_light_variants_retain_printed_clef_geometry_and_existing_labels(video,channel,spacing,anchors):
    frame=cv2.imread(str(FIXTURES/f'variants-{video}-native.png'))
    legend,_=channel_profile(channel)
    index,found,reason=select_score_variant(frame,legend['score_variants'],True)
    assert index==1 and len(found)==1 and reason is None
    gray=ink(frame,legend['score_variants'][index]['foreground'])
    notes,checks=label_heads(gray,legend)
    assert checks['spacing']==pytest.approx(spacing,abs=.02)
    assert checks['clef_confirmed'] and not checks['reasons']
    for x,pos,inst in anchors:
        note=next(n for n in notes if n['pos']==pos and abs(n['x']-x)<1)
        assert note['shape']=='filled' and note['inst']==inst and note['exportable']
    assert all(not n['exportable'] for n in notes if n['shape']!='filled' or n['pos'] in (2,4,8))
    assert not any(n.get(flag) for n in notes for flag in ('ghost','accent','flam','roll'))


def synthetic(dark=False,space=12):
    frame=np.full((160,600,3),15 if dark else 245,np.uint8)
    for y in np.arange(45,45+space*5,space):
        frame[y:y+2,20:580]=115 if dark else 80
    frame[45+space:45+space*3+1,30:34]=245 if dark else 0
    frame[45+space:45+space*3+1,41:45]=245 if dark else 0
    cv2.ellipse(frame,(180,round(45+space*3.5)),(9,6),-20,0,360,
                (245,245,245) if dark else (0,0,0),-1)
    return frame


@pytest.mark.parametrize('dark,index',[(False,1),(True,0)])
def test_each_inspected_polarity_requires_real_supported_lines(dark,index):
    frame=synthetic(dark)
    chosen,found,reason=select_score_variant(frame,VARIANTS,True)
    assert chosen==index and len(found)==1 and reason is None
    frame[68:72]=15 if dark else 245  # Delete one real native line.
    assert select_score_variant(frame,VARIANTS,True)[0] is None


@pytest.mark.parametrize('fixture,channel',[
    ('new-timbuell-red-native.png','timbuell'),
    ('new-baquetazo-green-native.png','baquetazo'),
    ('new-baquetazo-purple-native.png','baquetazo')])
def test_existing_native_dark_pilots_keep_the_original_polarity(fixture,channel):
    frame=cv2.imread(str(FIXTURES/fixture));legend=channel_profile(channel)[0]
    index,found,reason=select_score_variant(frame,legend['score_variants'],True)
    assert index==0 and found and reason is None


def test_competing_supported_polarities_are_rejected_not_ranked_by_yield():
    # Two real fields of opposite polarity need separate inspected regions.
    index,found,reason=select_score_variant(np.vstack([synthetic(True),synthetic()]),VARIANTS,True)
    assert index is None and not found and reason=='ambiguous native score polarity'


def test_yellow_paper_green_cursor_uses_native_green_excess_not_brown_note_stems():
    frame=cv2.imread(str(FIXTURES/'variants-LAdCiQAb-C0-native.png'))
    box=region_cursor(frame,15.09,'green_line')
    assert box and box[0]==pytest.approx(1120.5,abs=1)
    assert box[2]-box[1]>=15.09*8 and box[3]<=15.09*.55
    # Remove only the real cursor; ordinary coloured staff/stems must not win.
    frame[:,1117:1125]=frame[:,1109:1117]
    assert region_cursor(frame,15.09,'green_line') is None


def test_green_cursor_rejects_short_marks_and_multiple_long_candidates():
    frame=np.full((180,600,3),(130,220,220),np.uint8)
    frame[0:180,200:202]=(115,180,150)
    assert region_cursor(frame,12,'green_line')[0]==pytest.approx(200.5)
    frame[0:180,300:302]=(115,180,150)
    assert region_cursor(frame,12,'green_line') is None
    frame[0:180,300:302]=(130,220,220);frame[80:180,200:202]=(130,220,220)
    assert region_cursor(frame,12,'green_line') is None


def capture(frames,fps=12):
    class Capture:
        def __init__(self,*args):self.i=0
        def get(self,key):return fps if key==cv2.CAP_PROP_FPS else len(frames)
        def set(self,key,value):self.i=int(value)
        def read(self):
            if self.i>=len(frames):return False,None
            frame=frames[self.i].copy();self.i+=1;return True,frame
        def release(self):pass
    return Capture


@pytest.mark.parametrize('variants',[None,VARIANTS])
def test_empty_survey_becomes_segment_rejection_and_later_staff_is_still_read(tmp_path,variants):
    blank=np.full((160,600,3),245,np.uint8)
    frames=[blank]*60
    profile=dict(name='fixture',layout='kr_strip',score_region=[0,0,1,1],
                 foreground='white_strip',cursor_style='chromatic',prefer_staff_line_span=True)
    if variants:profile['score_variants']=variants
    with patch('layouts.cv2.VideoCapture',capture(frames)):
        geometry=detect_region('fixture',profile)
    assert geometry['survey_reasons']==['no native five-line staff in inspected score region']
    # The initial survey missed the later score. No video-wide rejection.
    if variants:frames[24:48]=[synthetic()]*24
    with patch('harvest_scan.cv2.VideoCapture',capture(frames)):
        scan=write('fixture',tmp_path,geometry=geometry)
    assert scan['records']
    rejected=[r for r in scan['records'] if r.get('reading_reasons')]
    assert rejected and all('no native five-line' in r['reading_reasons'][0] for r in rejected)
    legend={**channel_profile('alexblack')[0],'score_region':[0,0,1,1]}
    path=tmp_path/'legend.json';path.write_text(json.dumps(legend))
    systems=read(tmp_path,path)
    assert len(systems)==len(scan['records'])
    assert all(s['staff']['reasons'] for s,r in zip(systems,scan['records']) if r.get('reading_reasons'))
    if variants:
        assert any(not s['staff']['reasons'] and s['notes'] for s in systems)


def test_native_small_staff_is_selected_for_inspection_but_rejected_by_unchanged_size_gate():
    frame=synthetic(space=7)
    chosen,found,reason=select_score_variant(frame,VARIANTS,True)
    assert chosen==1 and found and reason is None
    _,checks=label_heads(ink(frame,'dark_strip'),channel_profile('alexblack')[0])
    assert 'native staff spacing below 8 px' in checks['reasons']


def test_scanner_splits_polarity_changes_and_preserves_native_window_times(tmp_path):
    frames=[synthetic()]*24+[synthetic(True)]*24
    geometry=dict(layout='kr_strip',top=0,bottom=160,left=0,right=600,
                  native_spacing=[12],foreground='white_strip',cursor_style='chromatic',
                  prefer_staff_line_span=True,score_variants=VARIANTS)
    with patch('harvest_scan.cv2.VideoCapture',capture(frames)):
        scan=write('fixture',tmp_path,geometry=geometry,start_frame=12,end_frame=36)
    assert scan['scan_start_frame']==12 and scan['frames']==36
    assert [(r['start'],r['end'],r['foreground']) for r in scan['records']]==[
        (12,24,'dark_strip'),(24,36,'white_strip')]
    assert all(r['variant_sample_frames'] for r in scan['records'])


def test_multiple_native_staffs_remain_separate_and_need_cursor_visits(tmp_path):
    frames=[np.vstack([synthetic(),synthetic()])]*24
    geometry=dict(layout='kr_strip',top=0,bottom=320,left=0,right=600,
                  native_spacing=[12],foreground='white_strip',cursor_style='chromatic',
                  prefer_staff_line_span=True,score_variants=VARIANTS)
    with patch('harvest_scan.cv2.VideoCapture',capture(frames)):
        scan=write('fixture',tmp_path,geometry=geometry)
    assert len(scan['records'])==2 and all(r['page_rows']==2 for r in scan['records'])
    systems=read(tmp_path,HERE/'legends/alexblack.json')
    assert len(systems)==2
    assert all('page row not visited by cursor' in s['staff']['reasons'] for s in systems)
    assert all(not s['staff'].get('continuation_evidence') for s in systems)


def test_continuation_proof_never_crosses_a_polarity_change(tmp_path):
    gray=cv2.imread(str(FIXTURES/'daily-white-continuation.png'),0)
    _,space=staffs(gray,prefer_line_span=True)[0];records=[]
    for i in range(3):
        cv2.imwrite(str(tmp_path/f'system_{i:03d}.png'),gray)
        records.append(dict(system=i,start=i*10,end=(i+1)*10,page_rows=1,
                            crop=[0,len(gray)],native_space=space,foreground='dark_strip'))
    legend=channel_profile('alexblack')[0]
    assert set(stable_strip_runs(tmp_path,records,legend))=={0,1,2}
    records[1]['foreground']='white_strip'
    assert stable_strip_runs(tmp_path,records,legend)=={}


def test_inspected_variant_changes_invalidate_cached_scans():
    profile=channel_profile('alexblack')[0]
    changed={**profile,'score_variants':[VARIANTS[0]]}
    assert region_scan_signature('source',profile)!=region_scan_signature('source',changed)


def test_explicit_electronic_pad_annotation_withholds_only_the_audited_source_window(tmp_path):
    frame=cv2.imread(str(FIXTURES/'variants-e2DLkb2lhnM-claps-native.png'))
    cv2.imwrite(str(tmp_path/'system_000.png'),ink(frame,'dark_strip'))
    source=tmp_path/'source';source.mkdir();video=source/'video.mp4'
    record=dict(system=0,start=200,end=300,page_rows=1,crop=[0,len(frame)],
                native_space=20.6,cursor_frames=[],cursor_x=[])
    scan=dict(fps=25,video=str(video),layout='kr_strip',record_cursors=True,records=[record])
    (tmp_path/'scan.json').write_text(json.dumps(scan))
    path=HERE/'legends/alexblack.json'
    before=read(tmp_path,path)[0]
    assert any(n['exportable'] for n in before['notes'])
    (source/'meta.json').write_text(json.dumps(dict(video_id='e2DLkb2lhnM')))
    rejected=read(tmp_path,path)[0]
    assert any('electronic pad' in q for q in rejected['staff']['reasons'])
    assert not any(n['exportable'] for n in rejected['notes'])
    scan['records'][0].update(start=500,end=600)
    (tmp_path/'scan.json').write_text(json.dumps(scan))
    later=read(tmp_path,path)[0]
    assert not any('electronic pad' in q for q in later['staff']['reasons'])
    assert any(n['exportable'] for n in later['notes'])


@pytest.mark.parametrize('proof,reasons,applied',[(False,[],False),
    (True,['scroll translation ambiguous'],False),(True,[],True)])
def test_white_clef_diagnostics_preserve_the_registration_and_proof_guards(tmp_path,proof,reasons,applied):
    gray=cv2.imread(str(FIXTURES/'daily-white-continuation.png'),0)
    cv2.imwrite(str(tmp_path/'system_000.png'),gray)
    record=dict(system=0,start=0,end=20,page_rows=1,native_space=14,
                cursor_frames=[],cursor_x=[],scroll_reasons=reasons)
    if proof:record['staff_geometry']=dict(supporting_frames=list(range(20)))
    scan=dict(fps=5,foreground='white_overlay',layout='dailychops',
              reconstructed_scroll=True,record_cursors=True,records=[record])
    (tmp_path/'scan.json').write_text(json.dumps(scan))
    staff=read(tmp_path,HERE/'legends/dailychops.json')[0]['staff']
    assert staff['clef_guard']['continuation_applied']==applied
    assert ('percussion clef / first bar not confirmed' in staff['reasons'])!=applied
    assert staff['clef_guard']['registration_reasons']==reasons


@pytest.mark.parametrize('start,seconds',[(-1,4),(0,0),(0,21)])
def test_bounded_notation_tool_rejects_unbounded_windows(tmp_path,start,seconds):
    with pytest.raises(ValueError):run('unused','alexblack',tmp_path/'out',start,seconds)
