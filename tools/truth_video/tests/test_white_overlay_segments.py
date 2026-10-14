"""White native ROI, absent notation, and local registration regressions."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import cv2
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import batch
from heads import staffs,staff_checks
from layouts import white_staff
from harvest_scroll import write,motion_pair
from strip import read

FIXTURES=Path(__file__).parent/'fixtures'


def test_partial_native_staff_uses_actual_line_span_without_resizing():
    # qLi8dFevOF0 frame 252, source 1920x1080, native y=833:941.
    gray=cv2.imread(str(FIXTURES/'daily-white-partial-native.png'),0)
    assert not staffs(gray,prefer_line_span=True)
    found,bounds=white_staff(gray)
    assert len(found)==1 and bounds[1]-bounds[0]<gray.shape[1]/2
    lines,space=found[0]
    assert 11<space<12
    checks=staff_checks(gray[:,bounds[0]:bounds[1]],lines,space,[],strict_clef=True)
    assert min(checks['line_support'])>=.45
    assert 'uneven staff lines' not in checks['reasons']
    assert 'native staff spacing below 8 px' not in checks['reasons']
    # An unprinted fifth line cannot be supplied by the windowing search.
    missing=gray.copy();missing[round(lines[2])-2:round(lines[2])+3]=255
    assert not white_staff(missing)[0]


def test_partial_search_keeps_two_distinct_native_staffs_ambiguous():
    gray=cv2.imread(str(FIXTURES/'daily-white-partial-native.png'),0)
    found,bounds=white_staff(np.concatenate([gray,gray]))
    assert len(found)==2 and bounds is None


def test_motion_uses_score_span_instead_of_stationary_video_ink():
    rng=np.random.default_rng(42)
    lines=[40.,52.,64.,76.,88.]
    score=np.full((110,900),255,np.uint8)
    for x,y in rng.integers([320,12],[790,100],size=(40,2)):
        cv2.circle(score,(int(x),int(y)),3,0,-1)
    moved=cv2.warpAffine(score,np.float32([[1,0,-13],[0,1,0]]),
                         (900,110),borderValue=255)
    # Dense stationary footage to the left must not outvote actual score ink.
    for image in (score,moved):image[:,:230]=0
    samples=[(i,g,None,g,[(lines,12.)],[280,840]) for i,g in enumerate((score,moved))]
    delta=motion_pair(*samples,dict(foreground='white_overlay',motion_left=0,motion_right=900))
    assert delta and delta['dx']==-13 and delta['correlation']>=.88 and delta['uniqueness']>=.025


def capture(frames):
    class Capture:
        def __init__(self,*args):self.i=0
        def get(self,key):return 5 if key==5 else len(frames)
        def read(self):
            if self.i==len(frames):return False,None
            frame=frames[self.i];self.i+=1;return True,frame
        def release(self):pass
    return Capture


def native_frames():
    data=np.load(FIXTURES/'daily-white-scroll-native.npz')
    frames=data['frames']
    geometry=dict(layout='dailychops',foreground='white_overlay',top=0,bottom=frames.shape[1],
                  native_spacing=[float(data['spacing'])],motion_left=int(data['motion_left']),
                  motion_right=int(data['motion_right']),split_scroll_discontinuities=True)
    return frames,geometry


def test_absent_intro_is_one_rejected_gap_and_later_scroll_is_read(tmp_path):
    frames,geometry=native_frames()
    blank=np.zeros_like(frames[:6])
    sequence=np.concatenate([blank,frames[:14]])
    with patch('harvest_scroll.cv2.VideoCapture',capture(sequence)):
        scan=write('fixture',tmp_path,geometry)
    absent=[r for r in scan['records'] if 'white-overlay native staff absent or partial lines unconfirmed' in r['scroll_reasons']]
    assert len(absent)==1 and (absent[0]['start'],absent[0]['end'])==(0,6)
    accepted=[r for r in scan['records'] if not r['scroll_reasons']]
    assert accepted and all(r['start']>=6 for r in accepted)
    systems=read(tmp_path,HERE/'legends/dailychops.json')
    assert systems[0]['staff']['reasons'] and not systems[0]['notes']
    assert any(not s['staff']['reasons'] for s in systems[1:])


def test_fixed_cursor_on_static_score_still_rejects_without_traversal(tmp_path):
    frames,geometry=native_frames()
    with patch('harvest_scroll.cv2.VideoCapture',capture(np.repeat(frames[:1],20,axis=0))):
        scan=write('fixture',tmp_path,{**geometry,'global_motion_confirmed':False})
    assert scan['records']
    assert all('scroll cursor missing or insufficient traversed span' in r['scroll_reasons'] for r in scan['records'])
    assert all(not r.get('staff_geometry') for r in scan['records'])
    assert all(s['staff']['reasons'] for s in read(tmp_path,HERE/'legends/dailychops.json'))


def test_false_global_probe_cannot_veto_independently_registered_segments(tmp_path):
    frames,geometry=native_frames()
    with patch('harvest_scroll.cv2.VideoCapture',capture(frames)):
        scan=write('fixture',tmp_path,{**geometry,'global_motion_confirmed':False})
    assert any(not r['scroll_reasons'] for r in scan['records'])
    assert any(not s['staff']['reasons'] for s in read(tmp_path,HERE/'legends/dailychops.json'))


def test_worker_false_probe_reaches_white_segment_reader(tmp_path,monkeypatch):
    import harvest_scroll,harvest_scan,layouts,strip,sanity,align,tiers
    song=tmp_path/'song';song.mkdir();(song/'video.mp4').write_bytes(b'video')
    out=tmp_path/'out';profile_path=tmp_path/'profile.json'
    profile_path.write_text(json.dumps(dict(name='fixture',layout='dailychops',enabled=True,
        white_overlay_verified=True,scrolling_verified=True,split_scroll_discontinuities=True,rules={})))
    monkeypatch.setattr(layouts,'detect',lambda *a:(_ for _ in ()).throw(ValueError('not light paper')))
    monkeypatch.setattr(layouts,'detect_white_overlay',lambda *a:dict(foreground='white_overlay'))
    monkeypatch.setattr(harvest_scroll,'probe',lambda *a:(False,[None]))
    calls=[]
    def segment_scan(video,folder,geometry,**kwargs):
        assert geometry['global_motion_confirmed'] is False
        calls.append('segments')
        data=dict(foreground='white_overlay',reconstructed_scroll=True,records=[dict(system=0,
            scroll_reasons=['scroll cursor missing or insufficient traversed span'])])
        batch.dump(folder/'scan.json',data);return data
    monkeypatch.setattr(harvest_scroll,'write',segment_scan)
    monkeypatch.setattr(harvest_scan,'write',lambda *a,**k:(_ for _ in ()).throw(AssertionError('wrong static scanner')))
    def read_strip(folder,*a):
        batch.dump(folder/'strip.json',[]);return []
    monkeypatch.setattr(strip,'read',read_strip)
    monkeypatch.setattr(sanity,'check',lambda *a:dict(flags=[]))
    monkeypatch.setattr(align,'build_truth',lambda *a,**k:None)
    monkeypatch.setattr(tiers,'write',lambda *a:None)
    monkeypatch.setattr(batch,'summarize',lambda *a:{})
    monkeypatch.setattr(batch,'quality_report',lambda *a:None)
    batch.worker(song,out,profile_path)
    assert calls==['segments'] and (out/'result.json').exists()
    scan=json.loads((out/'scan.json').read_text())
    assert scan['records'][0]['scroll_reasons']==['scroll cursor missing or insufficient traversed span']
