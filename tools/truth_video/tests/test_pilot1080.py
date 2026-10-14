"""Native pilot regressions, failed motion boundaries and TEST-ONLY fetch guard."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import cv2
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from heads import staffs, staff_checks
from layouts import ink, cursor
from harvest_flags import clef_score, ghost_score
from harvest_scroll import write

FIXTURES=Path(__file__).parent/'fixtures'


def test_fractional_native_staff_centroid_samples_the_actual_line():
    gray=cv2.imread(str(FIXTURES/'daily-lesson-native.png'),0)
    lines,space=staffs(gray,prefer_line_span=True)[0]
    # The fourth centroid is 110.99; row 110 is nearly blank, row 111 is staff.
    assert 110.9<lines[3]<111
    assert (gray[int(lines[3])]<230).mean()<.2
    checks=staff_checks(gray,lines,space,[],strict_clef=True)
    assert min(checks['line_support'])>.95
    assert checks['clef_confirmed'] and not checks['reasons']


def test_grey_clef_needs_both_native_strokes():
    cfg=json.loads((HERE/'legends/woshudiue.json').read_text())['clef_template']
    small=cv2.imread(str(FIXTURES/'woshudiue-grey-clef.png'),0)
    gray=np.full((len(small),1550),255,np.uint8);gray[:,:130]=small
    lines,space=staffs(cv2.imread(str(FIXTURES/'woshudiue-grey-clef.png'),0),prefer_line_span=True)[0]
    assert not staff_checks(gray,lines,space,[],strict_clef=True)['clef_confirmed']
    assert clef_score(gray,lines,space,[],cfg)>.99
    for a,b in ((18,24),(26,32)):
        missing=gray.copy();missing[:,a:b]=np.median(missing[:,a:b],axis=1).astype(np.uint8)[:,None]
        assert clef_score(missing,lines,space,[],cfg)==0


def test_white_overlay_cursor_rejects_performance_colours():
    data=np.load(FIXTURES/'daily-white-scroll-native.npz');frame=data['frames'][0].copy()
    found=cursor(frame,foreground='white_overlay')
    assert found and abs(found[0]-frame.shape[1]/2)<25
    x=round(found[0]);frame[:,x-40:x+40]=[10,220,20]
    assert cursor(frame,foreground='white_overlay') is None
    # Saturated stage colours/red cursor do not become black notation.
    sample=np.array([[[255,255,255],[0,0,255],[0,240,0],[80,80,80]]],np.uint8)
    assert ink(sample,'white_overlay').tolist()==[[0,255,255,255]]


def test_white_scroll_never_registers_across_a_bad_native_edge(tmp_path):
    data=np.load(FIXTURES/'daily-white-scroll-native.npz');frames=data['frames'].copy()
    frames[10]=frames[7]  # real rightward discontinuity
    class Capture:
        def __init__(self,*args):self.i=0
        def get(self,key):return 5 if key==5 else len(frames)
        def read(self):
            if self.i==len(frames):return False,None
            frame=frames[self.i];self.i+=1;return True,frame
        def release(self):pass
    geometry=dict(layout='dailychops',foreground='white_overlay',left=0,right=frames.shape[2],
                  top=0,bottom=frames.shape[1],native_spacing=[float(data['spacing'])],
                  motion_left=int(data['motion_left']),motion_right=int(data['motion_right']),
                  split_scroll_discontinuities=True)
    with patch('harvest_scroll.cv2.VideoCapture',Capture):scan=write('fixture',tmp_path,geometry)
    rows=scan['records']
    assert any('scroll translation ambiguous or rightward discontinuity' in r['scroll_reasons'] for r in rows)
    good=[r for r in rows if not r['scroll_reasons']]
    assert good and all(not (r['start']<10<r['end']) for r in good)
    assert all(len(r['sample_frames'])>=5 and r['native_space']>=8 for r in good)


def test_woshudiue_parentheses_cannot_be_supplied_by_one_side():
    profile=json.loads((HERE/'legends/woshudiue.json').read_text())
    for cfg in [profile['ghost_template'],*profile['ghost_templates']]:
        gray=cv2.imread(str(HERE/cfg['path']),0)
        n=dict(inst='snare',shape='filled',pos=5,x=cfg['anchor'][0],y=cfg['anchor'][1])
        assert ghost_score(gray,n,cfg['reference_spacing'],cfg)>.99
        for a,b in cfg['side_columns']:
            missing=gray.copy();missing[:,a:b]=np.median(missing[:,a:b],axis=1).astype(np.uint8)[:,None]
            assert ghost_score(missing,n,cfg['reference_spacing'],cfg)==0


def test_white_continuation_requires_this_videos_native_opening_clef(tmp_path):
    from strip import read
    gray=cv2.imread(str(FIXTURES/'daily-white-continuation.png'),0)
    cv2.imwrite(str(tmp_path/'system_000.png'),gray)
    _,space=staffs(gray,prefer_line_span=True)[0]
    record=dict(system=0,start=0,end=20,row=0,page_rows=1,crop=[0,len(gray)],native_space=space,
                cursor_frames=list(range(20)),cursor_x=np.linspace(20,gray.shape[1]-20,20).tolist(),scroll_reasons=[])
    scan=dict(fps=5,layout='dailychops',foreground='white_overlay',reconstructed_scroll=True,
              record_cursors=True,records=[record])
    (tmp_path/'scan.json').write_text(json.dumps(scan))
    before=read(tmp_path,HERE/'legends/dailychops.json')[0]
    assert 'percussion clef / first bar not confirmed' in before['staff']['reasons']
    scan['opening_clef']=dict(frame=300,method='native strict paired-stroke opening proof')
    (tmp_path/'scan.json').write_text(json.dumps(scan))
    after=read(tmp_path,HERE/'legends/dailychops.json')[0]
    assert 'percussion clef / first bar not confirmed' not in after['staff']['reasons']
    assert after['staff']['spacing']==before['staff']['spacing']


def test_current_detector_fetcher_cannot_fetch_test_only_channel(monkeypatch):
    sys.path.insert(0,str(HERE.parent/'harvest'))
    import harvest
    profile=next(c for c in json.loads((HERE.parent/'harvest/channels.json').read_text())['channels'] if c['name']=='woshudiue')
    monkeypatch.setattr(harvest,'listing',lambda *args:[('SHvfn-bM_rg',182,"Funkin Fusion Drum Cover 鼓譜 Drumtabs")])
    assert profile['test_only'] and not profile['trainable'] and not profile['fetch_enabled']
    assert harvest.candidates(profile)==[]
