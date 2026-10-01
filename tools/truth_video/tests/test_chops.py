"""Inset clefs, page-local visual timing and fail-closed promotional overlays."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from heads import staff_checks
from color_timing import anchors,attach
from harvest_scan import chromatic_obstructions
from import_samples import register
from batch import manifest_read

SCRATCH=Path(tempfile.gettempdir())/'groovemap-truth-video-tests';SCRATCH.mkdir(parents=True,exist_ok=True)


def staff(space=9,clef=True):
    g=np.full((120,1280),255,np.uint8);lines=[40+i*space for i in range(5)]
    for y in lines:g[y,181:1100]=120
    if clef:
        g[lines[1]:lines[3]+1,186:191]=0
        g[lines[1]:lines[3]+1,194:200]=0
    return g,lines


def test_inset_clef_at_actual_staff_start_and_raster_width():
    g,lines=staff();r=staff_checks(g,lines,9,[dict(x=230,inst='kick')],strict_clef=True)
    assert r['clef_confirmed']
    assert 'percussion clef / first bar not confirmed' not in r['reasons']
    assert not staff_checks(g,lines,9,[dict(x=230,inst='kick')])['clef_confirmed']


def test_strict_pair_rejects_two_stems_that_legacy_margin_mistakes_for_clef():
    g,lines=staff(clef=False)
    # No left inset: the legacy check would count both note stems as a clef.
    for y in lines:g[y,0:1100]=120
    g[49:69,60:63]=0;g[49:69,100:103]=0
    notes=[dict(x=60,inst='kick'),dict(x=100,inst='kick')]
    assert staff_checks(g,lines,9,notes)['clef_confirmed']
    assert not staff_checks(g,lines,9,notes,strict_clef=True)['clef_confirmed']


def test_missing_clef_is_not_rescued_by_later_stems_or_first_bar():
    g,lines=staff(clef=False)
    g[40:77,181:184]=0
    g[49:69,260:265]=0;g[49:69,280:286]=0
    r=staff_checks(g,lines,9,[dict(x=230,inst='kick')])
    assert not r['clef_confirmed']
    assert 'percussion clef / first bar not confirmed' in r['reasons']


def test_strict_clef_rejects_heads_immediately_after_inset_staff_start():
    from strip import label_heads
    g,lines=staff(clef=False)
    for x in (185,198,270,340,410,480,550,620):
        cv2.ellipse(g,(x,72),(5,4),-15,0,360,0,-1)
        g[49:73,x+4:x+7]=0
    legend=json.loads((Path(__file__).parents[1]/'legends/olliewoods.json').read_text())
    notes,checks=label_heads(g,legend)
    assert len(notes)==8
    assert np.allclose([n['x'] for n in notes[:2]],[185,198],atol=1)
    assert not checks['clef_confirmed']
    assert 'percussion clef / first bar not confirmed' in checks['reasons']


def test_strict_clef_keeps_real_dennis_chambers_native_pair():
    from strip import label_heads
    # Existing fixture is a native crop at x=170; restore that inset without scaling.
    g=cv2.imread(str(Path(__file__).parent/'fixtures/olliewoods.png'),0)
    g=cv2.copyMakeBorder(g,0,0,170,0,cv2.BORDER_CONSTANT,value=255)
    legend=json.loads((Path(__file__).parents[1]/'legends/olliewoods.json').read_text())
    notes,checks=label_heads(g,legend)
    assert min(n['x'] for n in notes)>199
    assert checks['clef_confirmed']
    assert 'percussion clef / first bar not confirmed' not in checks['reasons']


def test_inset_clef_does_not_bypass_native_spacing_gate():
    g,lines=staff(space=7);r=staff_checks(g,lines,7,[])
    assert 'native staff spacing below 8 px' in r['reasons']


def colored_system():
    notes=[dict(x=40+i*70,y=75,inst='kick',pos=1,shape='filled') for i in range(6)]
    s=dict(system=0,interval=[0,1],notes=notes,cursor_x=[],cursor_t=[],
           staff=dict(spacing=10,lines=[40,50,60,70,80],
                      reasons=['page row not visited by cursor']))
    base=np.full((110,500,3),255,np.uint8)
    for n in notes:cv2.ellipse(base,(n['x'],75),(6,4),0,0,360,(0,0,0),-1)
    frames=[]
    for i in range(28):
        f=base.copy()
        for j,n in enumerate(notes):
            if i in (2+j*4,3+j*4):cv2.ellipse(f,(n['x'],75),(6,4),0,0,360,(25,160,25),-1)
        frames.append(f)
    return s,base,frames


def measure(s,base,frames,page_notes):
    class Capture:
        def __init__(self,*a):self.i=0
        def read(self):
            if self.i==len(frames):return False,None
            f=frames[self.i];self.i+=1;return True,f
        def release(self):pass
    with tempfile.TemporaryDirectory(dir=SCRATCH) as name:
        p=Path(name);cv2.imwrite(str(p/'system_000.png'),cv2.cvtColor(base,cv2.COLOR_BGR2GRAY))
        (p/'strip.json').write_text(json.dumps([s]))
        (p/'scan.json').write_text(json.dumps(dict(video='synthetic',fps=30,top=0,
            records=[dict(system=0,start=0,end=len(frames),crop=[0,110])])))
        with patch('color_timing.cv2.VideoCapture',Capture):events=attach(p,page_notes=page_notes)
        return events,json.loads((p/'strip.json').read_text())[0]


def test_page_visit_requires_six_broad_stable_green_head_edges():
    s,base,frames=colored_system()
    events,read=measure(s,base,frames,True)
    assert len(events)==6 and len(read['color_x'])==6
    assert read['color_t']==sorted(read['color_t'])
    assert not read['staff']['reasons']
    assert read['display_interval']==[0,1]
    assert events[0]['first_frame']==2 and events[0]['uncertainty_seconds']==1/30
    assert len(measure(s,base,frames,False)[0])==0  # opt-in only


def test_pale_box_never_establishes_a_page_visit_or_bypasses_staff_failures():
    s,base,frames=colored_system();pale=[]
    for i in range(28):
        f=base.copy();x=40+min(5,i//4)*70
        # Keep the head black; a pale cursor background is not local colour ink.
        f[45:95,x-12:x+13]=np.maximum(f[45:95,x-12:x+13],(220,230,250))
        for n in s['notes']:cv2.ellipse(f,(n['x'],75),(6,4),0,0,360,(0,0,0),-1)
        pale.append(f)
    events,read=measure(s,base,pale,True)
    assert not events and 'color_x' not in read
    assert 'page row not visited by cursor' in read['staff']['reasons']
    s['staff']['reasons'].append('native staff spacing below 8 px')
    assert not measure(s,base,frames,True)[0]


def test_colour_anchors_reject_sparse_or_reversed_times():
    s,_,_=colored_system()
    for i,n in enumerate(s['notes']):n['visual_timing']=dict(time=i/10)
    assert anchors(s,30)[0]
    s['notes'][3]['visual_timing']['time']=0
    assert anchors(s,30)==([],[])
    s['notes'][3]['visual_timing']['time']=.3
    del s['notes'][0]['visual_timing']
    assert anchors(s,30)==([],[])


def test_promotional_letters_are_distinct_from_heads_and_cursor():
    f=np.full((140,500,3),255,np.uint8)
    f[10:120,100:103]=(25,220,25)  # green line
    f[15:125,150:180]=(180,245,250)  # pale yellow box
    cv2.ellipse(f,(220,70),(7,4),0,0,360,(25,160,25),-1)
    assert chromatic_obstructions(f,10)==[]
    cv2.putText(f,'MORE',(260,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,210),6)
    assert chromatic_obstructions(f,10)


def test_local_sample_registration_move_recovery_and_missing_video():
    with tempfile.TemporaryDirectory(dir=SCRATCH) as name:
        base=Path(name);samples=base/'samples';root=base/'harvest'
        for slug,video,with_video in [('chop','abcdefghijk',True),('missing','01234567890',False)]:
            p=samples/'golly'/slug;p.mkdir(parents=True)
            (p/'meta.json').write_text(json.dumps(dict(uploader='Vincent Golly',title='Analyzing Gospel Chops',video_id=video,duration=40)))
            (p/'attacks.htdemucs.npy').write_bytes(b'retained')
            if with_video:(p/'video.mp4').write_bytes(b'retained-video')
        rows=register('golly',root,samples)
        assert len(rows)==2 and all(r['status']=='fetched' for r in rows)
        assert (root/'songs/chop/video.mp4').read_bytes()==b'retained-video'
        assert not (samples/'golly/chop').exists()
        assert manifest_read(root/'manifest.jsonl')['01234567890']['retained_missing_inputs']==['video.mp4']
        assert register('golly',root,samples)==[]
        # A moved directory with no journal row is recoverable without copying.
        (root/'manifest.jsonl').unlink()
        assert len(register('golly',root,samples))==2


def test_native_frame_consensus_recovers_real_failed_median_without_scaling():
    from harvest_scan import stable_native_staff
    from heads import staffs
    fixtures=Path(__file__).parent/'fixtures'
    median=cv2.imread(str(fixtures/'ivannikov-intro-median.png'),0)
    native=cv2.imread(str(fixtures/'ivannikov-intro-native.png'),0)
    assert not staffs(median,min_space=6)
    assert len(staffs(native,min_space=6))==1
    recovered=stable_native_staff([native]*8,list(range(8)))
    assert recovered is not None
    image,geometry,evidence=recovered
    assert np.array_equal(image,native) and image.shape==median.shape
    assert geometry[0][1]>=8 and len(evidence['supporting_frames'])==8
    assert stable_native_staff([native]*5,list(range(5))) is None
    assert stable_native_staff([native]*4+[np.roll(native,4,axis=0)]*4,list(range(8))) is None
    changed=native.copy();changed[:35,300:600]=0
    assert stable_native_staff([native]*6+[changed]*2,list(range(8))) is None


def fake_tom_reservoir(count=1):
    from heads import staffs
    from strip import label_heads
    native=cv2.imread(str(Path(__file__).parent/'fixtures/ivannikov-intro-native.png'),0)
    _,space=staffs(native)[0]
    lines=staffs(native)[0][0]
    injected=native.copy()
    # Same x as a real kick: the injected head adds no new timing moment.
    cv2.ellipse(injected,(421,round(lines[-1]-7*space/2)),(5,3),-15,0,360,0,-1)
    legend=json.loads((Path(__file__).parents[1]/'legends/ivannikov.json').read_text())
    notes,checks=label_heads(injected,legend)
    assert not checks['reasons']
    assert any(n['inst']=='tom' and n['pos']==7 and n['exportable'] and abs(n['x']-421)<2 for n in notes)
    assert ((native<100)!=(injected<100)).mean()<.028
    samples=[native]*24
    # stable_native_staff selects supporting[len(supporting)//2], index 12.
    for i in [12,*[j for j in range(24) if j!=12]][:count]:samples[i]=injected
    return samples


def test_native_frame_consensus_withholds_single_frame_exportable_fake_tom():
    from harvest_scan import stable_native_staff
    from heads import staffs
    from strip import label_heads
    from align import moments
    with np.load(Path(__file__).parent/'fixtures/ivannikov-intro-reservoir.npz') as data:
        samples=list(data['samples']);frames=data['frames'].tolist()
    # The real 24-frame Jojo reservoir fails median search but has 21 agreeing frames.
    assert not staffs(np.median(np.stack(samples),axis=0).astype(np.uint8),min_space=6)
    clean=stable_native_staff(samples,frames)
    assert clean is not None and len(clean[2]['supporting_frames'])==21
    # The pre-fix selector picked frame 318, independently of note-head support.
    selected=frames.index(318)
    lines,space=staffs(samples[selected])[0]
    legend=json.loads((Path(__file__).parents[1]/'legends/ivannikov.json').read_text())
    before,checks=label_heads(samples[selected],legend)
    assert not checks['reasons']
    injected=samples[selected].copy()
    cv2.ellipse(injected,(421,round(lines[-1]-7*space/2)),(5,3),-15,0,360,0,-1)
    after,checks=label_heads(injected,legend)
    assert not checks['reasons']
    assert any(n['inst']=='tom' and n['pos']==7 and n['exportable'] and abs(n['x']-421)<2 for n in after)
    assert len(moments(dict(notes=before,staff=checks))[0])==len(moments(dict(notes=after,staff=checks))[0])
    assert all(((injected<100)!=(g<100)).mean()<=.028 for g in samples)
    samples[selected]=injected
    assert not staffs(np.median(np.stack(samples),axis=0).astype(np.uint8),min_space=6)
    recovered=stable_native_staff(samples,frames)
    if recovered is not None:
        notes,_=label_heads(recovered[0],legend)
        assert not any(n['inst']=='tom' and n['pos']==7 and n['exportable'] and abs(n['x']-421)<2 for n in notes)
        evidence=recovered[2]
        assert evidence['head_support_required']==11
        assert all(len(h['supporting_frames'])>=11 for h in evidence['head_support'])


def test_native_frame_consensus_requires_strict_majority_for_every_head():
    from harvest_scan import stable_native_staff
    from strip import label_heads
    legend=json.loads((Path(__file__).parents[1]/'legends/ivannikov.json').read_text())
    for count in (12,13):
        recovered=stable_native_staff(fake_tom_reservoir(count),list(range(24)))
        assert recovered is not None
        notes,_=label_heads(recovered[0],legend)
        fake=[n for n in notes if n['inst']=='tom' and n['pos']==7 and n['exportable'] and abs(n['x']-421)<2]
        assert bool(fake)==(count==13)
        assert recovered[2]['head_support_required']==13
        assert all(len(h['supporting_frames'])>=13 for h in recovered[2]['head_support'])
