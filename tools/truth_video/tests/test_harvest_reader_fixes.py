"""Local native regressions; no video batches, separation or detector calls."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import cv2
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import batch
from heads import staffs
from layouts import detect_white_overlay
from strip import label_heads, read, stable_strip_runs
from align import moments, emit
from manifest_io import append, read as read_manifest

FIXTURES=Path(__file__).parent/'fixtures'


def test_native_tie_is_inspectable_but_never_a_new_onset():
    # Native crop: 4VX2domLud8, system 10, retained reader-repair source.
    gray=cv2.imread(str(FIXTURES/'olliewoods-tie-native.png'),0)
    legend=json.loads((HERE/'legends/olliewoods.json').read_text())
    notes,staff=label_heads(gray,legend)
    start=next(n for n in notes if abs(n['x']-255.6)<2)
    continuation=next(n for n in notes if abs(n['x']-338.6)<2)
    assert start['exportable'] and not start.get('tied_continuation')
    assert continuation['tied_continuation'] and not continuation['exportable']
    assert continuation['tie_evidence']['source_x']==start['x']
    system=dict(system=10,notes=notes,staff=staff)
    xs,groups=moments(system)
    assert len(xs)==len(notes)-1
    assert all(notes[i] is not continuation for group in groups for i in group)
    report=dict(usable=True,times=list(range(len(xs))),moment_groups=groups,
                held_out={'within_40ms':1.})
    events=emit(system,report)
    assert any(n['x']==start['x'] for n in events)
    assert all(n['x']!=continuation['x'] for n in events)
    # Even an older/stale moment group may not turn the tie back into a hit.
    continuation['exportable']=True
    report['times']=[0];report['moment_groups']=[[notes.index(continuation)]]
    assert emit(system,report)==[]


@pytest.mark.parametrize('horizontal',[False,True])
def test_untied_native_heads_and_horizontal_ink_remain_onsets(horizontal):
    gray=cv2.imread(str(FIXTURES/'olliewoods-tie-native.png'),0)
    gray[133:150,263:332]=255
    if horizontal:cv2.line(gray,(263,142),(331,142),0,2)
    notes,_=label_heads(gray,json.loads((HERE/'legends/olliewoods.json').read_text()))
    assert not any(n.get('tied_continuation') for n in notes)
    assert next(n for n in notes if abs(n['x']-338.6)<2)['exportable']


def test_tied_kick_does_not_remove_a_simultaneous_snare():
    notes=[dict(x=20,inst='kick'),dict(x=60,inst='kick',tied_continuation=True),dict(x=60,inst='snare')]
    xs,groups=moments(dict(notes=notes,staff={'spacing':11}))
    assert xs.tolist()==[20,60] and groups==[[0],[2]]


def test_white_layout_changes_and_missing_opening_clef_are_not_program_errors():
    class Capture:
        def __init__(self,*args):self.i=0
        def get(self,key):return 5 if key==5 else 100
        def set(self,key,value):self.i=int(value)
        def read(self):
            frame=np.zeros((400,800,3),np.uint8)
            offset=12 if self.i>=50 else 0
            for y in range(310+offset,359+offset,12):cv2.line(frame,(0,y),(799,y),(240,240,240),1)
            return True,frame
        def release(self):pass
    with patch('layouts.cv2.VideoCapture',Capture):geometry=detect_white_overlay('fixture')
    assert not geometry.get('opening_clef')
    assert np.ptp(np.array(geometry['sampled_native_lines'])[:,0])>=12


def test_white_geometry_discontinuity_splits_and_rejects_only_its_segment(tmp_path):
    from harvest_scroll import write
    data=np.load(FIXTURES/'daily-white-scroll-native.npz')
    frames=data['frames']
    class Capture:
        def __init__(self,*args):self.i=0
        def get(self,key):return 5 if key==5 else len(frames)
        def read(self):
            if self.i==len(frames):return False,None
            frame=np.zeros((len(frames[0])+24,frames.shape[2],3),np.uint8)
            offset=0 if self.i<10 else 16
            frame[offset:offset+len(frames[0])]=frames[self.i]
            self.i+=1;return True,frame
        def release(self):pass
    geometry=dict(layout='dailychops',foreground='white_overlay',top=0,bottom=len(frames[0])+24,
                  native_spacing=[float(data['spacing'])],motion_left=int(data['motion_left']),
                  motion_right=int(data['motion_right']),split_scroll_discontinuities=True)
    with patch('harvest_scroll.cv2.VideoCapture',Capture):scan=write('fixture',tmp_path,geometry)
    assert any('white-overlay staff geometry changes across source frames' in r['scroll_reasons'] for r in scan['records'])
    accepted=[r for r in scan['records'] if not r['scroll_reasons']]
    assert accepted and all(not(r['start']<10<r['end']) for r in accepted)
    assert all(len(r['staff_geometry']['supporting_frames'])>=5 for r in accepted)
    systems=read(tmp_path,HERE/'legends/dailychops.json')
    assert any(not s['staff']['reasons'] for s in systems)


def test_white_local_geometry_proof_accepts_continuation_without_opening(tmp_path):
    gray=cv2.imread(str(FIXTURES/'daily-white-continuation.png'),0)
    cv2.imwrite(str(tmp_path/'system_000.png'),gray)
    record=dict(system=0,start=0,end=20,page_rows=1,native_space=14,
                cursor_frames=list(range(20)),cursor_x=np.linspace(20,gray.shape[1]-20,20).tolist(),
                scroll_reasons=[],staff_geometry=dict(supporting_frames=list(range(20)),method='stable native staff'))
    scan=dict(fps=5,foreground='white_overlay',layout='dailychops',reconstructed_scroll=True,
              record_cursors=True,records=[record])
    (tmp_path/'scan.json').write_text(json.dumps(scan))
    s=read(tmp_path,HERE/'legends/dailychops.json')[0]
    assert 'percussion clef / first bar not confirmed' not in s['staff']['reasons']
    scan['records'][0]['scroll_reasons']=['scroll translation ambiguous']
    (tmp_path/'scan.json').write_text(json.dumps(scan))
    s=read(tmp_path,HERE/'legends/dailychops.json')[0]
    assert 'percussion clef / first bar not confirmed' in s['staff']['reasons']


def test_inspected_strip_continuation_requires_three_supported_consecutive_staffs(tmp_path):
    gray=cv2.imread(str(FIXTURES/'daily-white-continuation.png'),0)
    _,space=staffs(gray,prefer_line_span=True)[0]
    legend=json.loads((HERE/'legends/copydrum.json').read_text())
    records=[]
    for i in range(3):
        cv2.imwrite(str(tmp_path/f'system_{i:03d}.png'),gray)
        records.append(dict(system=i,start=i*10,end=(i+1)*10,page_rows=1,crop=[0,len(gray)],native_space=space))
    assert stable_strip_runs(tmp_path,records[:2],legend)=={}
    assert set(stable_strip_runs(tmp_path,records,legend))=={0,1,2}
    records[1]['crop'][0]=30
    assert stable_strip_runs(tmp_path,records,legend)=={}
    records[1]['crop'][0]=0
    records[1]['reading_reasons']=['large chromatic overlay obscures staff']
    assert stable_strip_runs(tmp_path,records,legend)=={}
    assert stable_strip_runs(tmp_path,records,{**legend,'score_region':None})=={}


@pytest.mark.parametrize('key,value',[('enabled',False),('scrolling_verified',False),
                                     ('white_overlay_verified',None),('scroll_clef_omitted_verified',None)])
def test_scan_fingerprint_changes_when_approval_changes(tmp_path,key,value):
    (tmp_path/'video.mp4').write_bytes(b'video')
    profile=dict(enabled=True,scrolling_verified=True,white_overlay_verified='audit',scroll_clef_omitted_verified='audit')
    before=batch.scan_signature(tmp_path,'dailychops',approval_state=batch.scan_approvals(profile))
    after=batch.scan_signature(tmp_path,'dailychops',approval_state=batch.scan_approvals({**profile,key:value}))
    assert before!=after


@pytest.mark.parametrize('mirror',[False,True])
def test_reused_scan_reapplies_current_export_gate_before_strip(tmp_path,monkeypatch,mirror):
    import strip,sanity,align,tiers,bands
    source=tmp_path/'mirror';source.mkdir()
    if mirror:(source/'.wsl-mirror').write_text('WSL')
    song=source/'song';song.mkdir();(song/'video.mp4').write_bytes(b'video')
    reuse=tmp_path/'reuse';reuse.mkdir();out=tmp_path/'out'
    profile=dict(name='fixture',layout='dailychops',enabled=True,scrolling_verified=False,rules={})
    path=tmp_path/'profile.json';path.write_text(json.dumps(profile))
    (reuse/'result.json').write_text(json.dumps({'scan_signature':'same'}))
    (reuse/'scan.json').write_text(json.dumps(dict(reconstructed_scroll=True,records=[dict(system=0,scroll_reasons=[])])))
    monkeypatch.setattr(batch,'scan_signature',lambda *a,**k:'same')
    def read_strip(folder,*args):
        scan=json.loads((folder/'scan.json').read_text())
        assert scan['records'][0]['scroll_reasons']==['scrolling reading not validated for this channel']
        (folder/'strip.json').write_text('[]');return []
    monkeypatch.setattr(strip,'read',read_strip)
    monkeypatch.setattr(sanity,'check',lambda *a:dict(flags=[]))
    monkeypatch.setattr(align,'build_truth',lambda *a,**k:None)
    cache_calls=[]
    monkeypatch.setattr(bands,'load',lambda *a,**k:cache_calls.append(k.get('cache')))
    monkeypatch.setattr(tiers,'write',lambda *a:None)
    monkeypatch.setattr(batch,'summarize',lambda *a:{})
    monkeypatch.setattr(batch,'quality_report',lambda *a:None)
    batch.worker(song,out,path,reuse)
    assert (out/'result.json').exists()
    assert cache_calls==([False] if mirror else [])


def test_scan_gate_is_idempotent_and_static_scan_is_unaffected():
    reason='scrolling reading not validated for this channel'
    data=dict(reconstructed_scroll=True,records=[dict(scroll_reasons=['motion rejected',reason])])
    batch.apply_scan_gates(data,{'scrolling_verified':False})
    batch.apply_scan_gates(data,{'scrolling_verified':False})
    assert data['records'][0]['scroll_reasons']==['motion rejected',reason]
    batch.apply_scan_gates(data,{'scrolling_verified':True})
    assert data['records'][0]['scroll_reasons']==['motion rejected']
    static=dict(records=[{}])
    assert batch.apply_scan_gates(static,{})==dict(records=[{}])
    white=dict(foreground='white_overlay',records=[{}])
    batch.apply_scan_gates(white,{})
    assert white['records'][0]['reading_reasons']==['white-overlay reading not validated for this channel']
    batch.apply_scan_gates(white,{'white_overlay_verified':True})
    assert white['records'][0]['reading_reasons']==[]


def fixture_song(root):
    song=root/'songs/fixture';song.mkdir(parents=True)
    (song/'video.mp4').write_bytes(b'video');(song/'attacks.htdemucs.npy').write_bytes(b'attacks')
    append(root/'manifest.jsonl',video='id',channel='atthedrum',slug='fixture',status='fetched')
    profile,path=batch.channel_profile('atthedrum')
    return song,read_manifest(root/'manifest.jsonl')['id'],profile,path


def test_process_refuses_mirror_even_when_completed_result_exists(tmp_path):
    song,row,profile,path=fixture_song(tmp_path)
    out=song/'truth';out.mkdir()
    batch.dump(out/'result.json',dict(complete=True,signature=batch.signature(song,profile),
                                   scoring_systems=1,scoring_notes=2))
    result_before=(out/'result.json').read_bytes()
    (tmp_path/'.wsl-mirror').write_text('WSL')
    before=(tmp_path/'manifest.jsonl').read_bytes()
    with pytest.raises(batch.MirrorWriteError):batch.process(tmp_path,row,profile,path,1)
    assert (tmp_path/'manifest.jsonl').read_bytes()==before
    assert not (song/'.truth.lock').exists()
    assert (out/'result.json').read_bytes()==result_before
    with pytest.raises(batch.MirrorWriteError):batch.check_worker_destination(song,song/'truth')
    batch.check_worker_destination(song,tmp_path.parent/'scratch')


@pytest.mark.parametrize('flag',[[],['--report'],['--watch']])
def test_main_refuses_mirror_before_writing(tmp_path,monkeypatch,flag):
    (tmp_path/'.wsl-mirror').write_text('WSL')
    monkeypatch.setattr(sys,'argv',['batch.py','--root',str(tmp_path),*flag])
    with pytest.raises(SystemExit) as e:batch.main()
    assert e.value.code==2
    assert list(tmp_path.iterdir())==[tmp_path/'.wsl-mirror']


def test_explicit_scratch_attempt_on_mirror_preserves_source_and_journal(tmp_path,monkeypatch):
    root=tmp_path/'mirror';song,row,profile,path=fixture_song(root)
    (root/'.wsl-mirror').write_text('WSL')
    before={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(batch.subprocess,'run',lambda *a,**k:(_ for _ in ()).throw(KeyError('reader bug')))
    scratch=tmp_path/'scratch'
    result=batch.process(root,row,profile,path,1,scratch)
    assert not result['complete']
    assert read_manifest(scratch/'manifest.jsonl')['id']['status']=='unreadable'
    assert json.loads((scratch/'fixture/truth/truth.json').read_text())==[]
    assert json.loads((scratch/'fixture/truth/failure.json').read_text())['error_type']=='KeyError'
    assert before=={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(batch.MirrorWriteError):batch.process(root,row,profile,path,1,root/'scratch')


def test_scratch_report_on_mirror_writes_only_to_explicit_output(tmp_path,monkeypatch):
    root=tmp_path/'mirror';fixture_song(root)
    (root/'.wsl-mirror').write_text('WSL');scratch=tmp_path/'scratch'
    before={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(sys,'argv',['batch.py','--root',str(root),'--scratch-out',str(scratch),'--report'])
    batch.main()
    assert (scratch/'CHANNEL_RESULTS.md').exists() and (scratch/'channel-results.json').exists()
    assert before=={p.relative_to(root):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_process_setup_program_error_is_recorded_unreadable(tmp_path,monkeypatch):
    song,row,profile,path=fixture_song(tmp_path)
    monkeypatch.setattr(batch,'signature',lambda *a:(_ for _ in ()).throw(IndexError('setup bug')))
    assert batch.process(tmp_path,row,profile,path,1) is None
    assert read_manifest(tmp_path/'manifest.jsonl')['id']['truth_reason']=='IndexError: setup bug'


def test_one_program_error_never_aborts_later_video(tmp_path,monkeypatch):
    fixture_song(tmp_path)
    append(tmp_path/'manifest.jsonl',video='next',channel='atthedrum',slug='next',status='fetched')
    calls=[]
    def process(root,row,*args):
        calls.append(row['video'])
        if row['video']=='id':raise TypeError('reader program error')
        append(root/'manifest.jsonl',video=row['video'],status='read')
        return {'complete':True}
    monkeypatch.setattr(batch,'process',process)
    monkeypatch.setattr(batch,'report',lambda *a:None)
    monkeypatch.setattr(sys,'argv',['batch.py','--root',str(tmp_path),'--min-free-gb','0'])
    batch.main()
    rows=read_manifest(tmp_path/'manifest.jsonl')
    assert calls==['id','next']
    assert rows['id']['status']=='unreadable' and 'TypeError' in rows['id']['truth_reason']
    assert rows['next']['status']=='read'


def test_mark_template_variants_share_one_mark_name():
    # atthedrum draws its open 'o' in two glyphs; both templates must read as 'open'.
    from strip import marks
    template=cv2.imread(str(HERE/'harvest_marks/atthedrum/open_round.png'),0)
    gray=np.full((80,80),245,np.uint8)
    gray[30:30+template.shape[0],40:40+template.shape[1]]=template
    found=marks(gray,11,'harvest_marks/atthedrum')
    assert found and {m['mark'] for m in found}=={'open'}
