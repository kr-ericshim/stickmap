import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parents[1]/'harvest'))
from manifest_io import read,append
from layouts import cursor,ink,channel_profile
from color_timing import transition
from heads import staffs,staff_checks
from strip import label_heads
from batch import song_path,lock
from harvest_scroll import shift
import harvest_scan


SCRATCH=Path(tempfile.gettempdir())/'groovemap-truth-video-tests';SCRATCH.mkdir(parents=True,exist_ok=True)

class Harvest(unittest.TestCase):
    def test_manifest_partial_last_line_and_updates(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as name:
            p=Path(name)/'manifest.jsonl';append(p,video='a',status='fetched',channel='x');append(p,video='a',status='read')
            with p.open('ab') as f:f.write(b'{"video":')
            self.assertEqual(read(p)['a']['status'],'read');self.assertEqual(read(p)['a']['channel'],'x')
            with self.assertRaises(ValueError):append(p,video='b',status='fetched')

    def test_manifest_bad_committed_line_is_visible_error(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as name:
            p=Path(name)/'manifest.jsonl';p.write_text('broken\n')
            with self.assertRaises(ValueError):read(p)

    def test_path_traversal_and_unknown_identity(self):
        for slug in ('../outside','/tmp/x','..',''):
            with self.assertRaises(ValueError):song_path(Path('truth'),{'slug':slug})
        profile,path=channel_profile('../../kbdrums')
        self.assertFalse(profile['enabled']);self.assertIsNone(path)

    def test_colored_cursor_and_yellow_bar(self):
        for color in ((235,180,150),(80,220,80),(180,130,220),(30,140,240)):
            f=np.full((90,300,3),255,np.uint8);f[5:80,125:130]=color
            box=cursor(f);self.assertIsNotNone(box);self.assertAlmostEqual(box[0],127,delta=1)
        f=np.full((90,300,3),255,np.uint8);f[:,50:240]=(150,245,250)
        self.assertIsNone(cursor(f))

    def test_color_transition_requires_baseline_persistence_and_uniqueness(self):
        r=transition([0,0,1,1,0],list(range(5)),30)
        self.assertAlmostEqual(r['time'],.05);self.assertAlmostEqual(r['uncertainty_seconds'],1/30)
        for values in ([1,1,0,0],[0,0,1,0],[0,0,1,1,0,0,1,1]):
            self.assertIsNone(transition(values,list(range(len(values))),30))
        self.assertIsNone(transition([0,0,1,1],list(range(4)),30,[True,False,True,True]))

    def test_repeated_score_is_split_on_cursor_reset(self):
        base=cv2.imread(str(HERE/'fixtures/atthedrum.png'))
        frames=[]
        for repeat in range(2):
            for x in np.linspace(70,410,20).astype(int):
                f=base.copy();f[40:150,max(0,x-2):x+3]=(240,180,140);frames.append(f)
        class Capture:
            def __init__(self,*args):self.i=0
            def get(self,key):return 10 if key==cv2.CAP_PROP_FPS else len(frames)
            def read(self):
                if self.i==len(frames):return False,None
                f=frames[self.i];self.i+=1;return True,f
            def release(self):pass
        with tempfile.TemporaryDirectory(dir=SCRATCH) as name:
            geometry=dict(layout='kr_strip',top=0,bottom=len(base),left=0,right=base.shape[1],native_spacing=[11])
            with patch('harvest_scan.detect',return_value=geometry),patch('harvest_scan.cv2.VideoCapture',Capture):
                data=harvest_scan.write('fixture.mp4',name)
            starts={r['start'] for r in data['records']}
            self.assertIn(0,starts);self.assertIn(20,starts)

    def test_scroll_translation_and_ambiguous_flat_image(self):
        rng=np.random.default_rng(1)
        a=(rng.random((60,400))>.9).astype(np.uint8)*255
        b=cv2.warpAffine(a,np.float32([[1,0,-12],[0,1,0]]),(400,60))
        r=shift(a,b);self.assertIsNotNone(r);self.assertEqual(r['dx'],-12)
        self.assertIsNone(shift(np.zeros_like(a),np.zeros_like(a)))

    def test_real_native_crops_and_small_spacing_gate(self):
        for name in ('atthedrum','drumscribe','moyin','dailychops','olliewoods'):
            g=ink(cv2.imread(str(HERE/'fixtures'/f'{name}.png')))
            found=staffs(g,min_space=6)
            self.assertTrue(found,name)
            if name=='drumscribe':
                ls,sp=found[-1];self.assertLess(sp,8)
                self.assertIn('native staff spacing below 8 px',staff_checks(g,ls,sp,[])['reasons'])

    def test_disabled_channel_and_unverified_mapping_do_not_export(self):
        g=ink(cv2.imread(str(HERE/'fixtures/atthedrum.png')))
        profile,_=channel_profile('atthedrum');profile['enabled']=False
        notes,checks=label_heads(g,profile)
        self.assertTrue(any(r.startswith('channel disabled:') for r in checks['reasons']))
        profile['rules']={};notes,_=label_heads(g,profile)
        self.assertTrue(all(not n['exportable'] for n in notes))

    def test_open_mark_is_distinct_from_circled_crash(self):
        g=ink(cv2.imread(str(HERE/'fixtures/yeol-open-crash.png')))
        profile,_=channel_profile('yeol');notes,_=label_heads(g,profile)
        self.assertTrue(any(n['shape']=='circled' and n['inst']=='crash' for n in notes))
        self.assertTrue(any(n['shape']=='cross' and n.get('articulation')=='open' and n['exportable'] for n in notes))
        self.assertFalse(any(n['shape']=='circled' and n.get('articulation')=='open' for n in notes))

    def test_printed_key_overrides_generic_hat_ride_positions(self):
        profile,_=channel_profile('mydrumsheet')
        self.assertEqual(profile['rules']['cross:8']['inst'],'hihat')
        self.assertEqual(profile['rules']['cross:9']['inst'],'ride')

    def test_channel_ghost_parentheses_and_regular_snare_control(self):
        g=cv2.imread(str(HERE/'fixtures/mydrumsheet-ghost.png'),0)
        profile,_=channel_profile('mydrumsheet');notes,_=label_heads(g,profile)
        snare=[n for n in notes if n['inst']=='snare']
        self.assertTrue(any(abs(n['x']-(235.6-110))<2 and n['ghost'] for n in snare))
        self.assertTrue(any(abs(n['x']-(289-110))<2 and not n['ghost'] for n in snare))


if __name__=='__main__':unittest.main()
