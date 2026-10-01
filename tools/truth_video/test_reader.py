"""Regression checks for failures that can quietly change every timestamp or label."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import cv2
import numpy as np
from align import align_system, match, moments, prior, quality
from heads import heads, staffs, staff_checks
from scan import cursor_box, write
from strip import read
from evaluate import family, score

ROOT=Path(__file__).resolve().parents[2]


SCRATCH=Path(tempfile.gettempdir())/'groovemap-truth-video-tests';SCRATCH.mkdir(parents=True,exist_ok=True)

class ReaderTests(unittest.TestCase):
    def test_one_to_one_attacks(self):
        self.assertEqual(len(match([1,1.01],[1.005],.02)),1)

    def test_box_arrival_is_not_interpolated(self):
        s=dict(cursor_x=[100,200,300,400],cursor_t=[1,2,3,4],cursor_mode='box',
               cursor_coverage=.9,staff={'spacing':11},interval=[1,4])
        t,mode=prior(s,np.array([103,198,302,398]))
        np.testing.assert_allclose(t,[1,2,3,4],atol=.025)
        self.assertEqual(mode,'cursor')

    def test_chords_are_one_moment(self):
        x,groups=moments(dict(staff={'spacing':11},notes=[{'x':100},{'x':102},{'x':120}]))
        self.assertEqual(groups,[[0,1],[2]])

    def test_misaligned_dense_attacks_do_not_prove_truth(self):
        x=np.arange(25)*20.+100
        s=dict(system=0,interval=[1,4],cursor_x=x.tolist(),cursor_t=np.linspace(1,4,25).tolist(),
               cursor_coverage=.9,staff={'spacing':11,'reasons':[]},notes=[dict(x=v,inst='snare') for v in x])
        events,r=align_system(s,np.arange(0,5,.0125),0)
        self.assertFalse(r['usable'])
        self.assertTrue(any('dense attacks' in reason for reason in r['reasons']))

    def test_known_nonlinear_warp(self):
        rng=np.random.default_rng(3)
        t=np.r_[0,np.cumsum(rng.uniform(.1,.25,35))]+1
        x=np.arange(len(t))*25.+100
        s=dict(system=0,interval=[.9,float(t[-1]+.1)],cursor_x=x.tolist(),cursor_t=t.tolist(),
               cursor_coverage=.9,staff={'spacing':11,'reasons':[]},notes=[dict(x=float(v),inst='snare') for v in x])
        attacks=t+.035+.008*np.sin(t)
        events,r=align_system(s,attacks,.035)
        self.assertTrue(r['usable'],r['reasons'])
        self.assertGreater(r['held_out']['within_20ms'],.9)
        self.assertTrue(np.all(np.diff(r['times'])>0))

    def test_no_cursor_uses_interval(self):
        s=dict(cursor_x=[],cursor_t=[],interval=[2,6])
        t,mode=prior(s,np.array([100,200,300]))
        self.assertEqual(mode,'interval')
        np.testing.assert_allclose(t,[2,3.95,5.9])

    def test_family_scoring_ignores_rejected_gap(self):
        truth=[dict(time=1.,inst='ride',ghost=False),dict(time=10.,inst='hihat',ghost=False)]
        hits=[SimpleNamespace(time=1.,inst='cymbal'),SimpleNamespace(time=5.,inst='kick'),SimpleNamespace(time=10.,inst='hihat')]
        result=score(truth,hits,intervals=[[.9,1.1],[9.9,10.1]])
        self.assertEqual(len(result['kick'][2]),0)
        self.assertEqual(len(result['cymbal'][0]),1)
        self.assertEqual(family('ride_bell'),'cymbal')
        self.assertEqual(family('side_stick'),'snare')
        self.assertEqual(score([],[]),{})

    def test_broken_staff_is_not_a_staff(self):
        im=np.full((150,800),255,np.uint8)
        for y in (60,71,82,95,106):cv2.line(im,(20,y),(780,y),0,1)
        lines,space=staffs(im)[0]
        self.assertIn('uneven staff lines',staff_checks(im,lines,space,[])['reasons'])

    def test_page_and_red_cursor(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as temp:
            folder=Path(temp);video=folder/'page.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),30,(800,350))
            for i in range(90):
                im=np.full((350,800,3),255,np.uint8)
                for top in (70,230):
                    for k in range(5):cv2.line(im,(30,top+k*11),(770,top+k*11),(0,0,0),1)
                    for x in (42,49):cv2.rectangle(im,(x,top+11),(x+3,top+33),(0,0,0),-1)
                    for x in range(120,750,70):cv2.ellipse(im,(x,top+38),(7,5),-20,0,360,(0,0,0),-1)
                row=0 if i<45 else 1;x=100+(i%45)*14;top=70+row*160
                cv2.line(im,(x,top-25),(x,top+65),(70,75,160),2)
                writer.write(im)
            writer.release()
            d=write(video,folder,0,350)
            self.assertEqual(len(d['records']),2)
            self.assertTrue(all(r['page_rows']==2 for r in d['records']))
            self.assertEqual([len(r['cursor_frames']) for r in d['records']],[45,45])
            ss=read(folder)
            self.assertLessEqual(ss[0]['interval'][1],ss[1]['interval'][0])
            self.assertTrue(all((folder/'check'/f"system_{s['system']:03d}.png").exists() for s in ss))

    def test_no_cursor_can_pass_when_x_and_interval_identify_attacks(self):
        rng=np.random.default_rng(7)
        times=np.r_[0,np.cumsum(rng.uniform(.12,.29,25))]+2
        x=100+(times-times[0])*170
        s=dict(system=0,interval=[2,float(times[-1]+.1)],cursor_x=[],cursor_t=[],
               staff={'spacing':11,'reasons':[]},notes=[dict(x=float(v),inst='kick') for v in x])
        events,r=align_system(s,times+.025)
        self.assertTrue(r['usable'],r['reasons'])
        self.assertEqual(r['mode'],'interval')

    def test_parentheses_and_grace_regressions(self):
        samples=json.loads((ROOT/'truth/audit/flag_samples.json').read_text())
        expected={1:True,2:True,4:True,5:False,7:False,25:True,26:False,29:False,30:True}
        from strip import label_heads
        legend=json.loads((Path(__file__).parent/'legends/kbdrums.json').read_text())
        for index,wanted in expected.items():
            q=samples[index-1]
            im=cv2.imread(str(ROOT/'truth'/q['video']/f"system_{q['system']:03d}.png"),0)
            notes,_=label_heads(im,legend)
            nearest=min(notes,key=lambda n:abs(n['x']-q['predicted']['x'])+abs(n['y']-q['predicted']['y']))
            self.assertEqual(nearest[q['flag']],wanted,(index,q['flag']))

    def test_real_cj_upscaled_path_stays_unusable(self):
        folder=ROOT/'truth/cj-thompson'
        s=json.loads((folder/'strip.json').read_text())
        self.assertTrue(any(q['row']>0 for q in s))
        q=next(q for q in s if q['notes'])
        im=cv2.imread(str(folder/f"system_{q['system']:03d}.png"),0)
        found,lines,space=heads(cv2.resize(im,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC))
        self.assertTrue(found)
        self.assertGreaterEqual(space,8)
        self.assertTrue(all(any('low-resolution' in r or 'staff' in r or 'spacing' in r for r in q['staff']['reasons']) for q in s))


if __name__=='__main__':unittest.main()
