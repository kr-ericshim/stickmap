"""Safety contracts for automatic video anchors; no audio or Gemini calls."""
import copy
import sys
import unittest
import numpy as np
from common import ROOT, flatten
from hybrid import merge
from auto_timing import convert_auto, video_times
from native_cv import measured_lines
from test_hybrid import answer_fixture, cv_fixture


def inputs():
    cv=cv_fixture()
    cv['staff']['reasons']=[]
    cv['staff']['clef_confirmed']=True
    cv['notes'][0].update(exportable=True,confidence=.95,shape_confidence=.95)
    reading=merge(answer_fixture(),cv,'fixture_0')
    meta=dict(system=0,system_id='fixture_0',record={},automatic_anchor=dict(interval=[1.,3.],
              cursor_x=[0.,20.,40.,60.],cursor_t=[1.,1.5,2.,2.5],cursor_mode='sweep',owned_x=[0.,60.]))
    legend={'rules':{'filled:5':{'inst':'snare','verified':True}}}
    return reading,meta,cv,legend


class AutoTimingTests(unittest.TestCase):
    def test_cv_approval_is_required_even_for_a_verified_legend(self):
        reading,meta,cv,legend=inputs()
        self.assertTrue(convert_auto(reading,meta,cv,legend)['notes'][0]['exportable'])
        cv['notes'][0]['exportable']=False
        self.assertFalse(convert_auto(reading,meta,cv,legend)['notes'][0]['exportable'])

    def test_gemini_only_head_remains_unknown(self):
        reading,meta,cv,legend=inputs()
        head=reading['bars'][0]['events'][0]['heads'][0]
        head.update(id='miss-new',source='gemini-only-miss',instrument='snare',instrument_verified=True)
        note=convert_auto(reading,meta,cv,legend)['notes'][0]
        self.assertEqual(note['inst'],'unknown');self.assertFalse(note['exportable'])

    def test_unresolved_owned_head_rejects_but_unvisited_context_does_not(self):
        reading,meta,cv,legend=inputs()
        reading['bars'][0]['events'][0]['onset']=None
        result=convert_auto(reading,meta,cv,legend)
        self.assertTrue(any('unresolved heads' in r for r in result['staff']['reasons']))
        meta['automatic_anchor']['owned_x']=[30.,60.]
        result=convert_auto(reading,meta,cv,legend)
        self.assertFalse(any('unresolved heads' in r for r in result['staff']['reasons']))

    def test_tuplet_rhythm_and_video_bar_origin(self):
        reading,meta,cv,legend=inputs()
        proto=reading['bars'][0]['events'][0]
        reading['bars'][0]['events']=[];cv['notes']=[]
        for i in range(8):
            e=copy.deepcopy(proto);e.update(onset=f'{i}/24',duration='1/24')
            h=e['heads'][0];h.update(id=f'cv-{i+1}',x=float(i*8),cv_label=i+1)
            h['tuplet']={'ratio':'6:4','group':'b0-t0'}
            reading['bars'][0]['events'].append(e)
            cv['notes'].append(dict(label=i+1,x=i*8,y=25,pos=5,shape='filled',inst='snare',verified=True,exportable=True))
        a=meta['automatic_anchor'];a.update(cursor_x=[0.,16.,32.,48.,64.],cursor_t=[10.,10.2,10.4,10.6,10.8],owned_x=[0.,64.],interval=[10.,10.8])
        r=convert_auto(reading,meta,cv,legend)
        self.assertTrue(np.allclose(r['rhythm_t'],10+np.arange(8)*.1))
        self.assertEqual(r['notes'][1]['onset_q'],'1/6')
        self.assertFalse(r['time_anchor_evidence']['audio_used'])
        self.assertFalse(r['notes'][0]['head_independently_reviewed'])

    def test_ties_do_not_create_attacks(self):
        reading,meta,cv,legend=inputs()
        reading['bars'][0]['events'][0]['heads'][0]['tie_continuation']=True
        r=convert_auto(reading,meta,cv,legend)
        self.assertEqual(r['rhythm_x'],[])
        self.assertFalse(r['notes'][0]['exportable'])

    def test_box_cursor_uses_arrival_knot(self):
        a={'cursor_x':[10,30,50,70],'cursor_t':[1,2,3,4],'cursor_mode':'box'}
        self.assertEqual(video_times([34],a,10)[0],2)

    def test_geometry_requires_observed_lines(self):
        sys.path.insert(0,str(ROOT/'tools/truth_video'))
        im=np.full((100,300),255,np.uint8)
        expected=[20.,30.,40.,50.,60.]
        with self.assertRaises(ValueError):measured_lines(im,expected)
        for y in expected:im[int(y),:]=60
        ls,sp=measured_lines(im,expected)
        self.assertTrue(np.allclose(ls,expected));self.assertAlmostEqual(sp,10)
        im[40]=255
        with self.assertRaises(ValueError):measured_lines(im,expected)

    def test_unique_native_staff_survives_vertical_scene_change(self):
        sys.path.insert(0,str(ROOT/'tools/truth_video'))
        im=np.full((150,300),255,np.uint8)
        actual=[30.,46.,62.,78.,94.]
        for y in actual:im[int(y),:]=60
        ls,sp=measured_lines(im,np.asarray(actual)+18)
        self.assertTrue(np.allclose(ls,actual));self.assertAlmostEqual(sp,16)

    def test_acceptance_counts_unmatched_and_never_calls_empty_perfect(self):
        sys.path.insert(0,str(ROOT/'tools/truth_video'))
        from solos_gate import agreement
        ref=[dict(time=1.,inst='kick'),dict(time=2.,inst='snare')]
        est=[dict(time=1.01,inst='kick',system=0),dict(time=2.,inst='kick',system=0)]
        a=agreement(est,ref)
        self.assertEqual(a['agreement'],.5);self.assertFalse(a['passed'])
        self.assertIsNone(agreement([],ref)['agreement']);self.assertFalse(agreement([],ref)['passed'])


if __name__=='__main__':unittest.main()
