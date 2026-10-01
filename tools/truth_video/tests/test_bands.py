"""Band-limited onset evidence on synthesized drums: kicks read as low onsets, snares as mid onsets."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from scipy.signal import butter, sosfilt
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import bands
from align import align_system, build_truth, phase_established, PHASE

ROOT=HERE.parents[2]
SR=44100


def groove(beat=.4, bars=8, start=2.):
    """Eighth-note hats, kick on 1 and 3, snare on 2 and 4. Returns (times, instrument sets)."""
    times=start+np.arange(bars*8)*beat/2
    insts=[]
    for i in range(len(times)):
        q={'hihat'}
        if i%8 in (0,4):q.add('kick')
        if i%8 in (2,6):q.add('snare')
        insts.append(q)
    return times,insts


def continuing_attacks(times, beat=.4, bars=2):
    """The groove keeps going before and after the notated bars: no edge breaks a rhythmic tie."""
    return np.arange(times[0]-bars*4*beat, times[-1]+bars*4*beat+1e-9, beat/2)


def synthesize(times, insts, seconds=None):
    rng=np.random.default_rng(0)
    seconds=seconds or float(times[-1]+2)
    y=np.zeros(int(SR*seconds),np.float32)
    n=int(SR*.5);t=np.arange(n)/SR
    kick=(.8*np.sin(2*np.pi*55*t)*np.exp(-t/.12)).astype(np.float32)
    snare=sosfilt(butter(4,[300,3000],btype='band',fs=SR,output='sos'),rng.standard_normal(n))*np.exp(-t/.08)*.5
    hat=sosfilt(butter(4,6000,btype='high',fs=SR,output='sos'),rng.standard_normal(n))*np.exp(-t/.03)*.3
    for time,q in zip(times,insts):
        i=int(time*SR)
        for name,wave in (('kick',kick),('snare',snare),('hihat',hat)):
            if name in q:y[i:i+n]+=wave.astype(np.float32)
    return y


def synthetic_bands(times, insts, seconds=30.):
    """Ideal band flux: unit low peaks at kicks, unit mid peaks at snares, hat leak in mid, hats in high."""
    t=np.arange(0,seconds,bands.HOP_SECONDS).astype(np.float32)
    out={'t':t,'low':np.zeros_like(t),'mid':np.zeros_like(t),'high':np.zeros_like(t)}
    for time,q in zip(times,insts):
        i=int(round(time/bands.HOP_SECONDS))
        out['high'][i]=1
        if 'kick' in q:out['low'][i]=1;out['mid'][i]=max(out['mid'][i],.2)
        if 'snare' in q:out['mid'][i]=1;out['low'][i]=max(out['low'][i],.15)
        if q=={'hihat'}:out['mid'][i]=max(out['mid'][i],.3)
    return bands.with_peaks(out)


class BandEvidence(unittest.TestCase):
    def test_synthesized_drums_put_kicks_low_and_snares_mid(self):
        times,insts=groove()
        ob=bands.with_peaks(bands.onset_strength(synthesize(times,insts),SR))
        here=bands.agreement(ob,times,insts)
        self.assertGreaterEqual(here['kick'],.9);self.assertGreaterEqual(here['snare'],.9)
        beat=bands.agreement(ob,times+.4,insts)
        self.assertLessEqual(beat['pooled'],.3)
        eighth=bands.agreement(ob,times+.2,insts)
        self.assertLessEqual(eighth['pooled'],.3)
        check=bands.phase_check(ob,times,insts,attacks=times)
        self.assertGreaterEqual(check['margin'],.5)
        self.assertGreater(check['alias_candidates'],0)

    def test_shift_between_attacks_is_not_an_alias_candidate(self):
        times,insts=groove()
        ob=synthetic_bands(times,insts)
        check=bands.phase_check(ob,times,insts,attacks=continuing_attacks(times))
        # Only shifts within the attack tolerance of a whole eighth (.2 s) keep the moments on attacks.
        self.assertTrue(0<check['alias_candidates']<=4*9,check['alias_candidates'])
        self.assertLess(check['best_shift_pooled'],.2)
        self.assertLess(abs(check['best_shift']/.2-round(check['best_shift']/.2)),.25)
        none=bands.phase_check(ob,times,insts,attacks=times[::8])   # attacks only on the downbeats
        self.assertEqual(none['alias_candidates'],0);self.assertIsNone(none['best_shift'])

    def test_cache_round_trip_and_missing_stem(self):
        times,insts=groove(bars=2)
        with tempfile.TemporaryDirectory(dir=ROOT/'truth') as temp:
            song=Path(temp)
            self.assertIsNone(bands.load(song))
            import soundfile as sf
            sf.write(str(song/'drums.htdemucs.wav'),np.stack([synthesize(times,insts,8)]*2,1),SR)
            first=bands.load(song)
            self.assertTrue((song/bands.CACHE).exists())
            second=bands.load(song)
            np.testing.assert_allclose(first['low'],second['low'])
            self.assertEqual(second['tolerance'],bands.TOLERANCE)


class PhaseAliasAlignment(unittest.TestCase):
    def system(self, times, insts, lag=.3, number=0):
        x=100+np.arange(len(times))*20.
        notes=[]
        for xx,q in zip(x,insts):
            for inst in sorted(q):notes.append(dict(x=float(xx),inst=inst))
        return dict(system=number,interval=[float(times[0]+lag-.3),float(times[-1]+lag+.3)],cursor_x=x.tolist(),
                    cursor_t=(times+lag).tolist(),cursor_coverage=.9,staff={'spacing':11,'reasons':[]},notes=notes)

    def test_band_evidence_breaks_the_rhythmic_alias(self):
        times,insts=groove()
        s=self.system(times,insts);ob=synthetic_bands(times,insts);attacks=continuing_attacks(times)
        # Attacks on every eighth: a shift by an eighth or a beat agrees as well as the truth.
        _,blind=align_system(s,attacks)
        self.assertGreater(abs(blind['offset_seconds']+.3),.05)
        events,r=align_system(s,attacks,bands=ob)
        self.assertTrue(r['usable'],r['reasons'])
        self.assertAlmostEqual(r['offset_seconds'],-.3,delta=.02)
        self.assertGreaterEqual(r['phase']['margin'],PHASE['min_gain'])
        self.assertTrue(phase_established(r['phase']))
        self.assertTrue(all(abs(e['time']-times[np.argmin(abs(times-e['time']))])<.02 for e in events))

    def test_forced_alias_is_rejected_by_kick_snare_bands(self):
        times,insts=groove()
        s=self.system(times,insts);ob=synthetic_bands(times,insts)
        _,r=align_system(s,continuing_attacks(times),seed=.1,bands=ob)
        self.assertFalse(r['usable'])
        self.assertTrue(any(reason.startswith('phase alias') for reason in r['reasons']),r['reasons'])
        self.assertLess(r['phase']['pooled'],PHASE['min_agreement'])

    def test_song_level_phase_backs_short_systems_but_not_contradicting_ones(self):
        times,insts=groove();ob=synthetic_bands(times,insts);attacks=continuing_attacks(times)
        song_phase=bands.phase_check(ob,times,insts,attacks=attacks)
        self.assertTrue(phase_established(song_phase))
        short_t=times[:12];short_q=[{'hihat'}|({'kick'} if i in (0,4,8) else set()) for i in range(12)]
        s=self.system(short_t,short_q,number=1)
        _,alone=align_system(s,attacks,seed=-.3,bands=ob)
        self.assertFalse(alone['usable']);self.assertTrue(any('fewer than 6 notated kick/snare' in q for q in alone['reasons']))
        _,backed=align_system(s,attacks,seed=-.3,bands=ob,song_phase=song_phase)
        self.assertTrue(backed['usable'],backed['reasons'])
        # A system whose own kicks/snares prefer a shift is not rescued by the song.
        _,wrong=align_system(self.system(times,insts),attacks,seed=.1,bands=ob,song_phase=song_phase)
        self.assertFalse(wrong['usable'])

    def test_build_truth_without_stem_exports_nothing(self):
        times,insts=groove()
        with tempfile.TemporaryDirectory(dir=ROOT/'truth') as temp:
            song=Path(temp)/'song';folder=Path(temp)/'truth';song.mkdir();folder.mkdir()
            np.save(song/'attacks.htdemucs.npy',continuing_attacks(times))
            (folder/'strip.json').write_text(json.dumps([self.system(times,insts)]))
            truth,report=build_truth(song,folder)
            self.assertEqual(truth,[])
            self.assertIn('instrument-band phase check unavailable (no drum stem)',report['systems'][0]['reasons'])
            self.assertIsNone(report['band_evidence'])


if __name__=='__main__':unittest.main()
