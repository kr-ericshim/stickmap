import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from tiers import scoring_reasons,write


SCRATCH=Path(tempfile.gettempdir())/'stickmap-truth-video-tests';SCRATCH.mkdir(parents=True,exist_ok=True)

class Tiers(unittest.TestCase):
    def row(self):
        return dict(reasons=['held-out agreement below 55% / 80%'],times=[1,2,3,4,5,6],moments=6,
                    strictly_monotone=True,held_out_gain_20ms=.2,held_out_gain_40ms=.2,
                    after=dict(within_20ms=.95,within_40ms=.95),
                    held_out=dict(within_20ms=.4,within_40ms=.92),mode='cursor')

    def test_high_40ms_can_pass_without_20ms(self):
        self.assertEqual(scoring_reasons(self.row()),[])

    def test_staff_rejection_survives(self):
        r=self.row();r['reasons'].append('native staff spacing below 8 px')
        self.assertIn('native staff spacing below 8 px',scoring_reasons(r))

    def test_dense_control_and_nonmonotone_never_pass(self):
        r=self.row();r.update(held_out_gain_20ms=.1,held_out_gain_40ms=.1,strictly_monotone=False)
        self.assertGreaterEqual(len(scoring_reasons(r)),2)

    def test_no_cursor_requires_90(self):
        r=self.row();r['mode']='interval';r['held_out']['within_40ms']=.88
        self.assertTrue(scoring_reasons(r));r['mode']='cursor';self.assertFalse(scoring_reasons(r))

    def test_real_strict_files_unchanged_and_extras_disjoint(self):
        source=Path('truth/ken-jezer')
        with tempfile.TemporaryDirectory(dir=SCRATCH) as name:
            out=Path(name)
            for file in ['strip.json','truth.json','alignment.json']:
                (out/file).write_bytes((source/file).read_bytes())
            before={f:(out/f).read_bytes() for f in ['truth.json','alignment.json']}
            report=write(out)
            for f,b in before.items():self.assertEqual((out/f).read_bytes(),b)
            strict={r['system'] for r in report['systems'] if r['usable']}
            extra=json.loads((out/'truth.additional.json').read_text())
            self.assertTrue(extra);self.assertTrue(all(e['system'] not in strict for e in extra))


if __name__=='__main__':unittest.main()
