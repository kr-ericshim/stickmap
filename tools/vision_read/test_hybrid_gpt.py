"""Codex transport, frozen musical contract, retry and resume checks (offline)."""
import argparse
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode = True
from common import DATA, dump, flatten
from hybrid import instructions
from hybrid_gpt import Backoff, attached_prompt, one, rate_limited, retry_seconds
from test_hybrid import answer_fixture, cv_fixture


class CodexTests(unittest.TestCase):
    def test_musical_prompt_is_exactly_frozen(self):
        meta = dict(system_id='fixture_0', job_dir='/tmp/fixture')
        original = instructions(meta, cv_fixture())
        attached = attached_prompt(meta, cv_fixture())
        marker = '\nThe clean original'
        self.assertEqual(original[original.index(marker):], attached[attached.index(marker):])

    def test_rate_detection_and_retry_hint(self):
        self.assertTrue(rate_limited('rate_limit_exceeded: Retry-After: 2 minutes'))
        self.assertFalse(rate_limited('head 4290 x=429.25'))
        self.assertEqual(retry_seconds('Retry-After: 2 minutes', 30), 120)
        self.assertEqual(retry_seconds('try again in 500 ms', .1), .5)

    def test_real_subprocess_transport_rate_retry_and_resume(self):
        base = DATA/'hybrid-test-artifacts'; base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            root = Path(temp); source = root/'input/jobs/fixture_0'; source.mkdir(parents=True)
            cv = cv_fixture(); cv['wall_seconds'] = .1
            for name, value in [('cv.json', cv), ('job.json', dict(system_id='fixture_0', video='fixture', images=[])),
                                ('legend.json', dict(rules={})), ('annotation.json', {})]:
                dump(source/name, value)
            for name in ('original.png', 'annotated.png'):
                (source/name).write_bytes(b'fake image for transport test')
            fake = root/'fake-codex'
            fake.write_text('#!'+sys.executable+'\n'+
                'import sys,pathlib,json\nargs=sys.argv[1:]\n'+
                "assert args[:4] == ['exec','-m','gpt-6-luna','-c']\n"+
                "assert args[4] == 'model_reasoning_effort=xhigh'\n"+
                "assert args[args.index('-s')+1] == 'read-only'\n"+
                "assert pathlib.Path.cwd().name == 'fixture_0'\n"+
                "assert [args[i+1] for i,a in enumerate(args) if a=='-i'] == ['original.png','annotated.png']\n"+
                "assert pathlib.Path('original.png').exists() and pathlib.Path('annotated.png').exists()\n"+
                "counter=pathlib.Path('counter'); first=not counter.exists(); counter.write_text('1')\n"+
                "if first:\n print('rate_limit_exceeded; Retry-After: 0.01 seconds',file=sys.stderr,flush=True);sys.exit(1)\n"+
                "pathlib.Path(args[args.index('-o')+1]).write_text("+repr(json.dumps(answer_fixture()))+")\n")
            fake.chmod(0o700)
            args = argparse.Namespace(resume=False, codex=str(fake), model='gpt-6-luna', effort='xhigh',
                                      timeout=3, rate_retries=1, backoff_seconds=.01)
            folder = root/'run'; folder.mkdir()
            with patch('hybrid_gpt.HYBRID', root/'input'):
                result = one('fixture_0', args, folder, threading.Event(), Backoff())
                self.assertEqual(result['status'], 'ok')
                self.assertEqual([r['status'] for r in result['attempts']], ['rate_limited', 'ok'])
                head = flatten(json.loads((folder/'fixture_0/reading.json').read_text()))[0]
                self.assertEqual((head['instrument'], head['onset'], head['rhythm_source']), ('snare', '3/16', 'gpt-numbered'))
                args.resume = True; fake.unlink()
                self.assertEqual(one('fixture_0', args, folder, threading.Event(), Backoff()), result)


if __name__ == '__main__':
    unittest.main()
