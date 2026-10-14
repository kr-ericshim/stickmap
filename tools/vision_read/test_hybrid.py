"""Contract, scoring, annotation and quota tests without Gemini or audio."""
import copy
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
sys.dont_write_bytecode = True
from common import DATA, dump, flatten
from coordination import batch_lock, invoke, quota_text
from hybrid import annotate, base_head, cv_reading, merge
from hybrid_compare import extra_metrics, should_extend
from score import score_system
from test_harness import fixture


def cv_fixture():
    return {'system_id': 'fixture_0', 'width': 80, 'height': 60, 'staff': {'lines': [10, 20, 30, 40, 50], 'spacing': 10, 'clef_confirmed': False},
            'notes': [{'label': 1, 'x': 20, 'y': 25, 'pos': 5, 'shape': 'filled', 'inst': 'snare', 'verified': True, 'ghost': False, 'accent': False}],
            'barlines': [{'label': 'B1', 'x': 75}]}


def answer_fixture():
    return {'schema_version': 1, 'system_id': 'fixture_0', 'time_signature': None,
            'bars': [{'index': 0, 'partial_left': False, 'partial_right': False, 'x_start': 0, 'x_end': 80, 'start_label': None, 'end_label': None}],
            'heads': [{'label': 1, 'status': 'head', 'bar': 0, 'onset': '3/16', 'duration': '1/16', 'voice': '1', 'tie_continuation': False,
                       'tuplet': None, 'ghost': False, 'accent': False, 'grace': False, 'reason': None}],
            'misses': [], 'beat_checks': [], 'uncertain': []}


class HybridTests(unittest.TestCase):
    def test_cv_geometry_and_instrument_cannot_be_overridden(self):
        answer = answer_fixture()
        answer['heads'][0].update(instrument='kick', pos=1, x=1000, y=999, notehead='cross')
        reading = merge(answer, cv_fixture(), 'fixture_0')
        head = flatten(reading)[0]
        self.assertEqual((head['instrument'], head['pos'], head['x'], head['y'], head['notehead']), ('snare', 5, 20, 25, 'filled'))
        item, gold, _ = fixture()
        self.assertEqual(score_system(item, gold, reading)['metrics']['onset']['correct'], 1)
        self.assertEqual(extra_metrics(item, gold, reading)['onset_instrument']['correct'], 1)

    def test_every_label_exactly_once(self):
        for labels in ([], [1, 1], [2], ['1']):
            answer = answer_fixture()
            answer['heads'] = [{**answer['heads'][0], 'label': label} for label in labels]
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                merge(answer, cv_fixture(), 'fixture_0')

    def test_false_positive_is_recorded_and_removed(self):
        answer = answer_fixture()
        answer['heads'][0].update(status='false_positive', reason='a text dot rather than a notehead')
        reading = merge(answer, cv_fixture(), 'fixture_0')
        self.assertEqual(flatten(reading), [])
        self.assertEqual(reading['hybrid']['rejected_cv_heads'][0]['label'], 1)
        answer['heads'][0]['reason'] = None
        with self.assertRaises(ValueError):
            merge(answer, cv_fixture(), 'fixture_0')

    def test_miss_instrument_unverified_even_if_model_supplies_one(self):
        answer = answer_fixture()
        miss = {**answer['heads'][0], 'id': 'a', 'pos': 3, 'notehead': 'filled', 'x': None, 'y': None, 'instrument': 'tom', 'reason': 'unlabelled head'}
        answer['misses'] = [miss]
        reading = merge(answer, cv_fixture(), 'fixture_0')
        self.assertIsNone(flatten(reading)[1]['instrument'])
        self.assertEqual(flatten(reading)[1]['instrument_status'], 'unverified-instrument')

    def test_unlocated_miss_prevents_symbolic_whole_system_exact(self):
        item, gold, _ = fixture(); gold['complete'] = True
        reading = merge(answer_fixture(), cv_fixture(), 'fixture_0')
        self.assertEqual(extra_metrics(item, gold, reading)['symbolic_known_fields_system_exact']['correct'], 1)
        reading['bars'][0]['events'].append(copy.deepcopy(reading['bars'][0]['events'][0]))
        head = reading['bars'][0]['events'][-1]['heads'][0]
        head.update(id='miss-a', x=None, y=None)
        self.assertEqual(extra_metrics(item, gold, reading)['symbolic_known_fields_system_exact']['correct'], 0)

    def test_unknown_origin_null_and_tuplet_and_tie(self):
        answer = answer_fixture()
        answer['bars'][0]['partial_left'] = True
        answer['heads'][0].update(onset=None, duration='1/24', tuplet={'ratio': '6:4', 'group': 'g'}, tie_continuation=True)
        head = flatten(merge(answer, cv_fixture(), 'fixture_0'))[0]
        self.assertIsNone(head['onset']); self.assertTrue(head['tie_continuation'])
        self.assertEqual(head['duration'], '1/24')

    def test_barline_ids_resolve_to_cv_and_unknown_ids_fail(self):
        answer = answer_fixture(); answer['bars'][0]['end_label'] = 'B1'
        self.assertEqual(merge(answer, cv_fixture(), 'fixture_0')['bars'][0]['x_end'], 75)
        answer['bars'][0]['end_label'] = 'B2'
        with self.assertRaises(ValueError): merge(answer, cv_fixture(), 'fixture_0')

    def test_invalid_fraction_and_duplicate_miss_and_grace(self):
        for fraction in (0.25, '-1/4', '1/0'):
            answer = answer_fixture(); answer['heads'][0]['onset'] = fraction
            with self.subTest(fraction=fraction), self.assertRaises((ValueError, ZeroDivisionError)):
                merge(answer, cv_fixture(), 'fixture_0')
        answer = answer_fixture(); answer['heads'][0].update(duration='0', grace=True)
        self.assertEqual(flatten(merge(answer, cv_fixture(), 'fixture_0'))[0]['duration'], '0')
        answer['heads'][0]['grace'] = False
        with self.assertRaises(ValueError): merge(answer, cv_fixture(), 'fixture_0')

    def test_annotation_layout_and_clean_original_kept(self):
        import cv2
        import numpy as np
        base = DATA/'hybrid-test-artifacts'; base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            root = Path(temp); original = root/'original.png'; target = root/'annotated.png'
            image = np.full((60, 80, 3), 255, np.uint8)
            cv2.circle(image, (20, 25), 3, (0, 0, 0), -1); cv2.imwrite(str(original), image)
            before = original.read_bytes()
            layout = annotate(original, cv_fixture(), target)
            self.assertEqual(original.read_bytes(), before)
            self.assertEqual(cv2.imread(str(target)).shape[:2], (244, 80))
            self.assertEqual(layout['head_labels'][0]['label'], 1)
            a,b,c,d = layout['head_labels'][0]['box']
            self.assertTrue(d <= 92 or b >= 152)

    def test_cv_baseline_does_not_fabricate_rhythm(self):
        reading = cv_reading('fixture_0', cv_fixture())
        self.assertIsNone(flatten(reading)[0]['onset']); self.assertIsNone(flatten(reading)[0]['duration'])

    def test_quota_detection(self):
        self.assertTrue(quota_text('RESOURCE_EXHAUSTED (code 429)'))
        self.assertFalse(quota_text('head 4290 at x=429.25'))

    def test_process_quota_stops_without_waiting_for_exit(self):
        base = DATA/'hybrid-test-artifacts'; base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            root = Path(temp); stop = threading.Event()
            state, _ = invoke([sys.executable, '-B', '-c', 'import sys,time;print("RESOURCE_EXHAUSTED (code 429)",file=sys.stderr,flush=True);time.sleep(10)'],
                              root, root/'out.log', root/'err.log', 3, stop)
            self.assertEqual(state, 'quota_exhausted'); self.assertTrue(stop.is_set())
            state, _ = invoke(['no-such-binary'], root, root/'out2.log', root/'err2.log', 3, stop)
            self.assertEqual(state, 'quota_cancelled'); self.assertFalse((root/'out2.log').exists())

    def test_shared_lock_rejects_second_owner(self):
        with batch_lock('unit-test'):
            with self.assertRaises(RuntimeError):
                with batch_lock('other-test'): pass

    def test_runner_end_to_end_and_resume_with_local_fake_cli(self):
        import argparse
        import hybrid_run
        import cv2
        import numpy as np
        base = DATA/'hybrid-test-artifacts'; base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as temp:
            root = Path(temp); source = root/'input/jobs/fixture_0'; source.mkdir(parents=True)
            dump(source/'cv.json', cv_fixture())
            dump(source/'job.json', {'system_id': 'fixture_0', 'video': 'fixture', 'images': []})
            dump(source/'legend.json', {'rules': {}}); dump(source/'annotation.json', {})
            image = np.full((60, 80, 3), 255, np.uint8)
            for name in ('original.png', 'annotated.png'): cv2.imwrite(str(source/name), image)
            cv = cv_fixture(); cv['wall_seconds'] = .1; dump(source/'cv.json', cv)
            fake = root/'fake-agy'
            fake.write_text('#!'+sys.executable+'\nimport json\nprint('+repr(json.dumps(answer_fixture()))+')\n')
            fake.chmod(0o700)
            args = argparse.Namespace(resume=False, agy=str(fake), timeout=3)
            folder = root/'run'; folder.mkdir()
            with patch.object(hybrid_run, 'HYBRID', root/'input'):
                result = hybrid_run.one('fixture_0', args, folder, threading.Event())
                self.assertEqual(result['status'], 'ok')
                self.assertEqual(flatten(json.loads((folder/'fixture_0/reading.json').read_text()))[0]['instrument'], 'snare')
                args.resume = True; fake.unlink()
                self.assertEqual(hybrid_run.one('fixture_0', args, folder, threading.Event()), result)

    def test_vault_rejected_before_cv_reads_image(self):
        from hybrid import cv_prediction
        from tools.harvest.vault import Vault
        with patch.object(Vault, 'is_vault', return_value=True), self.assertRaisesRegex(ValueError, 'vault input forbidden'):
            cv_prediction({'video': 'sealed'})

    def test_rollout_requires_all_complete_gold_and_beats_best_baseline(self):
        ids = set('abcdef'); rows = [{'id': sid, 'status': 'ok'} for sid in ids]
        def metrics(joint, onset, systems=0):
            return {'symbolic_onset_instrument': {'value': joint}, 'full_symbolic_onset': {'value': onset}, 'known_fields_system_correct': {'correct': systems}}
        baselines = [metrics(0, 0), metrics(.52, .60), metrics(.63, .623)]
        self.assertTrue(should_extend(ids, rows, metrics(.85, .70), baselines))
        self.assertFalse(should_extend(ids, rows[:-1], metrics(.99, .99), baselines))
        rows[0]['status'] = 'quota_exhausted'
        self.assertFalse(should_extend(ids, rows, metrics(.99, .99), baselines))
        rows[0]['status'] = 'ok'
        self.assertFalse(should_extend(ids, rows, metrics(.849, .72), baselines))
        self.assertTrue(should_extend(ids, rows, metrics(.70, .724), baselines))

    def test_uncertain_and_rejected_heads_keep_adapter_blockers(self):
        from export_timing import convert
        answer = answer_fixture(); answer['heads'][0]['status'] = 'uncertain'
        reading = merge(answer, cv_fixture(), 'fixture_0')
        anchor = {'staff_reviewed': True, 'staff': {'lines': [10,20,30,40,50], 'spacing': 10, 'reasons': [], 'clef_confirmed': True},
                  'native_spacing': 10, 'bar_offsets_whole': ['0'], 'time_origin_seconds': 0, 'seconds_per_whole': 2,
                  'interval': [0,2], 'verified_heads': []}
        system = convert(reading, {'system': 0, 'system_id': 'fixture_0'}, {'rules': {'filled:5': {'inst':'snare','verified': True}}}, anchor)
        self.assertFalse(system['notes'][0]['exportable'])
        self.assertIn('hybrid numbered heads remain uncertain; independent review required', system['staff']['reasons'])
        reading['hybrid']['rejected_cv_heads'] = [{'label': 2, 'reason': 'a text dot'}]
        system = convert(reading, {'system': 0, 'system_id': 'fixture_0'}, {'rules': {}}, anchor)
        self.assertIn('hybrid false-positive rejections require independent coverage review', system['staff']['reasons'])


if __name__ == '__main__':
    unittest.main()
