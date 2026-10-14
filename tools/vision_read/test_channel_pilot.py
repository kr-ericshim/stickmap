"""Small offline contract tests for the cursor adapter and frozen run handoff."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import channel_pilot as pilot
from channel_pilot import COLLISION, FEW, UNRESOLVED, convert_cursor, decision, match_geometry
from common import DATA, dump, load, sha
from hybrid import cv_reading


class CursorAdapterTest(unittest.TestCase):
    def setUp(self):
        self.cv = dict(system_id='example_chan_0000', width=400, barlines=[],
                       staff=dict(lines=[40, 50, 60, 70, 80], spacing=10, reasons=[], clef_confirmed=True),
                       notes=[dict(label=i + 1, x=x, y=55, pos=5, shape='filled', inst='snare',
                                   verified=True, exportable=True, confidence=.95, shape_confidence=.95)
                              for i, x in enumerate([10, 13, 40, 70, 100, 130, 160, 200, 250])])
        self.original = dict(system=0, notes=copy.deepcopy(self.cv['notes']), staff=copy.deepcopy(self.cv['staff']),
                             interval=[0, 4], display_interval=[0, 5], cursor_x=[0, 80, 160, 240],
                             cursor_t=[0, 1, 2, 3], cursor_mode='box', cursor_coverage=.8,
                             color_x=[10, 100], color_t=[.1, 1.2], read_x=[14, 220], unvisited_heads=[])
        self.meta = dict(system_id=self.cv['system_id'], system=0,
                         channel_geometry=dict(offset_xy=[5, -3], matches=[]))
        self.legend = dict(rules={'filled:5': dict(inst='snare', verified=True)})
        self.reading = cv_reading(self.cv['system_id'], self.cv)
        for i, event in enumerate(self.reading['bars'][0]['events']):
            event.update(onset=str(max(0, i - 1)) + '/16', duration='1/16')

    def convert(self, **kwargs):
        return convert_cursor(self.reading, self.meta, self.cv, self.legend, self.original, **kwargs)

    def test_offset_chord_owned_span_and_existing_prior(self):
        original = copy.deepcopy(self.original)
        system = self.convert()
        self.assertEqual([n['x'] for n in system['notes'][:2]], [16.5, 16.5])
        self.assertEqual(system['notes'][0]['y'], 52)
        self.assertEqual(system['unvisited_head_ids'], ['cv-9'])
        self.assertEqual(system['notes'][-1]['x'], 205)
        self.assertNotIn('rhythm_x', system); self.assertNotIn('rhythm_t', system)
        for key in ('staff', 'interval', 'display_interval', 'cursor_x', 'cursor_t', 'color_x', 'color_t', 'read_x', 'unvisited_heads'):
            self.assertEqual(system[key], original[key])
        self.assertEqual(self.original, original)

    def test_unresolved_inside_blocks_outside_does_not(self):
        self.reading['bars'][0]['events'][-1]['onset'] = None
        self.assertNotIn(UNRESOLVED, self.convert()['staff']['reasons'])
        self.reading['bars'][0]['events'][-2]['onset'] = None
        system = self.convert()
        self.assertIn(UNRESOLVED, system['staff']['reasons'])
        self.assertTrue(system['vision_omitted_heads'][0]['blocking'])

    def test_grace_and_unverified_instrument(self):
        event = self.reading['bars'][0]['events'][0]
        event['heads'][0]['grace'] = True; event['onset'] = None
        self.cv['notes'][1]['verified'] = False
        system = self.convert()
        self.assertNotIn(UNRESOLVED, system['staff']['reasons'])
        self.assertFalse(system['vision_omitted_heads'][0]['blocking'])
        self.assertEqual(system['notes'][0]['inst'], 'unknown')
        self.assertFalse(system['notes'][0]['exportable'])

    def test_collisions_ties_and_few(self):
        # Different rational moments closer than .55 spacing must block.
        self.cv['notes'][2]['x'] = 16
        self.reading['bars'][0]['events'][2]['heads'][0]['x'] = 16
        self.assertIn(COLLISION, self.convert()['staff']['reasons'])
        for event in self.reading['bars'][0]['events'][3:]:
            event['heads'][0]['tie_continuation'] = True
        system = self.convert()
        self.assertIn(FEW, system['staff']['reasons'])
        self.assertFalse(system['notes'][-1]['exportable'])

    def test_cv_sanity_has_no_fabricated_rhythm(self):
        self.reading = cv_reading(self.cv['system_id'], self.cv)
        system = self.convert(cv_sanity=True)
        self.assertNotIn(UNRESOLVED, system['staff']['reasons'])
        self.assertEqual(system['notes'][0]['x'], 15)
        self.assertIsNone(system['notes'][0]['onset_q'])
        self.assertNotIn('rhythm_x', system)

    def test_nonconstant_crop_offset_rejected(self):
        self.meta['channel_geometry']['offset_xy'] = [0, 0]
        reference = copy.deepcopy(self.original)
        for n in reference['notes'][:5]:
            n['x'] += 3
        check = match_geometry(self.cv, reference)
        self.assertTrue(check['reasons'])
        exact = match_geometry(self.cv, self.original)
        self.assertEqual(exact['offset_xy'], [0, 0])
        self.assertFalse(exact['reasons'])

    def test_bar_onsets_are_separate_moments(self):
        # Onset zero in the next bar must not become a chord with bar zero.
        bar = self.reading['bars'][0]
        moved = bar['events'].pop(-2)
        moved['onset'] = '0'
        self.reading['bars'].append(dict(index=1, partial_left=False, partial_right=False, events=[moved]))
        self.reading['bar_count'] = 2
        system = self.convert()
        first = next(n for n in system['notes'] if n['id'] == 'cv-1')
        moved_note = next(n for n in system['notes'] if n['id'] == 'cv-8')
        self.assertNotEqual(first['x'], moved_note['x'])

    def test_unknown_rhythm_and_unlocated_miss_block(self):
        head = self.reading['bars'][0]['events'][2]['heads'][0]
        head['tie_continuation'] = None
        self.assertIn(UNRESOLVED, self.convert()['staff']['reasons'])
        head['tie_continuation'] = False; head['rhythm_uncertain'] = True
        self.assertIn(UNRESOLVED, self.convert()['staff']['reasons'])
        head['rhythm_uncertain'] = False; head['x'] = None
        self.assertIn(UNRESOLVED, self.convert()['staff']['reasons'])

    def test_decision_boundary(self):
        self.assertEqual(decision(4, 386, 193), '전체 수집 진행 권고')
        self.assertEqual(decision(4, 385, 193), '중간')
        self.assertEqual(decision(1, 1000, 193), '커서 있어도 실패 — 원인 분석 필요')

    def test_stage_a_b_integrity_and_missing_reading(self):
        # Synthetic local reading only; no GPT call and no real A/B outputs.
        with tempfile.TemporaryDirectory(prefix='channel-pilot-unit-', dir=DATA) as temporary:
            data = Path(temporary); out = data / 'channel_pilot'; song = data / 'input'
            sid = self.cv['system_id']; video = 'DMPkSQZyj76'
            source_strip = song / 'truth-reviewed/strip.json'
            dump(source_strip, [self.original])
            meta = {**self.meta, 'video': video, 'channel_pilot': pilot.VERSION,
                    'source_strip_sha256': sha(source_strip), 'cv_sha256': '', 'images': [],
                    'record': {'source_system': 0},
                    'automatic_anchor': dict(interval=[0, 4], cursor_x=[0, 80, 160, 240],
                                             cursor_t=[0, 1, 2, 3], cursor_mode='box', owned_x=[0, 230])}
            prepared = data / 'hybrid/jobs' / sid; job = data / 'runs/unit-run' / sid
            reading = copy.deepcopy(self.reading); reading['hybrid'] = {'rejected_cv_heads': []}
            for directory in (prepared, job):
                dump(directory / 'cv.json', self.cv); dump(directory / 'legend.json', self.legend)
                meta['cv_sha256'] = sha(directory / 'cv.json'); dump(directory / 'job.json', meta)
            dump(job / 'reading.json', reading)
            outcome = dict(status='ok', reading_sha256=sha(job / 'reading.json'),
                           identity={'inputs': {name: sha(prepared / name) for name in ('cv.json', 'legend.json', 'job.json')}})
            dump(job / 'result.json', outcome)
            dump(data / 'runs/unit-run/config.json', {'systems': [sid]})
            dump(out / 'systems.json', dict(systems=[sid], videos=[dict(video=video, systems_found=1,
                 dispositions=[dict(id=sid, system=0, status='found', reasons=[])])]))
            with patch.object(pilot, 'DATA', data), patch.object(pilot, 'PILOT', out), \
                    patch.object(pilot, 'source', return_value=(song, 'xiangsong')):
                pilot.stage('A', 'unit-run')
                pilot.stage('B', 'unit-run')
                a = load(out / 'A' / video / 'strip.json')[0]
                b = load(out / 'B' / video / 'strip.json')[0]
                self.assertEqual(a['cursor_x'], self.original['cursor_x'])
                self.assertNotIn('rhythm_x', a)
                self.assertTrue(b['rhythm_x']); self.assertEqual(b['cursor_x'], [])
                outcome['status'] = 'timeout'; dump(job / 'result.json', outcome)
                pilot.stage('A', 'unit-run')
                self.assertEqual(load(out / 'A' / video / 'staging.json')['systems_read'], 0)
                outcome['status'] = 'ok'; dump(job / 'result.json', outcome)
                reading['uncertain'] = ['tampered']; dump(job / 'reading.json', reading)
                with self.assertRaisesRegex(ValueError, 'reading changed after inference'):
                    pilot.stage('A', 'unit-run')


if __name__ == '__main__':
    unittest.main()
