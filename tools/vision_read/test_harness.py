"""Regression checks for evaluation risks, not model-reading claims."""
import copy
import sys
sys.dont_write_bytecode = True
import unittest
from common import validate
from run import parse_response
from score import aggregate, score_system


def fixture():
    item = {'id': 'fixture_0', 'video': 'fixture', 'channel': 'fixture', 'variants': [{}], 'primary': 0}
    note = {'id': 'n0', 'x': 20, 'y': 25, 'pos': 5, 'shape': 'filled', 'inst': 'snare', 'onset_q': '3/4', 'duration_q': '1/4', 'bar': 0,
            'ghost': False, 'accent': False, 'tuplet': None, 'tied_continuation': False}
    gold = {'notes': [note], 'staff_lines': [10, 20, 30, 40, 50], 'clef_visible': False, 'complete': False, 'exhaustive_regions': [[0, 0, 50, 60]], 'bars': None}
    head = {'id': 'h0', 'x': 20, 'y': 25, 'pos': 5, 'notehead': 'filled', 'instrument': 'snare', 'ghost': False, 'accent': False, 'grace': False, 'tuplet': None, 'tie_continuation': False}
    reading = {'schema_version': 1, 'system_id': 'fixture_0', 'staff': {'line_count': 5, 'lines_y': [10, 20, 30, 40, 50], 'clef_visible': False}, 'time_signature': None,
               'bar_count': 1, 'bars': [{'index': 0, 'partial_left': False, 'partial_right': False, 'events': [{'onset': '3/16', 'duration': '1/16', 'voice': '1', 'heads': [head]}]}], 'uncertain': [], 'beat_checks': []}
    return item, gold, reading


class HarnessTests(unittest.TestCase):
    def test_rational_units_and_bar(self):
        item, gold, reading = fixture()
        row = score_system(item, gold, reading)
        self.assertEqual(row['metrics']['onset']['correct'], 1)
        self.assertEqual(row['metrics']['duration']['correct'], 1)
        gold['notes'][0]['bar'] = 1
        self.assertEqual(score_system(item, gold, reading)['metrics']['onset']['correct'], 0)

    def test_partial_precision_and_duplicate(self):
        item, gold, reading = fixture()
        heads = reading['bars'][0]['events'][0]['heads']
        heads.append({**heads[0], 'id': 'h1', 'x': 200})
        self.assertEqual(score_system(item, gold, reading)['metrics']['head']['fp'], 0)
        heads.append({**heads[0], 'id': 'h2'})
        row = score_system(item, gold, reading)
        self.assertEqual(row['metrics']['head']['tp'], 1)
        self.assertEqual(row['metrics']['head']['fp'], 1)
        self.assertEqual(row['metrics']['head']['precision'], .5)

    def test_label_cannot_change_geometry_matching(self):
        item, gold, reading = fixture()
        reading['bars'][0]['events'][0]['heads'][0]['instrument'] = 'kick'
        row = score_system(item, gold, reading)
        self.assertEqual(row['metrics']['head']['correct'], 1)
        self.assertEqual(row['metrics']['instrument']['correct'], 0)
        self.assertEqual(row['by_instrument']['snare']['fn'], 1)
        self.assertEqual(row['by_instrument']['kick']['fp'], 1)

    def test_unknown_instrument_not_a_negative(self):
        item, gold, reading = fixture()
        gold['notes'][0]['inst'] = None
        row = score_system(item, gold, reading)
        self.assertEqual(row['metrics']['instrument']['total'], 0)
        self.assertEqual(row['by_instrument'], {})

    def test_tie_positive_and_not_attack(self):
        item, gold, reading = fixture()
        gold['notes'][0]['tied_continuation'] = True
        row = score_system(item, gold, reading)
        self.assertEqual(row['metrics']['tie']['positive_total'], 1)
        self.assertEqual(row['metrics']['tie']['positive_correct'], 0)
        self.assertEqual(row['metrics']['attack_onset']['total'], 0)
        reading['bars'][0]['events'][0]['heads'][0]['tie_continuation'] = True
        self.assertEqual(score_system(item, gold, reading)['metrics']['tie']['positive_correct'], 1)

    def test_empty_gold_and_partial_system_not_perfect(self):
        item, gold, reading = fixture()
        gold['notes'] = []
        row = score_system(item, gold, reading)
        summary = aggregate([row])
        self.assertIsNone(summary['metrics']['head']['value'])
        self.assertEqual(summary['metrics']['known_fields_system_correct']['correct'], 0)
        self.assertEqual(summary['metrics']['known_fields_system_correct']['total'], 0)

    def test_missing_run_counts_as_failures(self):
        item, gold, _ = fixture()
        row = score_system(item, gold, None)
        self.assertEqual(row['metrics']['head']['fn'], 1)
        self.assertEqual(row['metrics']['onset']['correct'], 0)
        self.assertEqual(row['metrics']['clef']['correct'], 0)

    def test_tuplet_rational_and_membership(self):
        item, gold, reading = fixture()
        gold['notes'][0]['tuplet'] = '6:4'
        gold['notes'][0]['duration_q'] = '1/6'
        event = reading['bars'][0]['events'][0]
        event['duration'] = '1/24'
        event['heads'][0]['tuplet'] = {'ratio': '6:4', 'group': 't0'}
        row = score_system(item, gold, reading)
        self.assertEqual(row['metrics']['duration']['correct'], 1)
        self.assertEqual(row['metrics']['tuplet_ratio']['correct'], 1)

    def test_strict_schema_and_final_json(self):
        import json
        _, _, reading = fixture()
        self.assertEqual(parse_response('status\n```json\n'+json.dumps(reading)+'\n```'), reading)
        self.assertEqual(validate(reading, 'fixture_0'), reading)
        with self.assertRaises(ValueError):
            parse_response(json.dumps(reading)+'\n'+json.dumps(reading))
        reading['bar_count'] = 2
        with self.assertRaises(ValueError):
            validate(reading, 'fixture_0')

    def test_equivalent_representation_only(self):
        _, _, reading = fixture()
        reading['time_signature'] = '4/4'
        reading['bars'][0]['events'][0]['heads'][0]['notehead'] = 'x'
        checked = validate(reading, 'fixture_0')
        self.assertEqual(checked['time_signature'], {'numerator': 4, 'denominator': 4})
        self.assertEqual(checked['bars'][0]['events'][0]['heads'][0]['notehead'], 'cross')
        self.assertEqual(checked['bars'][0]['events'][0]['onset'], '3/16')

    def test_symbolic_score_does_not_depend_on_pixels(self):
        item, gold, reading = fixture()
        gold['complete'] = True
        reading['bars'][0]['events'][0]['heads'][0]['x'] = 1000
        row = score_system(item, gold, reading)
        self.assertEqual(row['metrics']['head']['correct'], 0)
        self.assertEqual(row['metrics']['full_symbolic_onset']['correct'], 1)
        self.assertEqual(row['metrics']['full_symbolic_onset_duration']['correct'], 1)

    def test_timing_adapter_preserves_native_gate_and_no_export(self):
        from export_timing import convert
        _, _, reading = fixture()
        anchor = {'staff_reviewed': True, 'staff': {'lines': [10, 20, 30, 40, 50], 'spacing': 10, 'reasons': [], 'clef_confirmed': True},
                  'native_spacing': 7, 'bar_offsets_whole': ['0'], 'time_origin_seconds': 10, 'seconds_per_whole': 2, 'interval': [10, 12], 'verified_heads': []}
        meta = {'system': 0, 'system_id': 'fixture_0'}
        legend = {'rules': {'filled:5': {'inst': 'snare', 'verified': True}}}
        output = convert(reading, meta, legend, anchor)
        self.assertIn('native staff spacing below 8 px', output['staff']['reasons'])
        self.assertFalse(output['notes'][0]['exportable'])
        self.assertEqual(output['notes'][0]['native_x'], 20)
        self.assertEqual(output['notes'][0]['onset_q'], '3/4')
        self.assertEqual(output['rhythm_t'], [10.375])


if __name__ == '__main__':
    unittest.main()
