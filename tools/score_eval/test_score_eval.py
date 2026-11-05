import unittest
from score_eval import family, match_heads, time_pairs, display_threshold


def events(*items):
    return [dict(time=t, inst=inst) for t, inst in items]


class MatchingTests(unittest.TestCase):
    def test_max_cardinality_not_greedy_nearest(self):
        self.assertEqual(time_pairs([0, .06], [.04, .10]), [(0, 0), (1, 1)])

    def test_same_family_before_wrong_instrument(self):
        same, wrong, miss, extra = match_heads(events((0,'snare'),(.04,'kick')),
                                             events((.01,'kick'),(.045,'tom')))
        self.assertEqual(same, [(1,0)])
        self.assertEqual(wrong, [(0,1)])
        self.assertEqual((miss, extra), ([],[]))

    def test_chord_one_to_one_and_extra(self):
        same, wrong, miss, extra = match_heads(events((1,'kick'),(1,'snare')),
                         events((1,'kick'),(1,'snare'),(1,'hihat')))
        self.assertEqual(len(same), 2)
        self.assertEqual((wrong,miss,extra), ([],[],[2]))

    def test_boundary_and_missing(self):
        self.assertEqual(time_pairs([1],[1.05]), [(0,0)])
        self.assertEqual(time_pairs([1],[1.0501]), [])
        self.assertEqual(match_heads(events((0,'kick')),[]), ([],[],[0],[]))

    def test_families(self):
        self.assertEqual(family('tom_pos_3'),'tom')
        self.assertEqual(family('ride_bell'),'cymbal')
        self.assertEqual(family('side_stick'),'snare')

    def test_empty_and_duplicate_times(self):
        self.assertEqual(time_pairs([],[]), [])
        self.assertEqual(len(time_pairs([1,1],[1])), 1)

    def test_threshold(self):
        self.assertAlmostEqual(display_threshold(.48),.04)
        self.assertAlmostEqual(display_threshold(.72),.06)


if __name__=='__main__':
    unittest.main()
