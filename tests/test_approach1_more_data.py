import unittest
from experiments.l2_approach1_more_data import candidates,TOTALS,SEEDS


class MoreDataTests(unittest.TestCase):
    def test_twenty_distinct_candidates_include_new_families(self):
        rows=candidates()
        self.assertEqual(len(rows),20)
        self.assertEqual(len({r['name'] for r in rows}),20)
        self.assertEqual(len({r['seed'] for r in rows}),20)
        self.assertTrue({'elu','gated'} <= {r['family'] for r in rows})
        self.assertEqual({r['width'] for r in rows},{128,256})

    def test_counts_and_seed_ranges_are_separate(self):
        self.assertEqual(TOTALS,dict(train=300,validation=100))
        self.assertEqual(TOTALS['train']-81,219)
        self.assertEqual(TOTALS['validation']-47,53)
        self.assertGreaterEqual(abs(SEEDS['train']-SEEDS['validation']),100)


if __name__=='__main__': unittest.main()
