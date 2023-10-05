import sys

sys.path.insert(0, '/data/schoiaj/repos/nli_explain')

from ground_truth_eval.evaluate_explanation import find_common_tokens
import unittest


class TestEvaluation(unittest.TestCase):

    def test_find_common_tokens(self):
        pred_list = [
            ['man', 'walks', 'man'],
            ['man', 'man'],
            [],
        ]
        gt_list = [
            ['the', 'man', 'walks'],
            ['the', 'man', 'walks', 'man'],
            ['none'],
        ]
        answers = [
            ['man', 'walks'],
            ['man', 'man1'],
            [],
        ]

        for p, g, a in zip(pred_list, gt_list, answers):
            common = find_common_tokens(p, g)
            self.assertEqual(set(a), common)


if __name__ == '__main__':
    unittest.main()