"""We test Our versions, Mask Explainer and Naive Explainer only here.
X-Archipelago (Cross Archipelago) has been tested qualitatively through visualizaiton.
"""
import sys

sys.path.insert(0, '/data/schoiaj/repos/nli_explain')
from explainers.mask_explain.mask_explainer import MaskExplainer
from explainers.naive_explain.naive_explainer import NaiveExplainer
import torch

torch.use_deterministic_algorithms(True)
import unittest


class TestExplainers(unittest.TestCase):

    def setUp(self):
        self.naive_exp = NaiveExplainer('bert-base', device='cuda')
        self.mask_exp = MaskExplainer('bert-base',
                                      device='cuda',
                                      baseline_token='attention+[MASK]')

    def test_naive_formats(self):
        explanation, tokens, pred_class = self.naive_exp.explain(
            'I love food.', 'i hate, food')

        self.assertEqual(
            tokens, ['[CLS]', 'I', 'love', 'food', '[SEP]', 'i', 'hate', 'food', '[SEP]'])

        keys = [(i, j) for i in range(2, 4) for j in range(6, 8)]
        self.assertEqual(sorted(explanation.keys()), keys)

        inv_label_map = self.naive_exp.get_label_map(inv=True)
        self.assertEqual(inv_label_map[pred_class], 'contradiction')

    def test_get_mask(self):
        inputs = self.mask_exp.tokenizer('I love food.',
                                         text_pair='i hate food',
                                         return_tensors='pt')
        masked_inputs, mask = self.mask_exp.get_masks(inputs, mask_p=0.5, mask_n=5000)
        m = mask.float().mean(0)
        self.assertEqual(m[[0, 5, 9]].tolist(), [1., 1., 1.])
        self.assertAlmostEqual(mask.float().mean().item(), 0.65, places=2)
        self.assertTrue((masked_inputs['attention_mask'] == mask).all())

    def test_rho(self):
        inputs = self.mask_exp.tokenizer('I love food.',
                                         text_pair='i hate food',
                                         return_tensors='pt')
        torch.manual_seed(0)
        _, mask = self.mask_exp.get_masks(inputs, mask_p=0.5, mask_n=5000)
        torch.manual_seed(0)
        no_corr = self.mask_exp.explain('I love food.',
                                        'i hate food',
                                        mask_n=5000,
                                        no_correction=True)
        torch.manual_seed(0)
        with_corr = self.mask_exp.explain('I love food.',
                                          'i hate food',
                                          mask_n=5000,
                                          no_correction=False)

        self.assertEqual(no_corr[1], with_corr[1])
        self.assertEqual(no_corr[2], with_corr[2])
        self.assertEqual(no_corr[0].keys(), with_corr[0].keys())

        rho = mask.float().mean(0)
        for key in no_corr[0]:
            self.assertAlmostEqual(with_corr[0][key],
                                   float(no_corr[0][key] / rho[key[0]]),
                                   places=3)


if __name__ == '__main__':
    unittest.main()
