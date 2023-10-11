import sys

sys.path.insert(0, '/data/schoiaj/repos/nli_explain')
import pandas as pd
from utils.data_utils import get_token_rationales, get_contiguous_phrases
import unittest


class TestDataUtils(unittest.TestCase):

    def test_get_token_rationales(self):
        df = pd.DataFrame({
            'Sentence1_marked_1': [
                'test sent without highlight!',
                '*highlight* with *punctuation.*',
                '',
                '*multiple* *highlights!*',
            ],
            'Sentence1_marked_2': [
                'test sent without highlight!',
                'highlight with *punctuation.* *highlight*',
                '',
                '*multiple,* *highlights!*',
            ],
            'Sentence1_marked_3': [
                'test sent *without* highlight!',
                'highlight with *punctuation.*',
                'empty',
                'multiple *highlights!*',
            ],
            'Sentence2_marked_1': [
                'test sent without highlight!',
                '*highlight* with *punctuation.*',
                '',
                '*multiple* *highlights!*',
            ],
            'Sentence2_marked_2': [
                'test sent without highlight!',
                'highlight with *punctuation.* *highlight*',
                '',
                '*multiple* *highlights!*',
            ],
            'Sentence2_marked_3': [
                'test sent *without* highlight!',
                'highlight with *punctuation.*',
                'empty',
                'multiple *highlights!*',
            ],
        })

        answers_df = pd.DataFrame({
            'Sentence1_vote': [[], ['punctuation'], [], ['multiple', 'highlights']],
            'Sentence2_vote': [[], ['punctuation'], [], ['multiple', 'highlights']],
            'Sentence1_union': [['without'], ['highlight', 'punctuation', 'highlight'],
                                [], ['multiple', 'highlights']],
            'Sentence2_union': [['without'], ['highlight', 'punctuation', 'highlight'],
                                [], ['multiple', 'highlights']],
        })
        get_token_rationales(df, 'union')
        get_token_rationales(df, 'vote')

        for i in range(len(df)):
            out = df.iloc[i][[
                'Sentence1_vote', 'Sentence2_vote', 'Sentence1_union', 'Sentence2_union'
            ]]
            self.assertTrue((out == answers_df.iloc[i]).all())

    def test_get_contiguous_phrases(self):
        texts = [
            "I love eating food.",
            "I love eating food.",
            "I love eating food.",
            "I love eating food.",
            "I love eating food in the evening.",
            "I love eating food in the evening.",
        ]
        tokens = [
            ["I", "eating"],
            ["I", "food"],
            ["I", "love", "food"],
            ["I", "love", "eating"],
            ["I", "eating", "food", "the"],
            ["I", "the", "evening"],
        ]
        answers = [[['I'], ['eating']], [['I'], ['food']], [['I', 'love'], ['food']],
                   [['I', 'love', 'eating']], [['I'], ['eating', 'food'], ['the']],
                   [['I'], ['the', 'evening']]]

        for text, token, answer in zip(texts, tokens, answers):
            self.assertEqual(get_contiguous_phrases(text, token), answer)

    if __name__ == '__main__':
        unittest.main()
