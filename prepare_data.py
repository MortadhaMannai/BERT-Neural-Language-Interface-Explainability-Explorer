import pandas as pd

from utils.data_utils import get_token_rationales, get_interaction_rationales

DATA_ROOT = 'data/e-SNLI'

dev_df = pd.read_csv(f'{DATA_ROOT}/esnli_dev.csv')
get_token_rationales(dev_df, 'vote')
get_token_rationales(dev_df, 'union')
get_interaction_rationales(dev_df, 'vote')
get_interaction_rationales(dev_df, 'union')

dev_df[[
    'Sentence1', 'Sentence2', 'Sentence1_vote', 'Sentence2_vote', 'Sentence1_union',
    'Sentence2_union', 'interactions_vote', 'interactions_union', 'gold_label'
]].to_csv(f'{DATA_ROOT}/esnli_dev_processed.csv', index=False)

test_df = pd.read_csv(f'{DATA_ROOT}/esnli_test.csv')
get_token_rationales(test_df, 'vote')
get_token_rationales(test_df, 'union')
get_interaction_rationales(test_df, 'vote')
get_interaction_rationales(test_df, 'union')

test_df[[
    'Sentence1', 'Sentence2', 'Sentence1_vote', 'Sentence2_vote', 'Sentence1_union',
    'Sentence2_union', 'interactions_vote', 'interactions_union', 'gold_label'
]].to_csv(f'{DATA_ROOT}/esnli_test_processed.csv', index=False)
