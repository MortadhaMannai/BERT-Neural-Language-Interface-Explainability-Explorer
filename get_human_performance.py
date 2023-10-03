import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import json

import pandas as pd

from utils.data_utils import get_annotator_rationales

DATA_ROOT = 'data/e-SNLI'

test_df = pd.read_csv(f'{DATA_ROOT}/esnli_test.csv')

token_rationales, interaction_rationales = get_annotator_rationales(test_df)

for i, (token_rat,
        interaction_rat) in enumerate(zip(token_rationales, interaction_rationales), 1):
    """
    {
        "pred_label": "contradiction",
        "premise_rationales": [
            "choir",
            "songs",
            "church"
        ],
        "hypothesis_rationales": [
            "ceiling"
        ]
    },
    """
    token_explanation = [{
        'premise_rationales': rationale[0],
        'hypothesis_rationales': rationale[1]
    } for rationale in token_rat]

    with open(f'explanations/annotator{i}_token.json', 'w') as f:
        json.dump(token_explanation, f, indent=4)

    interaction_explanation = [{
        'pred_rationales': rationale
    } for rationale in interaction_rat]

    with open(f'explanations/annotator{i}_interaction.json', 'w') as f:
        json.dump(interaction_explanation, f, indent=4)
