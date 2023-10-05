"""
contains the base class of explainer interfaces.
"""
import sys

sys.path.insert(0, '/data/schoiaj/repos/nli_explain')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.utils import load_pretrained_config


class ExplainerInterface:

    def __init__(self, model_name, device='cpu', baseline_token='[MASK]'):
        config = load_pretrained_config(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config['model_card']).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_card'])
        self.label_map = config['label_map']
        self.device = device
        self.baseline_token = baseline_token

    def explain(premise, hypothesis, **kwargs):
        raise NotImplementedError

    def get_label_map(self, inv=False):
        if inv:
            return {idx: label for label, idx in self.label_map.items()}
        return self.label_map
