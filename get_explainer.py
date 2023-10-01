"""
A abstracted API for getting the API with only public config.
"""
import sys

sys.path.insert(0, '/data/schoiaj/repos/nli_explain')

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from explainers.archipelago.application_utils.text_utils import (AttentionXformer,
                                                                 TextXformer,
                                                                 get_input_baseline_ids,
                                                                 get_token_list,
                                                                 process_stop_words)
from explainers.archipelago.application_utils.text_utils_torch import \
    BertWrapperTorch
from explainers.archipelago.explainer import Archipelago, CrossArchipelago
from explainers.base_explainer import ExplainerInterface
from utils.utils import load_pretrained_config


class ArchExplainerInterface(ExplainerInterface):

    def __init__(self,
                 model_name,
                 device='cpu',
                 baseline_token='[MASK]',
                 explainer_class='arch'):
        config = load_pretrained_config(model_name)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_card'])
        model = AutoModelForSequenceClassification.from_pretrained(config['model_card'])
        self.model_wrapper = BertWrapperTorch(model, device)
        self.label_map = config['label_map']
        self.device = device

        if 'attention' in baseline_token:
            self.baseline_token = baseline_token.split('+')[1]
            self.xformer_class = AttentionXformer
        else:
            self.baseline_token = baseline_token
            self.xformer_class = TextXformer

        if 'cross_arch' in explainer_class:
            self.explainer_class = CrossArchipelago
        elif 'arch' in explainer_class:
            self.explainer_class = Archipelago
        else:
            raise NotImplementedError

        # if pairwise, we use archdetect only
        self.pairwise = True if 'pair' in explainer_class else False

    def explain(self,
                premise,
                hypothesis,
                topk=5,
                batch_size=32,
                do_cross_merge=False,
                output_indices=None):

        text_inputs, baseline_ids = get_input_baseline_ids(premise,
                                                           self.baseline_token,
                                                           self.tokenizer,
                                                           text_pair=hypothesis)
        _text_inputs = {k: v[np.newaxis, :] for k, v in text_inputs.items()}
        xf = self.xformer_class(text_inputs,
                                baseline_ids,
                                sep_token_id=self.tokenizer.sep_token_id)

        # use predicted class to explain the model's decision
        pred = np.argmax(self.model_wrapper(**_text_inputs)[0])

        if output_indices is None:
            output_indices = pred
        apgo = self.explainer_class(self.model_wrapper,
                                    data_xformer=xf,
                                    output_indices=output_indices,
                                    batch_size=batch_size)
        # NOTE: here, topk means sth different
        explanation, _ = apgo.explain(top_k=topk,
                                      use_embedding=True,
                                      do_cross_merge=do_cross_merge,
                                      get_cross_effects=self.pairwise,
                                      separate_effects=True)
        tokens = get_token_list(text_inputs['input_ids'], self.tokenizer)
        explanation, tokens = process_stop_words(explanation,
                                                 tokens,
                                                 strip_first_last=False)

        return explanation, tokens, pred
