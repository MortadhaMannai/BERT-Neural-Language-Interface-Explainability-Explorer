from utils.data_utils import get_clean_token_list
from explainers.base_explainer import ExplainerInterface


class SelectAll(ExplainerInterface):

    def explain(self, sent1, sent2, **kwargs):
        sent1 = get_clean_token_list(sent1)
        sent2 = get_clean_token_list(sent2)

        tokens = sent1 + ['[SEP]'] + sent2
        explanations = {tuple(range(len(tokens))): 1}
        pred = 0
        return explanations, tokens, pred
