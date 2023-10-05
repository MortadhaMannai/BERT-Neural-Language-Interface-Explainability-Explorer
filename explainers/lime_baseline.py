import math
import re
import sys

sys.path.insert(0, '/data/schoiaj/repos/nli_explain')
from lime.lime_text import LimeTextExplainer
import torch
from explainers.base_explainer import ExplainerInterface


class LimeExplainer(ExplainerInterface):

    def predictor(self, texts):
        texts = [t.replace('[[MASK]]', '[SEP]') for t in texts]
        sent1 = []
        sent2 = []
        for t in texts:
            s1, s2 = t.split('[SEP]')
            sent1.append(s1)
            sent2.append(s2)

        all_probs = []
        for i in range(math.ceil(len(texts) / self.bs)):
            batch_idx = slice(i * self.bs, (i + 1) * self.bs)
            encoded = self.tokenizer(sent1[batch_idx],
                                     text_pair=sent2[batch_idx],
                                     return_tensors="pt",
                                     padding=True).to(self.device)
            outputs = self.model(**encoded)
            probs = torch.softmax(outputs.logits, -1).detach().cpu()
            all_probs.append(probs)
        probs = torch.cat(all_probs, 0).numpy()
        return probs

    def explain(self, sent1, sent2, batch_size=32, mask_n=5000, output_indices=None):
        self.bs = batch_size
        str_to_predict = ' [SEP] '.join([sent1, sent2])

        explainer = LimeTextExplainer(class_names=self.get_label_map().keys(),
                                      bow=False,
                                      mask_string=self.baseline_token)
        exp = explainer.explain_instance(str_to_predict,
                                         self.predictor,
                                         top_labels=3,
                                         num_features=20,
                                         num_samples=mask_n)
        pred = int(exp.predict_proba.argmax())
        tokens = re.split(r'\W+', str_to_predict)
        tokens = [tok if tok != 'SEP' else '[SEP]' for tok in tokens]
        if output_indices is None:
            output_indices = pred
        explanations = {(k,): v for k, v in exp.as_map()[output_indices]}
        return explanations, tokens, pred
