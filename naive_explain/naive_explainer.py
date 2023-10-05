import sys

sys.path.insert(0, '/data/schoiaj/repos/nli_explain')

import itertools

import numpy as np
import torch
from utils.data_utils import perturb_text
from explainers.base_explainer import ExplainerInterface
from explainers.archipelago.explainer import cross_merge


class NaiveExplainer(ExplainerInterface):

    def __init__(self,
                 model_name,
                 device='cpu',
                 baseline_token='[MASK]',
                 interaction_occlusion=False):
        super().__init__(model_name, device=device, baseline_token=baseline_token)
        self.interaction_occlusion = interaction_occlusion

    def explain(self,
                premise,
                hypothesis,
                output_indices=None,
                sent_k=None,
                return_cache=False,
                do_cross_merge=False):
        """
        explain with naive occlusion: pairwise interaction

        sent_k = topk within each sent. Not needed if not for very long sents.

        interaction_occlusion: Either to use
            f(x + i + j) - f(x + i) - f(x + j) + f(x)
            as occlusion.
        """
        full_inp = self.tokenizer(premise, text_pair=hypothesis, return_tensors='pt')
        logits = torch.softmax(self.model(**full_inp.to(self.device)).logits[0], dim=-1)
        orig_confidence, pred_class = logits.max(-1)
        if output_indices is None:
            output_indices = pred_class.item()
        orig_confidence = orig_confidence.item()

        # perturb inputs
        perturbed_premise, pre_tokens = perturb_text(premise,
                                                     baseline_token=self.baseline_token)
        perturbed_hyp, hyp_tokens = perturb_text(hypothesis,
                                                 baseline_token=self.baseline_token)
        tokens = [self.tokenizer.cls_token] + pre_tokens \
               + [self.tokenizer.sep_token] + hyp_tokens + [self.tokenizer.sep_token]

        if sent_k is not None or self.interaction_occlusion:
            # premise first
            pre_confidences = []
            pre_effects = []
            for sent, _ in perturbed_premise:
                inp = self.tokenizer(sent, text_pair=hypothesis, return_tensors='pt')
                conf = torch.softmax(self.model(**inp.to(self.device)).logits[0],
                                     dim=-1)[output_indices].item()
                effect = orig_confidence - conf
                pre_confidences.append(conf)
                pre_effects.append(effect)

            hyp_confidences = []
            hyp_effects = []
            for sent, _ in perturbed_hyp:
                inp = self.tokenizer(premise, text_pair=sent, return_tensors='pt')
                conf = torch.softmax(self.model(**inp.to(self.device)).logits[0],
                                     dim=-1)[output_indices].item()
                effect = orig_confidence - conf
                hyp_confidences.append(conf)
                hyp_effects.append(effect)
        if sent_k is not None:
            pre_topk = torch.tensor(pre_confidences).topk(
                k=min(sent_k, len(pre_confidences)))[1].tolist()
            hyp_topk = torch.tensor(hyp_confidences).topk(
                k=min(sent_k, len(hyp_confidences)))[1].tolist()
        else:
            pre_topk = list(range(len(perturbed_premise)))
            hyp_topk = list(range(len(perturbed_hyp)))

        top_pairs = list(itertools.product(pre_topk, hyp_topk))
        explanation = {}
        for pair in top_pairs:
            inp = self.tokenizer(perturbed_premise[pair[0]][0],
                                 text_pair=perturbed_hyp[pair[1]][0],
                                 return_tensors='pt')
            conf = torch.softmax(self.model(**inp.to(self.device)).logits[0],
                                 dim=-1)[output_indices].item()
            if self.interaction_occlusion:
                effect = orig_confidence - pre_confidences[pair[0]] - hyp_confidences[
                    pair[1]] + conf
            else:
                effect = orig_confidence - conf

            # the pair idx should be in terms of full tokens: not pre and hyp separated.
            # original premise idx + 1 (reason: cls)
            # original hyp idx + len(premise) + 2 (reason: cls, sep)
            pair_idx = (perturbed_premise[pair[0]][1][1] + 1,
                        perturbed_hyp[pair[1]][1][1] + len(pre_tokens) + 2)
            explanation[pair_idx] = effect

        if do_cross_merge:
            explanation = sorted(explanation.items(), key=lambda x: x[1], reverse=True)
            sep_pos = tokens.index(self.tokenizer.sep_token)
            pre_set, cross_set, hyp_set = [], [], []
            for inter_set, strength in explanation:
                if inter_set[0] < sep_pos and inter_set[1] < sep_pos:
                    pre_set.append([inter_set, {'all': strength}])
                elif inter_set[0] < sep_pos and inter_set[1] > sep_pos:
                    cross_set.append([inter_set, {'all': strength}])
                else:
                    hyp_set.append([inter_set, {'all': strength}])

            explanation = cross_merge(pre_set, cross_set, hyp_set, sum_strength=True)

        return_value = (explanation, tokens, pred_class.item())
        cache = (orig_confidence,)

        if return_cache:
            return return_value + cache
        return return_value

    # deprecated
    def analyze_result(self, premise, hypothesis, prediction, confidence, conf_drops,
                       perturbations):
        print('premise:', premise)
        print('hypothesis:', hypothesis)
        print()
        class_map = ['contradiction', 'entailment', 'neutral']
        print(
            f'original prediction was {class_map[prediction]} / with confidence: {confidence}\n'
        )
        conf_drops = np.array(conf_drops)
        idx = conf_drops.argsort(axis=0)[::-1]
        conf_drops = conf_drops[idx]
        perturbations = [perturbations[i] for i in idx]
        pert_sents = [(s[0][0], s[1][0]) for s in perturbations]
        pert_pre_words = set([s[0][1] for s in perturbations])
        pert_hyp_words = set([s[1][1] for s in perturbations])

        for i, (pert, conf) in enumerate(zip(pert_sents, conf_drops), 1):
            print(f'{i}. {pert} | -{conf}')

        print()
        print('premise:', pert_pre_words, '\nhypothesis:', pert_hyp_words)
