"""
A abstracted API for getting the API with only public config.
"""
import sys

sys.path.insert(0, '/data/schoiaj/repos/nli_explain')

import torch

from explainers.base_explainer import ExplainerInterface
from explainers.integrated_hessians.path_explain.explainers.embedding_explainer_torch import \
    EmbeddingExplainerTorch
from explainers.archipelago.explainer import cross_merge


class IHBertExplainer(ExplainerInterface):

    def __init__(self, model_name, device='cpu', baseline_token='[MASK]'):
        super().__init__(model_name, device=device, baseline_token=baseline_token)

        self.explainer = EmbeddingExplainerTorch(self.prediction_model)

    ### Here we define functions that represent two pieces of the model:
    ### embedding and prediction
    def embedding_model(self, **inputs):
        # TODO: This is only for BERT!
        batch_embedding = self.model.bert.embeddings(**inputs)
        return batch_embedding

    def prediction_model(self, batch_embedding):
        # Note: this isn't exactly the right way to use the attention mask.
        # It should actually indicate which words are real words. This
        # makes the coding easier however, and the output is fairly similar,
        # so it suffices for this tutorial.
        # attention_mask = torch.ones(batch_embedding.shape[:2]).to(device)
        # attention_mask = tf.cast(attention_mask, dtype=tf.float32)

        encoder_outputs = self.model.bert.encoder(batch_embedding,
                                                  output_hidden_states=True,
                                                  return_dict=False)
        sequence_output = encoder_outputs[0]
        pooled_output = self.model.bert.pooler(sequence_output)
        logits = self.model.classifier(pooled_output)
        return logits

    def explain(self,
                premise,
                hypothesis,
                output_indices=None,
                batch_size=32,
                num_samples=256,
                use_expectation=False,
                do_cross_merge=False,
                get_cross_effects=False):
        if get_cross_effects:
            do_cross_merge = False
        inputs = self.tokenizer(premise, text_pair=hypothesis,
                                return_tensors='pt').to(self.device)
        ### First we need to decode the tokens from the batch ids.
        batch_sentences = self.tokenizer.tokenize(
            f'{self.tokenizer.cls_token} {premise} {self.tokenizer.sep_token}'
            f' {hypothesis} {self.tokenizer.sep_token}')
        batch_ids = inputs['input_ids']
        inputs.pop('attention_mask')
        batch_embedding = self.embedding_model(**inputs).detach()

        baseline_ids = torch.where(
            batch_ids == self.tokenizer.sep_token_id, self.tokenizer.sep_token_id,
            self.tokenizer.encode(self.baseline_token, add_special_tokens=False)[0])
        baseline_ids[:, 0] = self.tokenizer.cls_token_id
        baseline_inputs = {k: v for k, v in inputs.items()}
        baseline_inputs['input_ids'] = baseline_ids
        baseline_embedding = self.embedding_model(**baseline_inputs).detach()

        logits = self.prediction_model(batch_embedding)
        pred_label = int(torch.argmax(logits, -1))

        ### We are finally ready to explain our model
        explainer = EmbeddingExplainerTorch(self.prediction_model)

        ### For interactions, the hessian is rather large so we use a very small batch size
        interactions = explainer.interactions(
            inputs=batch_embedding,
            baseline=baseline_embedding,
            batch_size=batch_size,
            num_samples=num_samples,
            use_expectation=use_expectation,
            output_indices=pred_label if output_indices is None else output_indices,
            verbose=False)  # [bs, T, T]

        interactions = interactions[0]  # [T, T]
        explanations = {(i, j): interactions[i, j] for i in range(interactions.shape[0])
                        for j in range(interactions.shape[1])}

        if do_cross_merge or get_cross_effects:
            explanations = sorted(explanations.items(), key=lambda x: x[1], reverse=True)
            sep_pos = batch_sentences.index(self.tokenizer.sep_token)
            pre_set, cross_set, hyp_set = [], [], []
            for inter_set, strength in explanations:
                if inter_set[0] < sep_pos and inter_set[1] < sep_pos:
                    pre_set.append([inter_set, {'all': strength}])
                elif inter_set[0] < sep_pos and inter_set[1] > sep_pos:
                    cross_set.append([inter_set, {'all': strength}])
                else:
                    hyp_set.append([inter_set, {'all': strength}])
            if do_cross_merge:
                explanations = cross_merge(pre_set, cross_set, hyp_set, sum_strength=True)
            else:
                explanations = {pair: strength['all'] for pair, strength in cross_set}
        return explanations, batch_sentences, pred_label
