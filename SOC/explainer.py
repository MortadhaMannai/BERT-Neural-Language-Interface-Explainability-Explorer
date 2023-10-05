import torch
from transformers import AutoModelForSequenceClassification, AutoModelForMaskedLM, AutoTokenizer


class ContextSampler:
    """
    we use pretrained masked language model for sampling the context.
    """

    def __init__(self, model_name):
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def sample(self, batch_text, sample_n=5):
        inputs = self.tokenizer(batch_text, return_tensors='pt')
        non_mask_position = torch.where(
            inputs['input_ids'] != self.tokenzer.mask_token_id)
        logits = self.model(**inputs).logits  # [b, T, V]
        # TODO: just using topk as sampling k, but this is independent, not LM prob.
        prob, idx = logits.topk(sample_n)  # [b, T, sample_n]

        # set non_context part to original sequence
        # [b, T, sample_n]
        idx[non_mask_position] = inputs['input_ids'][non_mask_position].unsqueeze(-1)
        idx = idx.permute(0, 2, 1)  # [b, sample_n, T]
        sampled_text = []
        for i in idx:
            sampled_text.append(self.tokenizer.batch_decode(i))  # list of list of str

        return sampled_text


class SampleOcclusionExplainer:

    def __init__(self,
                 model_name,
                 sampler_lm_name='roberta-large',
                 tokenizer=None,
                 context_length=4,
                 baseline_token='[MASK]'):
        self.sampler = ContextSampler(sampler_lm_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.context_length = context_length
        self.baseline_token = baseline_token

    def ablate_context(self, inputs, feature_gorups):
        raise NotImplementedError

    def ablate_features(self, inputs, feature_gorups):
        raise NotImplementedError

    def attribute(self, inputs, feature_group, sample_n=5, output_idx=None):
        """
        attribute to the importance of the feature group with SOC.

        inputs: huggingface tokenizer inputs
        feature_group: list of list token positions
        sample_n: how many context to sample
        output_idx: the output index to attribute from. If none, use the predicted class.
        """
        if output_idx is None:
            # use predicted out class
            if len(inputs['input_ids'].shape) == 1:
                inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
                output_idx = self.model(**inputs).logits.argmax(-1).squeeze().item()

        context_ablated = self.ablate_context(inputs, feature_group)
        context_ablated_text = self.tokenizer.decode(context_ablated)

        sampled_context = self.sampler([context_ablated_text], sample_n=sample_n)
        sampled_context = sampled_context[0]
        sampled_context_ids = self.tokenizer(sampled_context, return_tensors='pt')
        # take care of token_type_ids so that it is still NLI
        sampled_context_ids['token_type_ids'] = inputs['token_type_ids'].repeat(
            sample_n, 1)

        feature_ablated = self.ablate_features(sampled_context_ids, feature_group)

        logits = self.model(**sampled_context_ids).logits  # [sample_n, C]
        occluded_logits = self.model(**feature_ablated).logits  # [sample_n, C]

        return (logits - occluded_logits).mean(0)[output_idx]
