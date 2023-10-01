import numpy as np
import copy
import string


def get_efficient_mask_indices(inst, baseline, input):
    invert = np.sum(1 * inst) >= len(inst) // 2
    if invert:
        context = input.copy()
        insertion_target = baseline
        mask_indices = np.argwhere(inst == False).flatten()
    else:
        context = baseline.copy()
        insertion_target = input
        mask_indices = np.argwhere(inst == True).flatten()
    return mask_indices, context, insertion_target


class TextXformer:
    # NOTE: this xformer is not the transformer from Vaswani et al., 2017

    def __init__(self, inputs, baseline_ids, sep_token_id):
        """
        inputs: dict of huggingface tokenizer output. assume each array have
            shape [T,].
        baseline_ids: a np array of shape [T,]
        """
        self.input = inputs
        self.input_ids = inputs['input_ids']
        self.baseline_ids = baseline_ids
        self._num_features = len(self.input_ids)
        self._sep_token_id = sep_token_id
        self._sep_pos = np.where(self.input_ids == sep_token_id)[0][0]

    def __call__(self, inst):
        """
        Insert content of input_ids into baseline
        """
        mask_indices, base, change = get_efficient_mask_indices(
            inst, self.baseline_ids, self.input_ids)
        for i in mask_indices:
            base[i] = change[i]
        return base

    def process_batch_ids(self, batch_ids):
        """
        batch_ids: list of numpy arrays, each is a input_ids
        """
        batch = {}
        for k, v in self.input.items():
            if k == 'input_ids':
                batch[k] = np.array(batch_ids)
            else:
                batch[k] = np.repeat(v[np.newaxis, :], len(batch_ids), axis=0)
        return batch

    def get_baseline_inputs(self):
        return self.process_batch_ids([self.baseline_ids])

    @property
    def sep_pos(self):
        return self._sep_pos

    @property
    def sep_token_id(self):
        return self._sep_token_id

    @property
    def num_features(self):
        return self._num_features


class AttentionXformer(TextXformer):

    # def __call__(self, inst):
    #     """
    #     Insert content of input_ids into baseline

    #     this function does nothing: since the inst will be binary,
    #     where 1 denotes the input and 0 denotes baseline. This is exactly the
    #     same as what we want in attention mask.
    #     """
    #     return inst

    def process_batch_ids(self, batch_ids):
        """
        batch_ids: list of numpy arrays, each is a input_ids
        """
        batch = {}
        for k, v in self.input.items():
            if k == 'attention_mask':
                batch[k] = np.array(batch_ids)
            else:
                batch[k] = np.repeat(v[np.newaxis, :], len(batch_ids), axis=0)
        return batch

    def get_baseline_inputs(self):
        return super().process_batch_ids([self.baseline_ids])


def process_stop_words(explanation, tokens, strip_first_last=True):
    explanation = copy.deepcopy(explanation)
    tokens = copy.deepcopy(tokens)
    stop_words = set([
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "s",
        "ll",
    ] + list(string.punctuation))
    for i, token in enumerate(tokens):
        if token in stop_words:
            if (i,) in explanation:
                explanation[(i,)] = 0.0

    if strip_first_last:
        explanation.pop((0,))
        explanation.pop((len(tokens) - 1,))
        tokens = tokens[1:-1]
    return explanation, tokens


def get_input_baseline_ids(text, baseline_token, tokenizer, text_pair=None):
    """
    return: dict of numpy arrays
    """
    inputs = prepare_huggingface_data(text, tokenizer, text_pair=text_pair)
    inputs = {k: v[0] for k, v in inputs.items()}
    baseline_id = prepare_huggingface_data(baseline_token, tokenizer)['input_ids'][0][1]

    # make baseline inputs
    input_ids = inputs['input_ids']
    baseline_ids = np.where(
        np.isin(input_ids, [tokenizer.sep_token_id, tokenizer.cls_token_id]), input_ids,
        baseline_id)

    return inputs, baseline_ids


def get_token_list(sentence, tokenizer):
    if isinstance(sentence, str):
        X = prepare_huggingface_data(sentence, tokenizer)
        batch_ids = X["input_ids"][0]
    else:
        batch_ids = sentence
    tokens = tokenizer.convert_ids_to_tokens(batch_ids)
    return tokens


def prepare_huggingface_data(sentences, tokenizer, text_pair=None):
    encoded_sentence = tokenizer(sentences,
                                 text_pair=text_pair,
                                 padding='longest',
                                 return_tensors='np',
                                 return_token_type_ids=True,
                                 return_attention_mask=True)
    return encoded_sentence
