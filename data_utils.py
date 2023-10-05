import os
from collections import Counter
import re
from itertools import chain, product

import pandas as pd
from nltk.corpus import stopwords

remove_punctuation = re.compile(r'[\.\*,;!:\(\)]')
split_punctuation = re.compile(r'[ \-\'&/`]')


def get_clean_token_list(text):
    return [remove_punctuation.sub('', w) for w in split_punctuation.split(text)]


def perturb_text(text, baseline_token='[MASK]'):
    text_list = get_clean_token_list(text)
    text_perturbed = []
    for i, delete in enumerate(text.split(' ')):
        if delete.lower() in set(stopwords.words('english')):
            continue
        text_perturbed.append(
            (' '.join([baseline_token if w == delete else w for w in text_list]), (delete,
                                                                                   i)))

    return text_perturbed, text_list


def get_contiguous_phrases(text, tokens):
    """
    merge tokens into one list if they are contiguous in the text.
    example:
        text = I love eating food
        tokens = [I, eating, food]

        returns: [[I], [eating, food]]
    """
    full_tokens = get_clean_token_list(text)
    phrases = []
    current_phrase = []
    current_phrase_idx = 0
    full_text_idx = 0
    for tok in tokens:
        while tok != full_tokens[full_text_idx]:
            full_text_idx += 1
        if len(current_phrase) == 0 or current_phrase_idx + 1 == full_text_idx:
            # start of new phrase or contiguous
            current_phrase.append(tok)
        else:
            phrases.append(current_phrase)
            current_phrase = [tok]
        current_phrase_idx = full_text_idx
    if len(current_phrase) > 0:
        phrases.append(current_phrase)

    return phrases


def get_interaction_rationales(df, how='vote'):
    """
    This function takes pandas dataframe and modifies it inplace: adds two columns.

    df: pd.DataFrame
    how: one of "vote", "union". vote option does majority vote.

    NOTE: this function assumes get_token_rationales function is already run.

    This funciton gets interaction rationales from e-SNLI dataset. The way it works:
    1. obtain word groups from highlighted words by merging contiguous words into a group
    2. assume all word groups in premise have interaction with all word group in hypothesis.

    NOTE: this is pretty naive, and there will be a lot of false positive in the annotation.
    Hence, we use precision in the metric later on.
    """
    assert f'Sentence1_{how}' in df.columns, "please run get_token_rationales first."

    matched_rationale = {f'interactions_{how}': []}
    for index, row in df.iterrows():
        sent1_groups = get_contiguous_phrases(row['Sentence1'], row[f'Sentence1_{how}'])
        sent2_groups = get_contiguous_phrases(row['Sentence2'], row[f'Sentence2_{how}'])
        matched_rationale[f'interactions_{how}'].append(
            list(product(sent1_groups, sent2_groups)))

    for k, v in matched_rationale.items():
        df[k] = v


def get_token_rationales(df, how='vote'):
    """
    This function takes pandas dataframe and modifies it inplace: adds two columns.

    df: pd.DataFrame
    how: one of "vote", "union". vote option does majority vote.
    """
    matched_rationale = {f'Sentence1_{how}': [], f'Sentence2_{how}': []}
    for i in range(len(df)):
        for j in (1, 2):
            rationales = []
            for k in (1, 2, 3):
                current_marked = df.iloc[i][f'Sentence{j}_marked_{k}']
                current_marked = current_marked.strip().split(' ')
                match = []
                for idx, w in enumerate(current_marked):
                    if w.startswith('*') and w.endswith('*'):
                        cleaned = remove_punctuation.sub('', w)
                        match.extend([(idx, x) for x in split_punctuation.split(cleaned)])

                rationales.append(match)
            counts = Counter(chain.from_iterable(rationales))
            if how == 'vote':
                matched_rationale[f'Sentence{j}_{how}'].append([
                    k[1]
                    for k, v in sorted(counts.items(), key=lambda x: x[0][0])
                    if v >= 2
                ])
            elif how == 'union':
                matched_rationale[f'Sentence{j}_{how}'].append(
                    [k[1] for k, _ in sorted(counts.items(), key=lambda x: x[0][0])])
            else:
                raise NotImplementedError

    for k, v in matched_rationale.items():
        df[k] = v


def get_annotator_rationales(df):
    annotators_token_rationale = [[], [], []]
    for i in range(len(df)):
        for k in (1, 2, 3):
            rationale = []
            for j in (1, 2):
                current_marked = df.iloc[i][f'Sentence{j}_marked_{k}']
                current_marked = current_marked.strip().split(' ')
                match = []
                for i, w in enumerate(current_marked):
                    if w.startswith('*') and w.endswith('*'):
                        cleaned = remove_punctuation.sub('', w)
                        match.extend(split_punctuation.split(cleaned))
                rationale.append(match)
            annotators_token_rationale[k - 1].append(rationale)

    annotators_interaction_rationale = [[], [], []]
    for index, row in df.iterrows():
        for i, annotator in enumerate(annotators_token_rationale):
            sent1_groups = get_contiguous_phrases(row['Sentence1'], annotator[index][0])
            sent2_groups = get_contiguous_phrases(row['Sentence2'], annotator[index][1])
            annotators_interaction_rationale[i].append(
                list(product(sent1_groups, sent2_groups)))

    return annotators_token_rationale, annotators_interaction_rationale


def load_df(data_path,
            how='union',
            label_map=None,
            mode='test',
            tokenizer=None,
            rationale_format='token'):
    """
    how: union vs vote - how to get the rationale from the three annotators
    label_map: output class index to corresponding label for the pretrained model
    mode: test of dev set
    tokenizer: tokenize the rationale so that it is compatible with the model's
               explanation
    format: should the rationale be in `token` format or `interaction` format
    """
    if data_path.endswith('.csv'):
        # if full path is given, load from it
        df = pd.read_csv(data_path)
    else:
        df = pd.read_csv(os.path.join(data_path, f'esnli_{mode}_processed.csv'))
    # get sentences
    sent1 = df['Sentence1'].tolist()
    sent2 = df['Sentence2'].tolist()

    # get rationales
    if rationale_format == 'token':
        sent1_rationale = [eval(rat) for rat in df[f'Sentence1_{how}']]
        sent2_rationale = [eval(rat) for rat in df[f'Sentence2_{how}']]
        if tokenizer is not None:
            sent1_rationale = [
                tokenizer.tokenize(' '.join(sent)) for sent in sent1_rationale
            ]
            sent2_rationale = [
                tokenizer.tokenize(' '.join(sent)) for sent in sent2_rationale
            ]
        gt_rationale = (sent1_rationale, sent2_rationale)
    elif rationale_format == 'interaction':
        gt_rationale = [eval(rat) for rat in df[f'interactions_{how}']]
        if tokenizer is not None:
            gt_rationale = [[[
                tuple(tokenizer.tokenize(' '.join(group))) for group in interaction
            ] for interaction in ex] for ex in gt_rationale]
    else:
        raise NotImplementedError

    # get labels
    label = df['gold_label']
    if label_map is not None:
        label = label.apply(label_map.get)
    label = label.tolist()

    return (sent1, sent2), gt_rationale, label
