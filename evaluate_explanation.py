import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.data_utils import load_df
from utils.utils import load_pretrained_config


def get_numbered_list(token_list):
    """
    If there's a duplicate, append number to the second occurence.

    [man, walks, man, man] -> [man, walks, man1, man2]
    """
    token_set = {}
    new_token_list = []
    for x in token_list:
        if x in token_set:
            new_token_list.append(x.lower() + str(token_set[x]))
            token_set[x] += 1
        else:
            new_token_list.append(x.lower())
            token_set[x] = 1
    return new_token_list


def find_common_tokens(pred, gt):
    """
    The main purpose of this function is to find common tokens in two lists.
    However, it should care about duplicates. e.g.
    [man, walks, man], [the, man, walks]  ->  common: [man, walks]
    [man, man], [the, man, walks, man]    ->  common: [man, walks, man]

    Therefore, normal set intersection logic will not work.

    TODO: 2 out of 3 problem.
        sentence: man man man
        pred: *man* man *man* -> [man, man]
        gt: *man* *man* man -> [man, man]

        the common token should be [man], only the first occurence. But now,
        we cannot account for that.

        NOTE: This is very very rare, and not sure if this happens in the dataset.
    """
    numbered_pred = get_numbered_list(pred)
    numbered_gt = get_numbered_list(gt)

    return set(numbered_pred) & set(numbered_gt)


def compute_f1(pred_rationale, gt_rationale):
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(gt_rationale) == 0:
        if len(pred_rationale) == 0:
            return 1, 1, 1
        return 0, 1, 0
    if len(pred_rationale) == 0:
        if len(gt_rationale) == 0:
            return 1, 1, 1
        return 0, 0, 0

    common_tokens = find_common_tokens(pred_rationale, gt_rationale)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0, 0, 0

    prec = len(common_tokens) / len(pred_rationale)
    rec = len(common_tokens) / len(gt_rationale)
    f1 = 2 * (prec * rec) / (prec + rec)

    return prec, rec, f1


def compute_token_f1_score(gt_rationale1,
                           gt_rationale2,
                           pred_rationale1=None,
                           pred_rationale2=None,
                           pred_rationales=None,
                           topk=5):
    if pred_rationale1 is not None:
        assert pred_rationale2 is not None
    else:
        assert pred_rationales is not None
        pred_rationales = {eval(k): v for k, v in pred_rationales.items()}
        topk_rationales = sorted(pred_rationales.items(),
                                 key=lambda x: x[1],
                                 reverse=True)[:topk]
        pred_rationale1, pred_rationale2 = set(), set()
        for (g1, g2), _ in topk_rationales:
            pred_rationale1.update(g1)
            pred_rationale2.update(g2)

        pred_rationale1, pred_rationale2 = list(pred_rationale1), list(pred_rationale2)

    p1, r1, f1_1 = compute_f1(pred_rationale1, gt_rationale1)
    p2, r2, f1_2 = compute_f1(pred_rationale2, gt_rationale2)

    p3, r3, f3_3 = compute_f1(pred_rationale1 + pred_rationale2,
                              gt_rationale1 + gt_rationale2)

    return [p1, r1, f1_1, p2, r2, f1_2, p3, r3, f3_3]


def jaccard_sim(first, second):
    numbered_first = set(get_numbered_list(first))
    numbered_second = set(get_numbered_list(second))

    if len(numbered_first | numbered_second) == 0:
        return 0
    return len(numbered_first & numbered_second) / len(numbered_first | numbered_second)


def interaction_f1(gt_rationales,
                   pred_rationales,
                   skip_intra_rationale=False,
                   max_only=False,
                   topk=None):
    """
    gt_rationales: list of list of 2 tuple of words:
        [
            [(sent1 group), (sent2 group)],   # 1st interaction rationale
            [(sent1 group), (sent2 group)],   # 2nd interaction rationale
            ...
        ]
    pred_rationales: same format as gt_rationales.
    """
    if isinstance(pred_rationales, dict):
        pred_rationales = {eval(k): v for k, v in pred_rationales.items()}
        sorted_rationales = sorted(pred_rationales.items(),
                                   key=lambda x: x[1],
                                   reverse=True)
        pred_rationales = [rationale for rationale, _ in sorted_rationales]
    precision = []
    for p in pred_rationales[:topk]:
        if skip_intra_rationale and (len(p[0]) == 0 or len(p[1]) == 0):
            # if the explanation is not cross-sentence, continue
            continue
        similarities = []
        for g in gt_rationales:
            similarities.append(compute_f1(p[0], g[0])[-1] * compute_f1(p[1], g[1])[-1])
        precision.append(max(similarities))

    if len(precision) == 0:
        return 0
    if max_only:
        return max(precision)
    return sum(precision) / len(precision)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='bert-base')
    parser.add_argument('--data_root', type=str, default='data/e-SNLI')
    parser.add_argument('--mode', type=str, default='test', choices=['dev', 'test'])
    parser.add_argument('--how', type=str, default='union', choices=['vote', 'union'])
    parser.add_argument('--explainer', type=str, default='arch')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--baseline_token', type=str, default='[MASK]')
    parser.add_argument('--metric',
                        type=str,
                        default='token_f1',
                        choices=['token_f1', 'interaction_f1', 'interaction_f1-max'])
    parser.add_argument('--do_cross_merge', action='store_true')
    parser.add_argument('--skip_neutral', action='store_true')
    parser.add_argument('--only_correct', action='store_true')
    parser.add_argument('--old_format', action='store_true')
    parser.add_argument('--test_annotator', type=int, default=-1)
    parser.add_argument('--save_path', type=str, default='eval_results')
    args = parser.parse_args()

    config = load_pretrained_config(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config['model_card'])
    # NOTE: tokenizing the rationales, since the explanations are also on subwords.
    # NOTE: not needed, since we merge token in process_tokens.
    data = load_df(args.data_root,
                   args.how,
                   mode=args.mode,
                   rationale_format=args.metric.split('_')[0])
    sent, gt_rationale, label = data
    if isinstance(gt_rationale, tuple) and len(gt_rationale) == 2:
        gt_rationale = list(zip(*gt_rationale))
    data = (gt_rationale, label)

    explanation_fname = (f'explanations/{args.model_name}_{args.explainer}'
                         f'_{args.mode}_BT={args.baseline_token}')
    if args.test_annotator > 0:
        args.only_correct = False
        explanation_fname = f'explanations/annotator{args.test_annotator}'
    if args.old_format:
        if args.metric == 'token_f1':
            # load from explanations that have token rationales:
            # format:
            # {
            #     "pred_label": "contradiction",
            #     "premise_rationales": [
            #         "choir",
            #         "songs"
            #     ],
            #     "hypothesis_rationales": [
            #         "has",
            #         "cracks",
            #         "the",
            #         "ceiling"
            #     ]
            # }
            explanation_fname += '_token'

        elif 'interaction_f1' in args.metric:
            # load from explanation that have interaction rationales:
            # format:
            # {
            #      "pred_label": "contradiction",
            #      "pred_rationales": [
            #           [("choir", "song"), ("ceiling")],
            #           [("choir", "song"), ("cracks")]
            #      ]
            # }
            explanation_fname += '_interaction'
    if args.do_cross_merge:
        explanation_fname += '_X'

    print('loading from:', explanation_fname + '.json')
    with open(explanation_fname + '.json', 'r') as f:
        explanations = json.load(f)

    run(data, explanations, args)


def run(data, explanations, args, verbose=False):
    scores = []
    indices = []
    pbar = enumerate(zip(*data, explanations))
    if verbose:
        pbar = tqdm(pbar, total=len(explanations))
    for i, (gt_rationale, label, exp) in pbar:
        if args.skip_neutral:
            if label == 'neutral':
                continue
        if args.only_correct:
            if label != exp['pred_label']:
                continue
        if args.metric == 'token_f1':
            if 'pred_rationales' in exp:
                scores.append([
                    compute_token_f1_score(gt_rationale[0],
                                           gt_rationale[1],
                                           pred_rationales=exp['pred_rationales'],
                                           topk=k) for k in range(1, args.topk + 1)
                ])
            else:
                scores.append([
                    compute_token_f1_score(gt_rationale[0],
                                           gt_rationale[1],
                                           pred_rationale1=exp['premise_rationales'],
                                           pred_rationale2=exp['hypothesis_rationales'],
                                           topk=k) for k in range(1, args.topk + 1)
                ])
        elif 'interaction_f1' in args.metric:
            if len(gt_rationale) == 0:  # this might happen with vote
                continue
            scores.append([
                interaction_f1(gt_rationale,
                               exp['pred_rationales'],
                               skip_intra_rationale=False,
                               max_only='max' in args.metric,
                               topk=k) for k in range(1, args.topk + 1)
            ])
            indices.append(i)
        else:
            raise NotImplementedError

    scores = np.array(scores)

    if args.metric == 'token_f1':
        p_at_1 = scores[:, 0, [0, 3, 6]].mean(0) * 100  # [3,]
        map_scores = scores[:, :, [0, 3, 6]].mean(0).mean(0) * 100  # [3,]
        mean_scores = scores.mean(0) * 100  # [k, 9]

        p_at_1 = np.pad(p_at_1[None, :], [[0, args.topk - 1], [0, 0]])  # [k, 3]
        map_scores = np.pad(map_scores[None, :], [[0, args.topk - 1], [0, 0]])  # [k, 3]

        all_scores = np.concatenate([mean_scores, p_at_1, map_scores], axis=1)  # [k, 15]

        results = pd.DataFrame(
            all_scores,
            columns=[
                f'{t}_{metric}' for t in ['premise', 'hypothesis', 'total']
                for metric in ['precision', 'recall', 'f1']
            ] + [
                f'{t}_{metric}' for t in ['p@1', 'map']
                for metric in ['premise', 'hypothesis', 'total']
            ],
            index=range(1, args.topk + 1))
    elif 'interaction_f1' in args.metric:
        results = pd.DataFrame(scores.mean(0), columns=[f'interaction_f1'])
        all_results = pd.DataFrame(
            scores,
            columns=[f'interaction_f1_top{k}' for k in range(1, args.topk + 1)],
            index=indices)

    save_name = f'{args.save_path}/{args.model_name}_{args.explainer}_{args.metric}_{args.mode}_{args.how}_{args.topk}_BT={args.baseline_token}'
    print(save_name)
    print(results)
    if args.test_annotator > 0:
        save_name = f'{args.save_path}/annotator{args.test_annotator}_{args.metric}_{args.mode}_{args.how}'

    if args.do_cross_merge:
        save_name += '_X'
    if args.skip_neutral:
        save_name += '_skip-neutral'
    if args.only_correct:
        save_name += '_only-correct'
    results.to_csv(f'{save_name}.csv', index=False)
    if 'interaction_f1' in args.metric:
        all_results.to_csv(f'{save_name}_all.csv')


if __name__ == "__main__":
    main()