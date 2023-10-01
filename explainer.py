import numpy as np
from itertools import chain, product


class Explainer:

    def __init__(
        self,
        model,
        input=None,
        baseline=None,
        data_xformer=None,
        output_indices=0,
        batch_size=20,
        verbose=False,
    ):
        """
        self.input : boolean array of all 1
        self.baseline: boolean array of all 0
        """

        input, baseline = self.arg_checks(input, baseline, data_xformer)

        self.model = model
        self.input = np.squeeze(input)
        self.baseline = np.squeeze(baseline)
        self.data_xformer = data_xformer
        self.output_indices = output_indices
        self.batch_size = batch_size
        self.verbose = verbose

    def arg_checks(self, input, baseline, data_xformer):
        if (input is None) and (data_xformer is None):
            raise ValueError("Either input or data xformer must be defined")

        if input is not None and baseline is None:
            raise ValueError("If input is defined, the baseline must also defined")

        if data_xformer is not None and input is None:
            input = np.ones(data_xformer.num_features).astype(bool)
            baseline = np.zeros(data_xformer.num_features).astype(bool)
        return input, baseline

    def verbose_iterable(self, iterable):
        if self.verbose:
            from tqdm import tqdm

            return tqdm(iterable)
        else:
            return iterable

    def batch_set_inference(self,
                            set_indices,
                            context,
                            insertion_target,
                            include_context=False):
        """
        Creates archipelago type data instances and runs batch inference on them
        All "sets" are represented as tuples to work as keys in dictionaries
        """

        num_batches = int(np.ceil(len(set_indices) / self.batch_size))

        scores = {}
        for b in self.verbose_iterable(range(num_batches)):
            batch_sets = set_indices[b * self.batch_size:(b + 1) * self.batch_size]
            data_batch = []
            for index_tuple in batch_sets:
                new_instance = context.copy()
                for i in index_tuple:
                    new_instance[i] = insertion_target[i]

                if self.data_xformer is not None:
                    new_instance = self.data_xformer(new_instance)

                data_batch.append(new_instance)

            if include_context and b == 0:
                # include context as the last item of the first batch
                if self.data_xformer is not None:
                    data_batch.append(self.data_xformer(context))
                else:
                    data_batch.append(context)

            preds = self.model(**self.data_xformer.process_batch_ids(data_batch))

            for c, index_tuple in enumerate(batch_sets):
                scores[index_tuple] = preds[c, self.output_indices]
            if include_context and b == 0:
                context_score = preds[-1, self.output_indices]

        output = {"scores": scores}
        if include_context and num_batches > 0:
            output["context_score"] = context_score
        return output


class Archipelago(Explainer):

    def __init__(
        self,
        model,
        input=None,
        baseline=None,
        data_xformer=None,
        output_indices=0,
        batch_size=20,
        verbose=False,
    ):
        super().__init__(
            model,
            input=input,
            baseline=baseline,
            data_xformer=data_xformer,
            output_indices=output_indices,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.inter_sets = None
        self.main_effects = None
        self.sep_pos = self.data_xformer.sep_pos

    def reset(self):
        self.inter_sets = None
        self.main_effects = None

    def archattribute(self, set_indices):
        """
        Gets archipelago attributions of index sets
        """
        if not set_indices:
            return dict()
        scores = self.batch_set_inference(set_indices,
                                          self.baseline,
                                          self.input,
                                          include_context=True)
        set_scores = scores["scores"]
        baseline_score = scores["context_score"]
        for index_tuple in set_scores:
            set_scores[index_tuple] -= baseline_score
        return set_scores

    def archdetect(
        self,
        get_main_effects=True,
        get_pairwise_effects=True,
        single_context=False,
        weights=[0.5, 0.5],
        use_embedding=False,
    ):
        """
        Detects interactions and sorts them
        Optional: gets archipelago main effects and/or pairwise effects from function reuse
        "Effects" are archattribute scores
        """
        if self.data_xformer is not None and use_embedding:
            _, input_emb = self.model(**{
                k: np.expand_dims(v, 0) for k, v in self.data_xformer.input.items()
            },
                                      return_embedding=True)
            _, base_emb = self.model(**self.data_xformer.get_baseline_inputs(),
                                     return_embedding=True)
            input_emb, base_emb = input_emb[0], base_emb[0]  # remove batch dim: bs=1
        else:
            input_emb, base_emb = None, None

        search_a = self.search_feature_sets(
            self.baseline,
            self.input,
            context_embedding=base_emb,
            insertion_target_embedding=input_emb,
            get_main_effects=get_main_effects,
            get_pairwise_effects=get_pairwise_effects,
        )
        inter_a = search_a["interactions"]

        # notice that input and baseline have swapped places in the arg list
        search_b = self.search_feature_sets(
            self.input,
            self.baseline,
            context_embedding=input_emb,
            insertion_target_embedding=base_emb,
        )
        inter_b = search_b["interactions"]

        inter_strengths = {}
        for pair in inter_a:
            if single_context:
                inter_strengths[pair] = inter_b[pair]**2
            else:
                inter_strengths[pair] = (weights[1] * inter_a[pair]**2 +
                                         weights[0] * inter_b[pair]**2)
        sorted_scores = sorted(inter_strengths.items(),
                               key=lambda kv: kv[1],
                               reverse=True)

        output = {"interactions": sorted_scores}
        for key in search_a:
            if key not in output:
                output[key] = search_a[key]
        return output

    def explain(self,
                top_k=None,
                separate_effects=False,
                use_embedding=False,
                do_cross_merge=False,
                get_cross_effects=False):
        if get_cross_effects:
            do_cross_merge = False
        if do_cross_merge:
            individual_topk = top_k
            top_k = None
        if (self.inter_sets is None) or (self.main_effects is None):
            detection_dict = self.archdetect(get_pairwise_effects=False,
                                             use_embedding=use_embedding)
            inter_strengths = detection_dict["interactions"]
            self.main_effects = detection_dict["main_effects"]
            self.inter_sets, _ = zip(*inter_strengths)

        if isinstance(top_k, int):
            thresholded_inter_sets = self.inter_sets[:top_k]
        elif top_k is None:
            thresholded_inter_sets = self.inter_sets
        else:
            raise ValueError("top_k must be int or None")

        if get_cross_effects or do_cross_merge:
            pre_set, cross_set, hyp_set = [], [], []
            for inter_set in thresholded_inter_sets:
                if inter_set[0] < self.sep_pos and inter_set[1] < self.sep_pos:
                    pre_set.append([inter_set, {}])
                elif inter_set[0] < self.sep_pos and inter_set[1] > self.sep_pos:
                    cross_set.append([inter_set, {}])
                else:
                    hyp_set.append([inter_set, {}])
            if do_cross_merge:
                inter_sets_merged = cross_merge(pre_set[:individual_topk],
                                                cross_set[:individual_topk],
                                                hyp_set[:individual_topk])
            else:  # get_cross_effects: just get cross_set
                inter_sets_merged = [pair for pair, _ in cross_set]
        else:
            inter_sets_merged = merge_overlapping_sets(thresholded_inter_sets)
        inter_effects = self.archattribute(inter_sets_merged)

        if separate_effects:
            return inter_effects, self.main_effects

        if do_cross_merge:
            merged_indices = set(inter_effects.keys())
            for idx in self.main_effects.keys():
                if idx not in chain.from_iterable(inter_effects.keys()):
                    merged_indices.add((idx,))
        elif get_cross_effects:
            merged_indices = set(inter_effects.keys())
        else:
            merged_indices = merge_overlapping_sets(
                set(self.main_effects.keys()) | set(inter_effects.keys()))
        merged_explanation = dict()
        for s in merged_indices:
            if s in inter_effects:
                merged_explanation[s] = inter_effects[s]
            elif s[0] in self.main_effects:
                assert len(s) == 1
                merged_explanation[s] = self.main_effects[s[0]]
            else:
                raise ValueError(
                    "Error: index should have been in either main_effects or inter_effects"
                )
        return merged_explanation

    def search_feature_sets(
        self,
        context,
        insertion_target,
        context_embedding=None,
        insertion_target_embedding=None,
        get_interactions=True,
        get_main_effects=False,
        get_pairwise_effects=False,
    ):
        """
        Gets optional pairwise interaction strengths, optional main effects, and optional pairwise effects
        "Effects" are archattribute scores
        All three options are combined to reuse function calls
        """
        num_feats = context.size
        idv_indices = [(i,) for i in range(num_feats)]

        preds = self.batch_set_inference(idv_indices,
                                         context,
                                         insertion_target,
                                         include_context=True)
        idv_scores, context_score = preds["scores"], preds["context_score"]

        output = {}

        if get_interactions:
            pair_indices = []
            pairwise_effects = {}
            for i in range(num_feats):
                for j in range(i + 1, num_feats):
                    pair_indices.append((i, j))

            preds = self.batch_set_inference(pair_indices, context, insertion_target)
            pair_scores = preds["scores"]

            inter_scores = {}
            for i, j in pair_indices:

                # interaction detection
                if context_embedding is not None and insertion_target_embedding is not None:
                    ell_i = np.linalg.norm(context_embedding[i] -
                                           insertion_target_embedding[i])
                    ell_j = np.linalg.norm(context_embedding[j] -
                                           insertion_target_embedding[j])
                    if ell_i * ell_j == 0:
                        # This means that it is a special token:
                        # the baseline and the input shares the special tokens.
                        continue
                else:
                    ell_i = np.abs(context[i].item() - insertion_target[i].item())
                    ell_j = np.abs(context[j].item() - insertion_target[j].item())
                inter_scores[(i, j)] = (1 / (ell_i * ell_j) *
                                        (context_score - idv_scores[(i,)] -
                                         idv_scores[(j,)] + pair_scores[(i, j)]))

                if get_pairwise_effects:
                    # leverage existing function calls to compute pairwise effects
                    pairwise_effects[(i, j)] = pair_scores[(i, j)] - context_score

            output["interactions"] = inter_scores

            if get_pairwise_effects:
                output["pairwise_effects"] = pairwise_effects

        if get_main_effects:  # leverage existing function calls to compute main effects
            main_effects = {}
            for i in idv_scores:
                main_effects[i[0]] = idv_scores[i] - context_score
            output["main_effects"] = main_effects

        return output


def merge_overlapping_sets(lsts, output_ints=False):
    """Check each number in our arrays only once, merging when we find
    a number we have seen before.

    O(N) mergelists5 solution from https://stackoverflow.com/questions/9110837/python-simple-list-merging-based-on-intersections
    """

    def locatebin(bins, n):
        """
        Find the bin where list n has ended up: Follow bin references until
        we find a bin that has not moved.
        """
        while bins[n] != n:
            n = bins[n]
        return n

    data = []
    for lst in lsts:
        if type(lst) not in {list, set, tuple}:
            lst = {lst}
        data.append(set(lst))

    bins = list(range(len(data)))  # Initialize each bin[n] == n
    nums = dict()

    sets = []
    for lst in lsts:
        if type(lst) not in {list, set, tuple}:
            lst = {lst}
        sets.append(set(lst))

    for r, row in enumerate(data):
        for num in row:
            if num not in nums:
                # New number: tag it with a pointer to this row's bin
                nums[num] = r
                continue
            else:
                dest = locatebin(bins, nums[num])
                if dest == r:
                    continue  # already in the same bin

                if dest > r:
                    dest, r = r, dest  # always merge into the smallest bin

                data[dest].update(data[r])
                data[r] = None
                # Update our indices to reflect the move
                bins[r] = dest
                r = dest

    # take single values out of sets
    output = []
    for s in data:
        if s:
            if output_ints and len(s) == 1:
                output.append(next(iter(s)))
            else:
                output.append(tuple(sorted(s)))

    return output


class CrossArchipelago(Archipelago):

    def __init__(
        self,
        model,
        input=None,
        baseline=None,
        data_xformer=None,
        output_indices=0,
        batch_size=20,
        verbose=False,
    ):
        super().__init__(
            model,
            input=input,
            baseline=baseline,
            data_xformer=data_xformer,
            output_indices=output_indices,
            batch_size=batch_size,
            verbose=verbose,
        )

    def batch_set_inference(self,
                            set_indices,
                            context,
                            insertion_target,
                            insertion_target2=None,
                            sep_pos=None,
                            include_context=False):
        """
        Creates archipelago type data instances and runs batch inference on them
        All "sets" are represented as tuples to work as keys in dictionaries
        """

        if insertion_target2 is not None:
            assert sep_pos is not None, "sep idx must be also provided!"

        num_batches = int(np.ceil(len(set_indices) / self.batch_size))

        scores = {}
        for b in self.verbose_iterable(range(num_batches)):
            batch_sets = set_indices[b * self.batch_size:(b + 1) * self.batch_size]
            data_batch = []
            for index_tuple in batch_sets:
                new_instance = context.copy()
                for i in index_tuple:
                    if insertion_target2 is not None:
                        insert_from = insertion_target if i < sep_pos else insertion_target2
                        new_instance[i] = insert_from[i]
                    else:
                        new_instance[i] = insertion_target[i]

                if self.data_xformer is not None:
                    new_instance = self.data_xformer(new_instance)

                data_batch.append(new_instance)

            if include_context and b == 0:
                if self.data_xformer is not None:
                    data_batch.append(self.data_xformer(context))
                else:
                    data_batch.append(context)

            preds = self.model(**self.data_xformer.process_batch_ids(data_batch))

            for c, index_tuple in enumerate(batch_sets):
                scores[index_tuple] = preds[c, self.output_indices]
            if include_context and b == 0:
                context_score = preds[-1, self.output_indices]

        output = {"scores": scores}
        if include_context and num_batches > 0:
            output["context_score"] = context_score
        return output

    def search_feature_sets(
        self,
        context,
        insertion_target,
        insertion_target2=None,
        context_embedding=None,
        insertion_target_embedding=None,
        get_interactions=True,
        get_main_effects=False,
        get_pairwise_effects=False,
    ):
        """
        Gets optional pairwise interaction strengths, optional main effects, and optional pairwise effects
        "Effects" are archattribute scores
        All three options are combined to reuse function calls
        """
        num_feats = context.size
        idv_indices = [(i,) for i in range(num_feats)]

        preds = self.batch_set_inference(idv_indices,
                                         context,
                                         insertion_target,
                                         insertion_target2=insertion_target2,
                                         sep_pos=self.sep_pos,
                                         include_context=True)
        idv_scores, context_score = preds["scores"], preds["context_score"]

        output = {}

        if get_interactions:
            pair_indices = []
            pairwise_effects = {}
            for i in range(1, num_feats - 1):
                if i == self.sep_pos:
                    continue
                for j in range(i + 1, num_feats - 1):
                    if j == self.sep_pos:
                        continue
                    pair_indices.append((i, j))

            preds = self.batch_set_inference(pair_indices,
                                             context,
                                             insertion_target,
                                             insertion_target2=insertion_target2,
                                             sep_pos=self.sep_pos)
            pair_scores = preds["scores"]

            inter_scores = {}
            for i, j in pair_indices:

                # interaction detection
                if context_embedding is not None and insertion_target_embedding is not None:
                    ell_i = np.linalg.norm(context_embedding[i] -
                                           insertion_target_embedding[i])
                    ell_j = np.linalg.norm(context_embedding[j] -
                                           insertion_target_embedding[j])
                    if ell_i * ell_j == 0:
                        # This means that it is a special token:
                        # the baseline and the input shares the special tokens.
                        continue
                else:
                    # This is actually setting h1 = h2 = 1
                    ell_i = np.abs(context[i].item() - insertion_target[i].item())
                    ell_j = np.abs(context[j].item() - insertion_target[j].item())

                inter_scores[(i, j)] = (1 / (ell_i * ell_j) *
                                        (context_score - idv_scores[(i,)] -
                                         idv_scores[(j,)] + pair_scores[(i, j)]))
                if inter_scores[(i, j)] == float('inf'):
                    import pdb
                    pdb.set_trace()

                if get_pairwise_effects:
                    # leverage existing function calls to compute pairwise effects
                    pairwise_effects[(i, j)] = pair_scores[(i, j)] - context_score

            output["interactions"] = inter_scores

            if get_pairwise_effects:
                output["pairwise_effects"] = pairwise_effects

        if get_main_effects:  # leverage existing function calls to compute main effects
            main_effects = {}
            for i in idv_scores:
                main_effects[i[0]] = idv_scores[i] - context_score
            output["main_effects"] = main_effects

        return output

    def archdetect(
        self,
        get_main_effects=True,
        get_pairwise_effects=True,
        single_context=False,
        cross_sent_context=True,
        weights=[0.25, 0.25, 0.25, 0.25],
        use_embedding=False,
    ):
        """
        Detects interactions and sorts them
        Optional: gets archipelago main effects and/or pairwise effects from function reuse
        "Effects" are archattribute scores
        """
        if self.data_xformer is not None and use_embedding:
            _, input_emb = self.model(**{
                k: np.expand_dims(v, 0) for k, v in self.data_xformer.input.items()
            },
                                      return_embedding=True)
            _, base_emb = self.model(**self.data_xformer.get_baseline_inputs(),
                                     return_embedding=True)
            input_emb, base_emb = input_emb[0], base_emb[0]  # remove batch dim: bs=1
        else:
            input_emb, base_emb = None, None

        search_a = self.search_feature_sets(
            self.baseline,
            self.input,
            context_embedding=base_emb,
            insertion_target_embedding=input_emb,
            get_main_effects=get_main_effects,
            get_pairwise_effects=get_pairwise_effects,
        )
        inter_a = search_a["interactions"]

        # notice that input and baseline have swapped places in the arg list
        search_b = self.search_feature_sets(
            self.input,
            self.baseline,
            context_embedding=input_emb,
            insertion_target_embedding=base_emb,
        )
        inter_b = search_b["interactions"]

        if cross_sent_context:
            search_p = self.search_feature_sets(
                np.where(
                    np.arange(len(self.input)) <= self.sep_pos, self.input,
                    self.baseline),
                self.baseline,
                insertion_target2=self.input,
                context_embedding=input_emb,
                insertion_target_embedding=base_emb,
            )
            inter_p = search_p["interactions"]

            search_h = self.search_feature_sets(
                np.where(
                    np.arange(len(self.input)) < self.sep_pos, self.baseline, self.input),
                self.input,
                insertion_target2=self.baseline,
                context_embedding=base_emb,
                insertion_target_embedding=input_emb,
            )
            inter_h = search_h["interactions"]

        inter_strengths = {}
        for pair in inter_a:
            strengths = {}
            if single_context:
                strengths['all'] = inter_b[pair]**2
            else:
                strengths['input'] = inter_a[pair]**2
                strengths['base'] = inter_b[pair]**2
                strengths['all'] = (weights[0] * inter_a[pair]**2 +
                                    weights[1] * inter_b[pair]**2)
                if cross_sent_context:
                    strengths['pre'] = inter_p[pair]**2
                    strengths['hyp'] = inter_h[pair]**2
                    strengths['all'] += (weights[2] * inter_p[pair]**2 +
                                         weights[3] * inter_h[pair]**2)
            inter_strengths[pair] = strengths
        sorted_scores = sorted(inter_strengths.items(),
                               key=lambda kv: kv[1]['all'],
                               reverse=True)

        output = {"interactions": sorted_scores}
        for key in search_a:
            if key not in output:
                output[key] = search_a[key]
        return output

    def explain(self,
                top_k=None,
                separate_effects=False,
                use_embedding=False,
                cross_sent_context=True,
                do_cross_merge=False,
                get_cross_effects=False):
        """
        get_cross_effects is different from all other file: just don't merge
        at the end.
        """
        if get_cross_effects:
            do_cross_merge = False
        if do_cross_merge:
            individual_topk = top_k
            top_k = None
        if (self.inter_sets is None) or (self.main_effects is None):
            detection_dict = self.archdetect(get_pairwise_effects=False,
                                             use_embedding=use_embedding,
                                             cross_sent_context=cross_sent_context)
            inter_strengths = detection_dict["interactions"]
            self.main_effects = detection_dict["main_effects"]
            self.inter_sets = inter_strengths

        if cross_sent_context:
            valid_inter_sets = []
            for inter_set in self.inter_sets:
                if inter_set[0][0] < self.sep_pos and inter_set[0][1] < self.sep_pos:
                    if inter_set[1]['hyp'] > inter_set[1]['pre']:
                        valid_inter_sets.append(inter_set)
                elif inter_set[0][0] < self.sep_pos and inter_set[0][1] > self.sep_pos:
                    # TODO
                    valid_inter_sets.append(inter_set)
                else:
                    if inter_set[1]['pre'] > inter_set[1]['hyp']:
                        valid_inter_sets.append(inter_set)
            self.inter_sets = sorted(valid_inter_sets,
                                     key=lambda x: x[1]['all'],
                                     reverse=True)
        if isinstance(top_k, int):
            thresholded_inter_sets = self.inter_sets[:top_k]
        elif top_k is None:
            thresholded_inter_sets = self.inter_sets
        else:
            raise ValueError("top_k must be int or None")

        if get_cross_effects or do_cross_merge:
            pre_set, cross_set, hyp_set = [], [], []
            for inter_set in thresholded_inter_sets:
                if inter_set[0][0] < self.sep_pos and inter_set[0][1] < self.sep_pos:
                    pre_set.append(inter_set)
                elif inter_set[0][0] < self.sep_pos and inter_set[0][1] > self.sep_pos:
                    cross_set.append(inter_set)
                else:
                    hyp_set.append(inter_set)
            if do_cross_merge:
                inter_sets_merged = cross_merge(pre_set[:individual_topk],
                                                cross_set[:individual_topk],
                                                hyp_set[:individual_topk])
            else:  # get_cross_effects: just get cross_set
                inter_sets_merged = [pair for pair, _ in cross_set]
        else:
            inter_sets_merged = merge_overlapping_sets(
                [pair for pair, _ in thresholded_inter_sets])
        inter_effects = self.archattribute(inter_sets_merged)

        if separate_effects:
            return inter_effects, self.main_effects

        if do_cross_merge:
            merged_indices = set(inter_effects.keys())
            for idx in self.main_effects.keys():
                if idx not in chain.from_iterable(inter_effects.keys()):
                    merged_indices.add((idx,))
        elif get_cross_effects:
            merged_indices = set(inter_effects.keys())
        else:
            merged_indices = merge_overlapping_sets(
                set(self.main_effects.keys()) | set(inter_effects.keys()))
        merged_explanation = dict()
        for s in merged_indices:
            if s in inter_effects:
                merged_explanation[s] = inter_effects[s]
            elif s[0] in self.main_effects:
                assert len(s) == 1
                merged_explanation[s] = self.main_effects[s[0]]
            else:
                raise ValueError(
                    "Error: index should have been in either main_effects or inter_effects"
                )
        return merged_explanation


# def cross_merge(pre_set, cross_set, hyp_set, sum_strength=False):
#     """
#     *_set:  [(idx1, idx2), {pre: score, hyp: score, all: score}]
#     sum_strength: get sum of merged interaction strengths. if True,
#         {(interaction): strength}
#         if false, [(interaction)]
#     """
#     if sum_strength:
#         merged = {}
#     else:
#         merged = []
#     used = {'pre': set(), 'hyp': set()}
#     for cross in cross_set:
#         feature_group = list(cross[0])
#         if sum_strength:
#             feature_strengths = cross[1]['all']
#         for i, pre in enumerate(pre_set):
#             if cross[0][0] in pre[0] and i not in used['pre']:
#                 feature_group = sorted(set(feature_group + list(pre[0])))
#                 if sum_strength:
#                     feature_strengths += pre[1]['all']
#                 used['pre'].add(i)
#                 break

#         for j, hyp in enumerate(hyp_set):
#             if cross[0][1] in hyp[0] and j not in used['hyp']:
#                 feature_group = sorted(set(feature_group + list(hyp[0])))
#                 if sum_strength:
#                     feature_strengths += hyp[1]['all']
#                 used['hyp'].add(j)
#                 break
#         if sum_strength:
#             merged[tuple(feature_group)] = feature_strengths
#         else:
#             merged.append(tuple(feature_group))

#     return merged


def cross_merge(pre_set, cross_set, hyp_set, sum_strength=False):
    """
    *_set:  [(idx1, idx2), {pre: score, hyp: score, all: score}]
    sum_strength: get sum of merged interaction strengths. if True,
        {(interaction): strength}
        if false, [(interaction)]
    """
    pre_set, cross_set, hyp_set = pre_set.copy(), cross_set.copy(), hyp_set.copy()
    pre_set = list(list(zip(*pre_set))[0])
    cross_set = list(list(zip(*cross_set))[0])
    hyp_set = list(list(zip(*hyp_set))[0])
    merged = []
    while cross_set:
        inter_set = cross_set.pop(0)
        if len(cross_set) == 0:
            merged.append(tuple(inter_set))
            return merged
        other_inter = cross_set.pop(0)

        evidence_candidates = list(product(inter_set, other_inter))
        num_evidences = len(evidence_candidates)
        evidence_found = 0

        for evidence in evidence_candidates:
            evidence = tuple(sorted(evidence))
            if (evidence[0] == evidence[1] or set(inter_set).issubset(set(evidence)) or
                    evidence == other_inter):
                ## (i, j) + (j, k) -> should find (i, k) only
                num_evidences -= 1
            elif evidence in pre_set:
                evidence_found += 1
                pre_set.remove(evidence)
            elif evidence in hyp_set:
                evidence_found += 1
                hyp_set.remove(evidence)
            elif evidence in cross_set:
                evidence_found += 1
                cross_set.remove(evidence)
        if evidence_found >= num_evidences / 2:
            inter_set = tuple(sorted(set(inter_set + other_inter)))
            cross_set.insert(0, inter_set)
        else:
            merged.append(tuple(inter_set))
            cross_set.insert(0, tuple(other_inter))
    return merged