import torch
from torch import nn
from transformers import AutoModel, AutoModelForSequenceClassification

from loss import length_regularizer, continuity_regularizer
from utils import freeze, unfreeze


class Encoder(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.head = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids).last_hidden_state  # [N, T, E]
        return torch.sigmoid(self.head(out).squeeze(-1))  # [N, T]


class Decoder(nn.Module):

    def __init__(self, model_name, mask_id=103):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                        num_labels=3)
        self.mask_id = mask_id

    def forward(self, **selected_inputs):
        """
        inputs:
            input_ids: N, T
            attention_mask: N, T
            token_type_ids: N, T
        """
        return self.model(**selected_inputs).logits


class VerificationNetwork(nn.Module):

    def __init__(self,
                 model_name,
                 mask_id=103,
                 mask_threshold=0.5,
                 relevance_threshold=0.1,
                 reg_strengths=(2e-4, 4e-4)):
        super().__init__()
        self.encoder = Encoder(model_name)
        self.decoder = Decoder(model_name, mask_id=mask_id)
        self.mask_threshold = mask_threshold
        self.mask_id = mask_id

        self.score_cache = None
        self.z_cache = None

        self.loss = nn.CrossEntropyLoss(reduction='none')
        self.z_loss = nn.BCELoss(reduction='none')
        self.reg_strengths = reg_strengths
        self.relevance_threshold = relevance_threshold

    def select_inputs(self, input_ids, attention_mask, token_type_ids, selection_mask):
        """
        slection_mask: N, T. Binary mask of 1 and 0

        NOTE: do not process token_type_ids here: no need actually.
        """
        # do not change [PAD] to [MASK]
        input_ids = torch.where((selection_mask == 0) & (attention_mask == 1),
                                self.mask_id, input_ids)
        # mask the [MASK] tokens from attention.
        attention_mask = (attention_mask * selection_mask).long()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
        }

    def forward(self, input_ids, attention_mask, token_type_ids, token_only=False):
        selection_score = self.encoder(input_ids, attention_mask, token_type_ids)
        if self.training:
            selection_mask = torch.bernoulli(selection_score)
        else:
            selection_mask = (selection_score >= self.mask_threshold).float()
        self.score_cache = selection_score
        self.z_cache = selection_mask
        if self.training:
            self.z_cache.retain_grad()
        selected_inputs = self.select_inputs(input_ids, attention_mask, token_type_ids,
                                             selection_mask)
        if token_only:
            return selected_inputs

        scores = self.decoder(**selected_inputs)
        return scores, selected_inputs

    def calc_loss(self, x, y):
        ce_loss = self.loss(x, y)  # [N, ]
        reg_loss = (self.reg_strengths[0] * length_regularizer(self.z_cache) +
                    self.reg_strengths[1] * continuity_regularizer(self.z_cache))
        loss_term = ce_loss + reg_loss
        loss_term = loss_term.mean()
        return loss_term

    def backward(self, loss_term):
        # decoder backward
        loss_term.backward(retain_graph=True)

        # encoer backward
        encoder_loss = (
            loss_term *
            self.z_loss(self.score_cache, self.z_cache.detach())).sum(-1).mean(0)
        freeze(self.decoder)
        encoder_loss.backward(retain_graph=True)
        self.score_cache.backward(self.z_cache.grad)
        unfreeze(self.decoder)

    def check_handshake(self, input_ids, attention_mask, token_type_ids):
        """
        return binary mask of size [N,] indicating which batch do not have
        handshake
        """
        selection_score = self.encoder(input_ids, attention_mask, token_type_ids)
        selection_mask = selection_score >= self.mask_threshold
        selected_inputs = self.select_inputs(input_ids, attention_mask, token_type_ids,
                                             selection_mask)

        s_selection_score = self.encoder(**selected_inputs)
        s_selection_mask = s_selection_score >= self.mask_threshold

        return (selection_mask == s_selection_mask).all(dim=1)

    def infer(self, **inputs):
        """
        return: [N, T, 3].
            0 - not relevant, 1 - clearly relevant, 2 - selected don't know

        TODO: find clearly relevant tokens
        """
        selected_inputs = self(**inputs, token_only=True)  # [N, T]
        raise NotImplementedError


class ExplanationEvaluator:

    def __init__(self, explainer, verification_net):
        self.explainer = explainer
        self.verification_net = verification_net

    def verify(self, inputs):
        # TODO: implement metric 2 and 3
        explanation = self.explainer(**inputs)  # [N, T]
        top1 = explanation.argmax(-1)  # [N,]

        tok_distribution = self.verification_net.infer(**inputs)  # [N, T]

        # eval metric 1: amount of the most important token is found in Nx,
        # the set of guaranteed not important tokens.
        wrong_top1_percent = (tok_distribution[range(tok_distribution.shape[0]),
                                               top1] == 0).long().mean()

        # eval metric 2:

        # eval metric 3: