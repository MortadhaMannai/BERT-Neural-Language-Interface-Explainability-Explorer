import torch
from transformers import BertForSequenceClassification


class BertWrapperTorch:

    def __init__(self, model, device, merge_logits=False):
        """
        TODO: make the model be anything
        """
        assert isinstance(model, BertForSequenceClassification)
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.merge_logits = merge_logits

    @torch.no_grad()
    def get_embedding(self, **inputs):
        if not isinstance(inputs['input_ids'], torch.Tensor):
            inputs = {k: torch.LongTensor(v).to(self.device) for k, v in inputs.items()}
        inputs.pop('attention_mask', None)
        embedding = self.model.bert.embeddings(**inputs)
        return embedding

    @torch.no_grad()
    def get_predictions(self, batch_embedding):
        # NOTE: this works only when the model is BertForSequenceClassification
        encoder_outputs = self.model.bert.encoder(batch_embedding,
                                                  output_hidden_states=True,
                                                  return_dict=False)
        sequence_output = encoder_outputs[0]
        pooled_output = self.model.bert.pooler(sequence_output)
        logits = self.model.classifier(pooled_output)
        return logits.cpu()

    def __call__(self, return_embedding=False, **inputs):
        batch_embeddings = self.get_embedding(**inputs)
        batch_predictions = self.get_predictions(batch_embeddings)
        if self.merge_logits:
            batch_predictions2 = (batch_predictions[:, 1] - batch_predictions[:, 0])
            batch_predictions = batch_predictions2.unsqueeze(1)

        outs = batch_predictions.numpy()
        if return_embedding:
            outs = (batch_predictions.numpy(), batch_embeddings.cpu().numpy())

        return outs
