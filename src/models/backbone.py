from transformers import BertModel, BertPreTrainedModel
from torch import nn


class Encoder(BertPreTrainedModel):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

    def forward(self, input_ids, mask):
        output = self.bert(input_ids, attention_mask=mask)
        last_hidden = output[0]
        return last_hidden


class Backbone(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder.from_pretrained(args.PRETRAINED_MODEL_NAME)

    def forward(self, input_ids, mask):
        last_hidden = self.encoder(input_ids, mask)
        return last_hidden


