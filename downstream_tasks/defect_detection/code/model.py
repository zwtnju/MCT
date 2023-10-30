# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        classifier_dropout = config.classifier_dropout if config.classifier_dropout is not None else \
            config.hidden_dropout_prob

        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        if "bert" in args.model_type:
            self.classifier = RobertaClassificationHead(config)
        elif "bart" in args.model_type:
            self.classifier = PLBartClassificationHead(
                config.d_model,
                config.d_model,
                config.num_labels,
                config.classifier_dropout,
            )
        elif "gpt" or "codegen2" in args.model_type:
            if "gpt-neo" == args.model_type:
                self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            else:
                self.classifier = nn.Linear(config.n_embd, config.num_labels, bias=False)
        else:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, input_ids=None, labels=None):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        if self.args.model_type == "roberta":
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]

        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                   labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs['decoder_hidden_states'][-1]
            eos_mask = input_ids.eq(self.config.eos_token_id)

            if len(torch.unique(eos_mask.sum(1))) > 1:
                raise ValueError("All examples must have the same number of <eos> tokens.")
            outputs = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[:, -1, :]

        logits = self.classifier(outputs)
        prob = torch.sigmoid(logits)
        if labels is not None:
            labels = labels.float()
            loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
            loss = -loss.mean()
            return loss, prob
        else:
            return prob
