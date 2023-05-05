# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, encoder, mode, tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.mode = mode
        self.tokenizer = tokenizer

    def forward(self, code_inputs=None, nl_inputs=None):
        if code_inputs is not None:
            if self.mode == "codebert":
                outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(self.tokenizer.pad_token_id))[0]
            else:
                decoder_input_ids = code_inputs
                outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(self.tokenizer.pad_token_id),
                                       decoder_input_ids=decoder_input_ids,
                                       decoder_attention_mask=decoder_input_ids.ne(self.tokenizer.pad_token_id))[0]
            outputs = (outputs * code_inputs.ne(self.tokenizer.pad_token_id)[:, :, None]).sum(1) / code_inputs.ne(self.tokenizer.pad_token_id).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            if self.mode == "codebert":
                outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(self.tokenizer.pad_token_id))[0]
            else:
                decoder_input_ids = nl_inputs
                outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(self.tokenizer.pad_token_id),
                                       decoder_input_ids=decoder_input_ids,
                                       decoder_attention_mask=decoder_input_ids.ne(self.tokenizer.pad_token_id))[0]
            outputs = (outputs * nl_inputs.ne(self.tokenizer.pad_token_id)[:, :, None]).sum(1) / nl_inputs.ne(self.tokenizer.pad_token_id).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
