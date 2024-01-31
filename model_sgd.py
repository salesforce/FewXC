import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import (
    RobertaPreTrainedModel, 
    XLMRobertaModel
    )

import numpy as np


class PrefixEncoder(torch.nn.Module):
    r'''
    The torch.nn model to encode the prefix

    Input shape: (batch-size, prefix-length)

    Output shape: (batch-size, prefix-length, 2*layers*hidden)
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(config.pre_seq_len, config.num_hidden_layers * 2 * config.hidden_size)

    def forward(self, prefix: torch.Tensor):
        past_key_values = self.embedding(prefix)
        return past_key_values


"""
Intent classification will be applied using NLI style classification
"""
class XLMR4SGD(RobertaPreTrainedModel):
    def __init__(self, config, model_name_or_path, freeze_backbone = False):
        super().__init__(config)

        self.backbone_model = XLMRobertaModel.from_pretrained(model_name_or_path)

        if freeze_backbone:
            for param in self.backbone_model.parameters():
                param.requires_grad = False

        # Predict intent via NLI-style intent classification
        self.intent_cls = SequenceClassificationHead(hidden_size = self.backbone_model.config.hidden_size, 
                                                    dropout_prob = self.backbone_model.config.hidden_dropout_prob, 
                                                    num_labels = 2)
        """
        self.requested_slot_cls
            input: (domain-slot pair, utterance), output: none, dontcare, prediction
                - none: the domain-slot pair is not mentioned
                - dontcare: user can accept any value for this slot
                - prediction: the slot value prediction module should predict a value
        """
        self.requested_slot_cls = SequenceClassificationHead(hidden_size = self.backbone_model.config.hidden_size, 
                                                    dropout_prob = self.backbone_model.config.hidden_dropout_prob, 
                                                    num_labels = 3)

        self.slot_value_categorical_cls = SequenceClassificationHead(hidden_size = self.backbone_model.config.hidden_size, 
                                                    dropout_prob = self.backbone_model.config.hidden_dropout_prob, 
                                                    num_labels = 2)

        # Non-categorical slot value prediction is framed as binary token level classification for start and end position
        self.slot_value_noncategorical_cls = TokenClassificationHead(hidden_size = self.backbone_model.config.hidden_size, 
                                                                    dropout_prob = self.backbone_model.config.hidden_dropout_prob,
                                                                    num_labels = 2)

    def forward(self, features, task, labels = None, start_positions = None, end_positions = None, device = 'cpu'):
        for k in features:
            features[k] = features[k].to(device)
        input_ids, attention_mask = features['input_ids'], features['attention_mask']
        outputs = self.backbone_model(input_ids, attention_mask)
        if task == 'intent':
            return self.intent_cls(outputs, labels)
        elif task == 'slot_gate':
            return self.requested_slot_cls(outputs, labels)
        elif task == 'slot_value_categorical':
            return self.slot_value_categorical_cls(outputs, labels)
        elif task == 'slot_value_noncategorical':
            return self.slot_value_noncategorical_cls(outputs, start_positions, end_positions)
        else:
            raise Exception(f"Task: {task} is not supported!")



class XLMR4SGDPrefix(RobertaPreTrainedModel):
    def __init__(self, config, model_name_or_path):
        super().__init__(config)

        self.backbone_model = XLMRobertaModel.from_pretrained(model_name_or_path)
        #self.backbone_model = XLMRobertaModel(config)
        for param in self.backbone_model.parameters():
                param.requires_grad = False

        self.pre_seq_len = config.pre_seq_len
        self.n_layer = config.num_hidden_layers
        self.n_head = config.num_attention_heads
        self.n_embd = config.hidden_size // config.num_attention_heads

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

        self.prefix_tokens = torch.arange(self.pre_seq_len).long()
        self.prefix_encoder = PrefixEncoder(config)        

        # Predict intent via NLI-style intent classification
        self.intent_cls = SequenceClassificationHead(hidden_size = self.backbone_model.config.hidden_size, 
                                                    dropout_prob = self.backbone_model.config.hidden_dropout_prob, 
                                                    num_labels = 2)
        """
        self.requested_slot_cls
            input: (domain-slot pair, utterance), output: none, dontcare, prediction
                - none: the domain-slot pair is not mentioned
                - dontcare: user can accept any value for this slot
                - prediction: the slot value prediction module should predict a value
        """
        self.requested_slot_cls = SequenceClassificationHead(hidden_size = self.backbone_model.config.hidden_size, 
                                                    dropout_prob = self.backbone_model.config.hidden_dropout_prob, 
                                                    num_labels = 3)

        self.slot_value_categorical_cls = SequenceClassificationHead(hidden_size = self.backbone_model.config.hidden_size, 
                                                    dropout_prob = self.backbone_model.config.hidden_dropout_prob, 
                                                    num_labels = 2)

        # Non-categorical slot value prediction is framed as binary token level classification for start and end position
        self.slot_value_noncategorical_cls = TokenClassificationHead(hidden_size = self.backbone_model.config.hidden_size, 
                                                                    dropout_prob = self.backbone_model.config.hidden_dropout_prob,
                                                                    num_labels = 2)
    
    def get_prompt(self, batch_size):
        prefix_tokens = self.prefix_tokens.unsqueeze(0).expand(batch_size, -1).to(self.backbone_model.device)
        past_key_values = self.prefix_encoder(prefix_tokens)
        past_key_values = past_key_values.view(
            batch_size,
            self.pre_seq_len,
            self.n_layer * 2,
            self.n_head,
            self.n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
        return past_key_values


    def forward(self, features, task, labels = None, start_positions = None, end_positions = None, device = 'cpu'):
        for k in features:
            features[k] = features[k].to(device)
 
        input_ids, attention_mask = features['input_ids'], features['attention_mask']
         
        batch_size = input_ids.shape[0]
        past_key_values = self.get_prompt(batch_size=batch_size)
        prefix_attention_mask = torch.ones(batch_size, self.pre_seq_len).to(self.backbone_model.device)
        attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        outputs = self.backbone_model(input_ids, attention_mask, past_key_values=past_key_values)

        if task == 'intent':
            return self.intent_cls(outputs, labels)
        elif task == 'slot_gate':
            return self.requested_slot_cls(outputs, labels)
        elif task == 'slot_value_categorical':
            return self.slot_value_categorical_cls(outputs, labels)
        elif task == 'slot_value_noncategorical':
            return self.slot_value_noncategorical_cls(outputs, start_positions, end_positions)
        else:
            raise Exception(f"Task: {task} is not supported!")




class SequenceClassificationHead(nn.Module):
    """
    Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py
    """
    def __init__(self, hidden_size, dropout_prob, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, outputs_from_roberta, labels = None):
        x = outputs_from_roberta[0][:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        logits = self.out_proj(x)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        # When labels is None, assuming in evaluation mode, thus return logits
        return logits

# class SequenceClassificationHeadBySimilarity(nn.Module):
#     def __inti__(self, ref_reps):


#     def forward(self, quer_reps):


class TokenClassificationHead(nn.Module):
    """
        - https://huggingface.co/course/chapter7/7?fw=tf
        - https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py#L1473
    """
    def __init__(self, hidden_size, dropout_prob, num_labels):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)


    def forward(self, outputs_from_roberta, start_positions, end_positions):
        sequence_output = outputs_from_roberta[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            # assuming it's evluation mode if no labels provided
            start_idx = torch.argmax(start_logits, dim=1)
            end_idx = torch.argmax(end_logits, dim=1) + 1
            return start_idx, end_idx


