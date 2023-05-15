import torch
from copy import deepcopy
from torch import nn
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoModel, AutoConfig
from transformers.utils import ModelOutput
from utils.assets import get_prompts_data, mean_pooling


class Baseline(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path, config=self.config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.classifier = nn.Linear(
            self.config.hidden_size, 0, device=self.device)
        new_prompt_data = get_prompts_data(self.args, self.config, self.device)
        self.prompt = nn.Parameter(new_prompt_data, requires_grad=True)

        # for prefix
        self.n_layer = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size // self.config.num_attention_heads

        if self.args.frozen:
            # frozen model's parameters
            for param in self.model.parameters():
                param.requires_grad = False

        self.num_labels = 0
        self.num_tasks = 0
        self.old_model = None

    def get_prompts_data(self):
        if self.args.prompt_mode == "prompt":
            new_prompt_data = torch.randn(
                self.args.pre_seq_len, self.config.hidden_size, device=self.device)
        elif self.args.prompt_mode == "prefix":
            # simple prefix module
            new_prompt_data = torch.randn(
                self.args.pre_seq_len, self.config.num_hidden_layers * self.config.hidden_size * 2, device=self.device)
        return nn.Parameter(new_prompt_data, requires_grad=True)

    def new_task(self, num_labels):
        self.old_num_labels = self.num_labels

        # save old model for distillation
        if self.num_tasks > 0 and self.args.lwf:
            self.old_model = None
            self.old_model = deepcopy(self)
            # freeze old model's parameters for accelerating
            for param in self.old_model.parameters():
                param.requires_grad = False

        self.num_tasks += 1

        with torch.no_grad():
            # expand classifier
            num_old_labels = self.num_labels
            self.num_labels += num_labels
            w = self.classifier.weight.data.clone()
            b = self.classifier.bias.data.clone()
            self.classifier = nn.Linear(
                self.config.hidden_size, self.num_labels, device=self.device)
            self.classifier.weight.data[:num_old_labels] = w
            self.classifier.bias.data[:num_old_labels] = b

    def get_prelogits(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None
    ):
        bs = input_ids.size(0)
        prompt = self.prompt.repeat(bs, 1, 1)

        bs, psl, hs = prompt.size()
        if self.args.prompt_mode == "prompt":
            prompt = prompt.view(bs, psl, hs)
            raw_embedding = self.model.embeddings(
                input_ids, position_ids, token_type_ids)
            inputs_embeds = torch.cat([prompt, raw_embedding], dim=1)
        elif self.args.prompt_mode == "prefix":
            past_key_values = prompt.view(
                bs, psl, self.n_layer * 2, self.n_head, self.n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            inputs_embeds = self.model.embeddings(
                input_ids, position_ids, token_type_ids)
        prompt_attention_mask = torch.ones(
            bs, psl, dtype=torch.long, device=attention_mask.device)
        attention_mask = torch.cat(
            [prompt_attention_mask, attention_mask], dim=1)

        outputs = self.model(
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        if self.args.rep_mode == "cls":
            prelogits = outputs[1]
        elif self.args.rep_mode == "avg":
            sequence_output = outputs[0]
            prelogits = mean_pooling(sequence_output, attention_mask[:, psl:])
        else:
            raise NotImplementedError

        prelogits = self.dropout(prelogits)
        return outputs, prelogits

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None,
        get_logits=False,
    ):

        # for code readability
        args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "inputs_embeds": inputs_embeds,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "past_key_values": past_key_values,
        }

        outputs, pooled_output = self.get_prelogits(**args)
        logits = self.classifier(pooled_output)

        if get_logits:
            return logits

        if self.training:
            if self.old_model is not None and self.args.lwf:
                with torch.no_grad():
                    old_logits = self.old_model(**args, get_logits=True)

                # distillation loss
                dis_loss = lwf_loss(
                    logits[:, :self.old_num_labels], old_logits, self.args.T)

            if self.num_tasks > 1 and self.args.buffer_ratio == 0 and self.args.buffer_size == 0:
                logits[:, :self.old_num_labels] = -1e4

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, logits.shape[-1]), labels.view(-1))
                if self.old_model is not None and self.args.lwf:
                    loss += dis_loss
        else:
            loss = None

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class SequenceClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def lwf_loss(pred, soft, T=2):
    """
    args:
        pred: logits of student model, [n, old_num_labels]
        soft: logits of teacher model, [n, old_num_labels]
        T: temperature
    return:
        loss: distillation loss (batch mean)
    """
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]
