import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoModel, AutoConfig
from transformers.utils import ModelOutput
from utils.assets import get_prompts_data, mean_pooling


class MSP(nn.Module):
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
        self.task_range = []

        self.prompts = nn.ParameterList()

    def get_prompts_data(self):
        # inherit previous prompts
        if self.args.prompt_fusion_mode == "mean" and self.num_tasks > 0:
            new_prompt_data = torch.mean(torch.stack(
                [prompt for prompt in self.prompts]), dim=0)
        elif self.args.prompt_fusion_mode == "last" and self.num_tasks > 0:
            # inherit last task prompt
            new_prompt_data = self.prompts[-1].data.clone()
        else:
            new_prompt_data = get_prompts_data(
                self.args, self.config, self.device)
        return new_prompt_data

    def new_task(self, num_labels):
        # frozen previous prompts
        for param in self.prompts.parameters():
            param.requires_grad = False

        prompts_data = self.get_prompts_data()
        self.prompts.append(nn.Parameter(prompts_data, requires_grad=True))
        self.num_tasks += 1
        self.task_range.append((self.num_labels, self.num_labels + num_labels))

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
        indices,
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
        prompt = torch.stack([self.prompts[idx] for idx in indices])

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
        task_id=None,
        oracle=False,
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

        bs = input_ids.size(0)
        if self.training:
            indices = [self.num_tasks - 1] * bs
        else:
            if oracle:
                indices = [task_id] * bs
            else:
                msp = torch.zeros(bs, device=self.device)
                ids = torch.zeros(bs, dtype=torch.long, device=self.device)
                for idx in range(self.num_tasks):
                    indices = [idx] * bs
                    outputs, pooled_output = self.get_prelogits(indices, **args)
                    logits = self.classifier(pooled_output)
                    for j, index in enumerate(indices):
                        start, end = self.task_range[index]
                        logits[j][:start] = -1e4
                        logits[j][end:] = -1e4
                    _msp = torch.max(F.softmax(logits, dim=-1), dim=-1)[0]
                    ids[torch.where(_msp > msp)] = idx
                    msp[torch.where(_msp > msp)] = _msp[torch.where(_msp > msp)]
                indices = ids.tolist()

        outputs, pooled_output = self.get_prelogits(indices, **args)

        logits = self.classifier(pooled_output)
        for idx, index in enumerate(indices):
            start, end = self.task_range[index]
            logits[idx][:start] = -1e4
            logits[idx][end:] = -1e4

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

        if isinstance(indices, list):
            indices = torch.tensor(indices)
        hittings = (indices == task_id).sum(
        ).item() if task_id is not None else None

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            prelogits=pooled_output,
            hittings=hittings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@dataclass
class SequenceClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    prelogits: torch.FloatTensor = None
    hittings: int = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
