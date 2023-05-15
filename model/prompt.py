import torch
from torch import nn
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from utils.assets import get_prompts_data


class AutoPrompt(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.model = AutoModel.from_pretrained(args.model_name_or_path)
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        if self.args.frozen:
            # frozen model's parameters
            for param in self.model.parameters():
                param.requires_grad = False

        self.n_layer = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size // self.config.num_attention_heads

        self.classifiers = nn.ModuleList()
        self.prompts = nn.ParameterList()
        self.num_tasks = 0

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
        new_classifier = torch.nn.Linear(self.config.hidden_size, num_labels)
        new_prompt = self.get_prompts_data()
        self.num_tasks += 1
        self.classifiers.append(new_classifier)
        self.prompts.append(nn.Parameter(new_prompt, requires_grad=True))

    def switch_task(self, task_id):
        assert task_id < self.num_tasks
        self.classifier = self.classifiers[task_id]
        self.prompt = self.prompts[task_id]

    def get_prompt(self, batch_size):
        return self.prompt.expand(batch_size, -1, -1)

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
    ):

        batch_size = input_ids.shape[0]

        prompts = self.get_prompt(batch_size)
        bs, psl, hs = prompts.size()
        if self.args.prompt_mode == "prompt":
            prompts = prompts.view(bs, psl, hs)
            raw_embedding = self.model.embeddings(
                input_ids, position_ids, token_type_ids)
            inputs_embeds = torch.cat([prompts, raw_embedding], dim=1)
        elif self.args.prompt_mode == "prefix":
            past_key_values = prompts.view(
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

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
