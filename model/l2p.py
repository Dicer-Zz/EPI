import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from utils.assets import get_prompts_data, mean_pooling


class L2P(nn.Module):
    """
    Learning to prompt for continual learning
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path, config=self.config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pool = PromptPool(self.config, args, self.device)
        self.classifier = nn.Linear(
            self.config.hidden_size, 0, device=self.device)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.num_old_labels = 0
        self.num_labels = 0

        self.n_layer = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size // self.config.num_attention_heads

        if self.args.frozen:
            # frozen model's parameters
            for param in self.model.parameters():
                param.requires_grad = False

    def new_task(self, num_labels):
        self.num_old_labels = self.num_labels
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # get querys
        last_hidden_state = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        ).last_hidden_state
        querys = mean_pooling(last_hidden_state, attention_mask)

        prompts, similarity = self.pool(querys)
        bs, topk, psl, hs = prompts.size()
        if self.args.prompt_mode == "prompt":
            prompts = prompts.view(bs, topk * psl, hs)
            raw_embedding = self.model.embeddings(
                input_ids, position_ids, token_type_ids)
            inputs_embeds = torch.cat([prompts, raw_embedding], dim=1)
        elif self.args.prompt_mode == "prefix":
            past_key_values = prompts.view(
                bs, topk * psl, self.n_layer * 2, self.n_head, self.n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            inputs_embeds = self.model.embeddings(
                input_ids, position_ids, token_type_ids)
        prompt_attention_mask = torch.ones(
            bs, topk * psl, dtype=torch.long, device=attention_mask.device)
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
            pooled_output = outputs[1]
        elif self.args.rep_mode == "avg":
            sequence_output = outputs[0]
            pooled_output = mean_pooling(
                sequence_output, attention_mask[:, topk * psl:])
        else:
            raise NotImplementedError

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        # trick for task boundary setting
        if self.training and self.args.buffer_ratio == 0 and self.args.buffer_size == 0:
            logits[:, :self.num_old_labels] = -1e4

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))

            # optimization for query mechanism
            loss = loss - 0.5 * similarity.mean()

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class PromptPool(nn.Module):
    """
    PromptPool:
        a pool of prompts for l2p
    """

    def __init__(self, config, args, device):
        super().__init__()
        self.args = args
        self.config = config
        self.pool_size = args.pool_size
        self.topk = args.topk
        self.pre_seq_len = args.pre_seq_len
        self.hidden_size = config.hidden_size

        self.query_mode = args.query_mode  # ["cosine", "euclidean"]
        self.prompt_mode = args.prompt_mode  # ["prompt", "prefix"]
        self.device = device

        prompts_key_init = torch.zeros  # randn, rand, zeros
        prompts_init = torch.rand  # randn, rand, zeros

        prompts_key = prompts_key_init(
            self.pool_size, self.hidden_size, device=self.device)
        self.prompts_key = nn.Parameter(prompts_key, requires_grad=True)

        new_prompt_data = torch.stack([get_prompts_data(
            args, config, self.device, prompts_init) for _ in range(self.pool_size)])
        self.prompts = torch.nn.Parameter(new_prompt_data, requires_grad=True)

    def forward(self, querys):
        # querys: [bs, hidden_size]
        if self.query_mode == "cosine":
            # [bs, pool_size]
            similarity = F.cosine_similarity(querys.unsqueeze(
                1), self.prompts_key.unsqueeze(0), dim=-1)
        elif self.query_mode == "euclidean":
            # negative euclidean distance for sorting
            similarity = -torch.cdist(querys, self.prompts_key, p=2)
        else:
            raise NotImplementedError
        _, indices = torch.topk(similarity, self.topk, dim=-1)  # [bs, topk]
        # return similarity corresponding to indices
        similarity = torch.gather(similarity, dim=-1, index=indices)
        # similarity: [bs, topk]
        return self.prompts[indices], similarity
