import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoConfig
from utils.assets import mahalanobis, mean_pooling


class NonParameteric(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path, config=self.config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.args.frozen:
            # frozen model's parameters
            for param in self.model.parameters():
                param.requires_grad = False

        self.num_labels = 0
        self.num_tasks = 0
        self.task_range = []

        self.query_size = self.config.hidden_size
        if self.args.reduce_dim_mode == "mask":
            # reduce dimension
            self.mask = torch.rand(
                self.config.hidden_size) > self.args.reduce_ratio
            self.query_size = self.mask.sum()
            print(
                f"Reduced dimension: {self.config.hidden_size} -> {self.query_size}")
        elif self.args.reduce_dim_mode == "pca":
            from sklearn.decomposition import PCA
            self.query_size = int(
                (1 - self.args.reduce_ratio) * self.model.config.hidden_size)
            print(
                f"Reduced dimension: {self.config.hidden_size} -> {self.query_size}")
            self.pca = PCA(n_components=self.query_size)

        self.task_means_over_classes = nn.ParameterList()
        self.accumulate_shared_covs = nn.Parameter(torch.zeros(
            self.query_size, self.query_size), requires_grad=False)
        self.cov_inv = nn.Parameter(torch.ones(
            self.query_size, self.query_size), requires_grad=False)

    def new_task(self, num_labels):
        self.num_tasks += 1
        self.task_range.append((self.num_labels, self.num_labels + num_labels))
        self.num_labels += num_labels

    def new_statistic(self, mean, cov):
        self.task_means_over_classes.append(
            nn.Parameter(mean.cuda(), requires_grad=False))

        self.accumulate_shared_covs.data = self.accumulate_shared_covs.data.cpu()
        self.accumulate_shared_covs += cov

        self.cov_inv = nn.Parameter(torch.linalg.pinv(
            self.accumulate_shared_covs / self.num_tasks, hermitian=True), requires_grad=False)

    def get_logits(self, prelogits):
        """
        arguments:
            prelogits: [bs, hidden_size]
        return:
            logits: [bs, num_labels]
        """
        if self.args.reduce_dim_mode == "pca":
            prelogits = self.pca.transform(prelogits.cpu().numpy())
            prelogits = torch.from_numpy(prelogits).cuda()

        scores_over_tasks = []
        for idx, mean_over_classes in enumerate(self.task_means_over_classes):
            num_labels, _ = mean_over_classes.shape
            score_over_classes = []
            for c in range(num_labels):
                if self.args.query_mode == "cosine":
                    score = - \
                        F.cosine_similarity(prelogits, mean_over_classes[c])
                elif self.args.query_mode == "euclidean":
                    score = torch.cdist(
                        prelogits, mean_over_classes[c].unsqueeze(0)).squeeze(1)
                elif self.args.query_mode == "mahalanobis":
                    score = mahalanobis(
                        prelogits, mean_over_classes[c], self.cov_inv, norm=2)
                else:
                    raise NotImplementedError

                score_over_classes.append(score)
            # [num_labels, n]
            score_over_classes = torch.stack(score_over_classes)
            scores_over_tasks.append(score_over_classes)
        # [self.num_labels, n]
        scores_over_tasks = torch.cat(scores_over_tasks, dim=0)
        _, indices = torch.min(scores_over_tasks, dim=0)

        return indices, scores_over_tasks

    def forward(
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
        past_key_values=None,
        get_prelogits=False,
    ):
        outputs = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
            output_attentions = output_attentions,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict,
            past_key_values = past_key_values,
        )
        last_hidden_state = outputs.last_hidden_state
        prelogits = mean_pooling(last_hidden_state, attention_mask)

        if get_prelogits:
            return prelogits

        preds, logits = self.get_logits(prelogits)

        return preds, logits