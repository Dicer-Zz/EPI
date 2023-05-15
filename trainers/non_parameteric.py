import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.non_parameteric import NonParameteric


class Trainer:

    def __init__(self, args):
        self.args = args
        self.model = NonParameteric(args)
        self.task_num = 0

    def new_task(self, train_dataset, val_dataset, num_labels):
        self.task_num += 1
        # expand classifier and prefix
        self.model.new_task(num_labels)
        # statistic
        self.statistic(train_dataset)
        # evaluting
        # self.evaluating(val_dataset)

    def evaluating(self, dataset):
        loader = DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True)
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for _, batch in enumerate(tqdm(loader, desc="Evaluating")):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()
                pred, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                correct += torch.sum(pred == labels).item()
                total += len(labels)

        return correct / total

    def evaluating_for_datsets(self, datasets):
        eval_acc = [self.evaluating(dataset) for dataset in datasets]
        print(f"-" * 50)
        print(
            f"Average Evaluating Accuracy: {np.mean(eval_acc): .4f}")
        print(f"{np.around(eval_acc, 4)}")
        print(f"-" * 50)
        return eval_acc

    def statistic(self, dataset):
        mean, cov = self.get_mean_and_cov(dataset)
        self.model.new_statistic(mean, cov)

    def get_mean_and_cov(self, dataset):
        loader = DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=False)
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            prelogits = []
            labels = []
            for _, batch in enumerate(tqdm(loader, desc="Statistic...")):
                input_ids, attention_mask, label = batch
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                prelogit = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    get_prelogits=True,
                )
                prelogits.extend(prelogit.tolist())
                labels.extend(label.tolist())

        if self.args.reduce_dim_mode == "pca":
            from sklearn.decomposition import PCA
            reduced_dim = int((1 - self.args.reduce_ratio)
                              * self.model.config.hidden_size)
            pca = PCA(n_components=reduced_dim)
            prelogits = pca.fit_transform(prelogits)
            # update model pca
            if self.model.num_tasks == 1:
                self.model.pca.mean_ = pca.mean_
                self.model.pca.components_ = pca.components_
            else:
                self.model.pca.mean_ = (
                    self.model.pca.mean_ * (self.model.num_tasks-1) + pca.mean_) / self.model.num_tasks
                self.model.pca.components_ = (self.model.pca.components_ * (
                    self.model.num_tasks-1) + pca.components_) / self.model.num_tasks

        prelogits = torch.tensor(prelogits)
        labels = torch.tensor(labels)
        labels_space = torch.unique(labels)

        task_mean = prelogits.mean(dim=0)
        task_cov = torch.cov((prelogits - task_mean).T)

        mean_over_classes = []
        cov_over_classes = []
        for c in labels_space:
            embeds = prelogits[labels == c]
            if embeds.numel() > 0:
                mean = prelogits[labels == c].mean(dim=0)
                cov = torch.cov((prelogits - mean).T)
            else:
                mean = task_mean
                cov = task_cov
            mean_over_classes.append(mean)
            cov_over_classes.append(cov)

        mean_over_classes = torch.stack(mean_over_classes)
        shared_cov = torch.stack(cov_over_classes).mean(dim=0)

        return mean_over_classes, shared_cov
