import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.epi import EPI
from transformers import get_linear_schedule_with_warmup


class Trainer:

    def __init__(self, args):
        self.args = args
        self.model = EPI(args)
        self.task_num = 0

    def new_task(self, train_dataset, val_dataset, num_labels):
        self.task_num += 1
        # expand classifier and prefix
        self.model.new_task(num_labels)
        # fit to new dataset
        self.training(train_dataset, val_dataset)
        # statistic
        self.statistic(train_dataset)
        # evaluting
        # self.evaluating(val_dataset, self.task_num - 1, oracle=True)

    def training(self, dataset, val_dataset=None):
        loader = DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.args.lr_list[self.task_num - 1], weight_decay=0.0)
        num_training_steps = len(
            loader) * self.args.epochs_list[self.task_num - 1]
        num_warmup_steps = num_training_steps * self.args.warmup_ratio
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps)

        self.model.cuda()
        best_acc, best_epoch = 0, -1
        for epoch in range(self.args.epochs_list[self.task_num - 1]):
            self.model.train()
            correct, total = 0, 0
            total_loss = 0
            for _, batch in enumerate(tqdm(loader, desc=f"Training Epoch {epoch}")):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()
                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss.backward()
                total_loss += loss.item()
                optimizer.step()
                scheduler.step()

                pred = torch.argmax(outputs.logits, dim=1)
                correct += torch.sum(pred == labels).item()
                total += len(labels)

            print(f"Epoch {epoch} Training Accuracy: {correct/total}")
            print(f"Epoch {epoch} Average Loss: {total_loss/len(loader)}")

            if val_dataset is not None:
                acc, hit_acc = self.evaluating(
                    val_dataset, self.task_num - 1, oracle=True)
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    best_model = deepcopy(self.model.state_dict())
                print(f"Epoch {epoch} Validation Accuracy: {acc}")
                print(f"Epoch {epoch} Validation Hitting Accuracy: {hit_acc}")

        print(f"Best Epoch: {best_epoch}")
        print(f"Best Accuracy: {best_acc}")

        print(f"Loading best model...")
        self.model.load_state_dict(best_model)

    def evaluating(self, dataset, task_id, oracle=False):
        loader = DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True)
        self.model.cuda()
        self.model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            hittings = 0
            for _, batch in enumerate(tqdm(loader, desc="Evaluating")):
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_id=task_id,
                    oracle=oracle,
                )
                logits = outputs.logits
                pred = torch.argmax(logits, dim=1)
                hittings += outputs.hittings
                correct += torch.sum(pred == labels).item()
                total += len(labels)

            print(
                f"Evaluating Accuracy{' (oracle)' if oracle else ''}: {correct/total: .4f}")
            print(
                f"Hitting Accuracy{' (oracle)' if oracle else ''}: {hittings/total: .4f}")
        return correct / total, hittings / total

    def evaluating_for_datsets(self, datasets, oracle=False):
        accuracies = [self.evaluating(dataset, task_id, oracle=oracle)
                      for task_id, dataset in enumerate(datasets)]
        eval_acc = [acc[0] for acc in accuracies]
        hit_acc = [acc[1] for acc in accuracies]
        print(f"-" * 50)
        print(
            f"Average Evaluating Accuracy{' (oracle)' if oracle else ''}: {np.mean(eval_acc): .4f}")
        print(f"{np.around(eval_acc, 4)}")
        print(
            f"Average Hitting Accuracy{' (oracle)' if oracle else ''}: {np.mean(hit_acc): .4f}")
        print(f"{np.around(hit_acc, 4)}")
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
                    by_prefix=self.args.query_mode == "maha_ft",
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
                mean = embeds.mean(dim=0)
                cov = torch.cov((embeds - mean).T)
            else:
                mean = task_mean
                cov = task_cov
            mean_over_classes.append(mean)
            cov_over_classes.append(cov)

        mean_over_classes = torch.stack(mean_over_classes)
        shared_cov = torch.stack(cov_over_classes).mean(dim=0)

        return mean_over_classes, shared_cov
