import csv
import random
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from datasets import load_dataset
from utils.assets import dataset_classes
from typing import Union


class SentsDataset(Dataset):
    """
    A dataset for sentences
    """

    def __init__(self, sents, labels, tokenizer, max_length):
        """
        args:
            sents: a list of sentence
            labels: a list of label
            tokenizer: a tokenizer
            max_length: maximum sequence length
        """
        self.labels = labels
        encoding = tokenizer(sents, truncation=True, padding="max_length",
                             max_length=max_length, return_tensors="pt")
        self.input_ids = encoding['input_ids']
        self.attention_mask = encoding['attention_mask']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.labels[idx]


def statistic(dataset: Union[Dataset, Subset, ConcatDataset]):
    def get_labels(dataset):
        if isinstance(dataset, Subset):
            return np.array(get_labels(dataset.dataset))[dataset.indices]
        elif isinstance(dataset, ConcatDataset):
            return np.concatenate([get_labels(dataset.datasets[i])
                                   for i in range(len(dataset.datasets))])
        elif isinstance(dataset, Dataset):
            return dataset.labels
    labels = get_labels(dataset)
    print(np.bincount(labels))


def get_dataset(args, tokenizer, offset=True, joint=False, task_label=False):
    """
    args:
        args: control arguments
        tokenizer: tokenizer for tokenize
        split: for combined dataset (train or test)
        offset: whethere share same label space for all datasets (by offset some dataset label)
        joint: whethere concat all datasets
        task_label: concat all datasets and label them by task_id
    return:
        if joint is True:
            concated training and testing dataset
        else:
            a list of training datasets and a list of testing datasets
    """
    global dataset_classes
    dataset_classes = dataset_classes[args.dataset]
    if args.dataset == None or args.dataset == "5ds":
        if args.n_per_class != -1:
            # sampled setting
            training, _ = get_combined_dataset(
                args.order, tokenizer, "train", args.n_per_class, args.max_length, offset, joint, task_label)
            validation, _ = get_combined_dataset(
                args.order, tokenizer, "train", 2000, args.max_length, offset, joint, task_label)
        else:
            # full setting
            training, validation = get_combined_dataset(
                args.order, tokenizer, "train", args.n_per_class, args.max_length, offset, joint, task_label, val_n_per_class=500)
        testing, _ = get_combined_dataset(
            args.order, tokenizer, "test", -1 if not args.debug else args.n_per_class, args.max_length, offset, joint, task_label)
    elif args.dataset in ["WOS5736", "WOS11967", "WOS46985"]:
        # train/dev/test: 0.6/0.2/0.2
        dataset = load_dataset("web_of_science", args.dataset)["train"]
        dataset = dataset.train_test_split(0.2, shuffle=True, seed=42)
        train_dataset, test_dataset = dataset["train"], dataset["test"]
        training, validation = get_wos_dataset(
            train_dataset, args.order, tokenizer, args.n_per_class, args.max_length, offset, joint, task_label, val_ratio=0.25)
        testing, _ = get_wos_dataset(
            test_dataset, args.order, tokenizer, -1 if not args.debug else args.n_per_class, args.max_length, offset, joint, task_label)
    elif args.dataset == "20news":
        # train/dev/test: 0.5/0.1/0.4
        dataset = load_dataset("SetFit/20_newsgroups")
        train_dataset, test_dataset = dataset["train"], dataset["test"]
        training, validation = get_news_dataset(
            train_dataset, args.order, tokenizer, args.n_per_class, args.max_length, offset, joint, task_label, val_ratio=0.167)
        testing, _ = get_news_dataset(
            test_dataset, args.order, tokenizer, -1 if not args.debug else args.n_per_class, args.max_length, offset, joint, task_label)
    else:
        raise NotImplementedError

    if isinstance(training, list):
        for idx, dataset in enumerate(args.order):
            print(f"{dataset} dataset statistic:")
            statistic(training[idx])
            statistic(validation[idx])
            statistic(testing[idx])
    else:
        print("Dataset statistic:")
        statistic(training)
        statistic(validation)
        statistic(testing)
    return training, validation, testing


def get_combined_dataset(datasets_name, tokenizer, split="train", n_per_class=2000, max_length=256, offset=False, joint=False, task_label=False, val_n_per_class=0):

    datasets = []
    validation = []

    cur_offset = 0
    for idx, dataset in enumerate(datasets_name):
        file = f"data/{dataset}/{split}.csv"
        sents, labels = read_data(file, dataset_classes[dataset], n_per_class)
        if offset or joint:
            if task_label:
                labels = [idx] * len(labels)
            else:
                labels = [label + cur_offset for label in labels]
            cur_offset += dataset_classes[dataset]

        val_sents, val_labels = [], []
        train_sents, train_labels = [], []
        capacity = {label: val_n_per_class for label in set(labels)}
        for sent, label in zip(sents, labels):
            if capacity[label] > 0:
                val_sents.append(sent)
                val_labels.append(label)
                capacity[label] -= 1
            else:
                train_sents.append(sent)
                train_labels.append(label)

        val_set = SentsDataset(val_sents, val_labels,
                               tokenizer, max_length) if val_n_per_class > 0 else None
        validation.append(val_set)
        dataset = SentsDataset(train_sents, train_labels, tokenizer, max_length)
        datasets.append(dataset)

    if joint:
        if val_n_per_class > 0:
            return ConcatDataset(datasets), ConcatDataset(validation)
        else:
            return ConcatDataset(datasets), None
    return datasets, validation


def get_wos_dataset(dataset, datasets_id, tokenizer, n_per_class=2000, max_length=256, offset=True, joint=False, task_label=False, val_ratio=0.0):
    """
    input_data: input text
    label: class label in all datasets label space
    label_level_1: task label
    label_level_2: class label in corresponding task label space
    """

    dataset = dataset.shuffle(seed=42)

    if task_label:
        text = list(dataset["input_data"])
        labels = list(dataset["label_level_1"])
        val_size = int(len(text) * val_ratio)
        val_text, val_labels = text[:val_size], labels[:val_size]
        text, labels = text[val_size:], labels[val_size:]
        val_set = SentsDataset(val_text, val_labels,
                                 tokenizer, max_length) if val_ratio > 0 else None
        return SentsDataset(text, labels, tokenizer, max_length), val_set

    num_task = len(set(dataset["label_level_1"]))
    sents_of_task = [[] for _ in range(num_task)]
    for _, sample in enumerate(dataset):
        task = sample["label_level_1"]
        sents_of_task[task].append(
            (sample["input_data"], sample["label"] if offset else sample["label_level_2"]))

    if offset:
        # for joint training
        capacity = [n_per_class for _ in range(len(set(dataset["label"])))]
    datasets = []
    validations = []
    for task in range(num_task):
        sampled_sents, sampled_labels = [], []
        if not offset:
            # for separate training
            capacity = [n_per_class for _ in range(
                len(set(dataset["label_level_2"])))]
        for sent, label in sents_of_task[task]:
            if capacity[label] > 0 or n_per_class == -1:
                sampled_sents.append(sent)
                sampled_labels.append(label)
                capacity[label] -= 1
        val_size = int(len(sampled_sents) * val_ratio)
        val_sents, val_labels = sampled_sents[:
                                              val_size], sampled_labels[:val_size]
        sampled_sents, sampled_labels = sampled_sents[val_size:], sampled_labels[val_size:]
        datasets.append(SentsDataset(
            sampled_sents, sampled_labels, tokenizer, max_length))

        val_set = SentsDataset(val_sents, val_labels,
                               tokenizer, max_length) if val_ratio > 0 else None
        validations.append(val_set)

    datasets = [datasets[int(task)] for task in datasets_id]
    validations = [validations[int(task)] for task in datasets_id]
    if joint:
        # joint all of selected the datasets
        if val_ratio > 0:
            return ConcatDataset(datasets), ConcatDataset(validations)
        else:
            return ConcatDataset(datasets), None
    return datasets, validations


def get_news_dataset(dataset, datasets_id, tokenizer, n_per_class=2000, max_length=256, offset=True, joint=False, task_label=False, val_ratio=0.0):
    """
    text: input text
    label: class label in all datasets label space
    label_text: class name corresponding to label
    """
    text2id = {
        "comp.graphics": 0,
        "rec.autos": 0,
        "sci.crypt": 0,
        "misc.forsale": 0,
        "talk.politics.misc": 0,
        "talk.religion.misc": 0,

        "comp.os.ms-windows.misc": 1,
        "rec.motorcycles": 1,
        "sci.electronics": 1,
        "talk.politics.guns": 1,
        "alt.atheism": 1,

        "comp.sys.ibm.pc.hardware": 2,
        "rec.sport.baseball": 2,
        "sci.med": 2,
        "talk.politics.mideast": 2,
        "soc.religion.christian": 2,

        "comp.sys.mac.hardware": 3,
        "rec.sport.hockey": 3,
        "sci.space": 3,
        "comp.windows.x": 3,
    }
    text2label = {
        "comp.graphics": 0,
        "rec.autos": 1,
        "sci.crypt": 2,
        "misc.forsale": 3,
        "talk.politics.misc": 4,
        "talk.religion.misc": 5,

        "comp.os.ms-windows.misc": 6,
        "rec.motorcycles": 7,
        "sci.electronics": 8,
        "talk.politics.guns": 9,
        "alt.atheism": 10,

        "comp.sys.ibm.pc.hardware": 11,
        "rec.sport.baseball": 12,
        "sci.med": 13,
        "talk.politics.mideast": 14,
        "soc.religion.christian": 15,

        "comp.sys.mac.hardware": 16,
        "rec.sport.hockey": 17,
        "sci.space": 18,
        "comp.windows.x": 19,
    }
    dataset = dataset.shuffle(seed=42)

    if task_label:
        text = list(dataset["text"])
        labels = list(map(lambda x: text2id[x], dataset["label_text"]))
        val_size = int(len(text) * val_ratio)
        val_text, val_labels = text[:val_size], labels[:val_size]
        text, labels = text[val_size:], labels[val_size:]
        val_set = SentsDataset(val_text, val_labels,
                                 tokenizer, max_length) if val_ratio > 0 else None
        return SentsDataset(text, labels, tokenizer, max_length), val_set

    if not offset:
        raise NotImplementedError

    num_task = len(set(text2id.values()))
    sent_of_task = [[] for _ in range(num_task)]
    for _, sample in enumerate(dataset):
        task = text2id[sample["label_text"]]
        sent_of_task[task].append((sample["text"], text2label[sample["label_text"]]))

    capacity = [n_per_class for _ in range(len(text2label))]
    datasets = []
    validations = []
    for task in range(num_task):
        sampled_sents, sampled_labels = [], []
        for sent, label in sent_of_task[task]:
            if capacity[label] > 0 or n_per_class == -1:
                sampled_sents.append(sent)
                sampled_labels.append(label)
                capacity[label] -= 1
        val_size = int(len(sampled_sents) * val_ratio)
        val_sents, val_labels = sampled_sents[:
                                              val_size], sampled_labels[:val_size]
        sampled_sents, sampled_labels = sampled_sents[val_size:], sampled_labels[val_size:]
        datasets.append(SentsDataset(
            sampled_sents, sampled_labels, tokenizer, max_length))

        val_set = SentsDataset(val_sents, val_labels,
                               tokenizer, max_length) if val_ratio > 0 else None
        validations.append(val_set)

    datasets = [datasets[int(task)] for task in datasets_id]
    validations = [validations[int(task)] for task in datasets_id]
    if joint:
        if val_ratio > 0:
            return ConcatDataset(datasets), ConcatDataset(validations)
        else:
            return ConcatDataset(datasets), None
    return datasets, validations


def read_data(file_path, n_classes, n_per_class, return_question=False):
    """
    args:
        file_path: path to the data file
        n_classes: number of classes
        n_per_class: number of samples per class
    returns:
        a list of sentences and a list of labels
    """
    sents, questions, labels = [], [], []
    with open(file_path) as csv_file:
        lines = csv.reader(csv_file)
        for line in lines:
            label, question, sent = line
            label = int(label) - 1
            sents.append(sent)
            questions.append(question)
            labels.append(label)

    # shuffle the data with the same order
    sents, questions, labels = np.array(
        sents), np.array(questions), np.array(labels)
    idx = np.arange(len(sents))
    np.random.seed(seed=42)
    np.random.shuffle(idx)
    sents, questions, labels = sents[idx].tolist(
    ), questions[idx].tolist(), labels[idx].tolist()

    if n_per_class == -1:
        return (sents, questions, labels) if return_question else (sents, labels)

    # sample n_per_class samples per class
    capacity = [n_per_class for _ in range(n_classes)]
    sampled_sents, sampled_questions, sampled_labels = [], [], []
    for sent, question, label in zip(sents, questions, labels):
        if capacity[label] > 0 or n_per_class == -1:
            sampled_sents.append(sent)
            sampled_questions.append(question)
            sampled_labels.append(label)
            capacity[label] -= 1

    return (sampled_sents, sampled_questions, sampled_labels) if return_question else (sampled_sents, sampled_labels)
