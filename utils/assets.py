import torch
import random
import numpy as np
from transformers import HfArgumentParser
from dataclasses import dataclass, field


@dataclass
class TrainingArguments:
    batch_size: int = field(
        default=32,
        metadata={
            "help": "Batch size for training."
        }
    )
    lr: float = field(
        default=3e-2,
        metadata={
            "help": "Learning rate for training."
        }
    )
    lr_list: str = field(
        default=None,
        metadata={
            "help": "Learning rate for each task."
        }
    )
    epochs: int = field(
        default=5,
        metadata={
            "help": "Number of epochs for training."
        }
    )
    epochs_list: str = field(
        default=None,
        metadata={
            "help": "Number of epochs of training for each task."
        }
    )
    warmup_ratio: float = field(
        default=0.1,
        metadata={
            "help": "Warm up ratio for training"
        }
    )
    buffer_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Buffer ratio for replaying"
        }
    )
    buffer_size: int = field(
        default=0,
        metadata={
            "help": "Buffer size for replaying (high priority)"
        }
    )
    replay_freq: int = field(
        default=10,
        metadata={
            "help": "Replaying after every replay_freq steps"
        }
    )
    dataset: str = field(
        default="5ds",
        metadata={
            "help": "Dataset name for training.",
            "choices": ["5ds", "20news", "WOS11967", "WOS5736", "WOS46985"]
        }
    )
    order: str = field(
        default=None,
        metadata={
            "help": "Order of datasets to use for training."
        }
    )
    n_per_class: int = field(
        default=-1,
        metadata={
            "help": "Number of samples per class to use for training. -1 means all."
        }
    )
    max_length: int = field(
        default=256,
        metadata={
            "help": "Maximum length for tokenization."
        }
    )
    gpu_id: int = field(
        default=0,
        metadata={
            "help": "GPU ID to use."
        }
    )
    save_dir: str = field(
        default=None,
        metadata={
            "help": "Directory to save model."
        }
    )
    seed: int = field(
        default=42,
        metadata={
            "help": "Random seed for training."
        }
    )
    debug: bool = field(
        default=False,
        metadata={
            "help": "Debug mode."
        }
    )


@dataclass
class ModelArguments:
    method: str = field(
        default="epi",
        metadata={
            "help": "Method to use for training."
            "epi - epi method"
            "baseline - baseline method"
            "l2p - learning to prompt"
            "msp - maximum softmax probability"
            "np - non-parameteric method (e.g., euclidean, mahalanobis",
            "choices": ["epi", "baseline", "l2p", "msp", "np"]
        }
    )
    model_name_or_path: str = field(
        default="bert-base-uncased",
        metadata={
            "help": "Path or name of the model."
        }
    )
    rep_mode: str = field(
        default="avg",
        metadata={
            "help": "How to represent a sentence."
            "avg - average of word embeddings"
            "cls - [CLS] token embedding",
            "choices": ["cls", "avg"]
        }
    )
    query_mode: str = field(
        default="mahalanobis",
        metadata={
            "help": "How to calculate the distance from sample to prototype in feature space."
            "cosine - cosine similarity"
            "euclidean - euclidean distance"
            "mahalanobis - mahalanobis distance"
            "maha_ft - fine-tuning before get mean and cov",
            "choices": ["cosine", "euclidean", "mahalanobis", "maha_ft"]
        }
    )
    prompt_mode: str = field(
        default="prefix",
        metadata={
            "help": "How to add prompt to the input."
            "none - don't add prompt"
            "prefix (deep prompt) - add prompt to the beginning of each hidden states"
            "prompt - add prompt to the beginning of the input",
            "choices": ["none", "prefix", "prompt"]
        }
    )
    prompt_fusion_mode: str = field(
        default=None,
        metadata={
            "help": "How to fuse tasks prompt."
            "mean - initialize prompt with mean of all the tasks prompt"
            "last - initialize prompt with last task prompt"
            "concat - concat prompts for new task"
            "attention - TODO",
            "choices": ["mean", "last", "concat", "attention"]
        }
    )
    frozen: bool = field(
        default=True,
        metadata={
            "help": "Whether to freeze the pre-trained model."
        }
    )
    lwf: bool = field(
        default=False,
        metadata={
            "help": "Use LWF loss"
        }
    )
    pre_seq_len: int = field(
        default=16,
        metadata={
            "help": "Length of the prompt."
        }
    )
    pool_size: int = field(
        default=20,
        metadata={
            "help": "The size of prompt pool (for L2P)"
        }
    )
    T: float = field(
        default=2.0,
        metadata={
            "help": "Temperature for KD loss"
        }
    )
    topk: int = field(
        default=5,
        metadata={
            "help": "Prepend top k prompts to the input tokens (for L2P)"
        }
    )
    reduce_dim_mode: str = field(
        default=None,
        metadata={
            "help": "Whether to reduce dimension of prelogits"
            "None - don't reduce dimension"
            "mask - mask some prelogits"
            "pca - use PCA to reduce dimension",
            "choices": ["mask", "pca"]
        }
    )
    reduce_ratio: float = field(
        default=0.0,
        metadata={
            "help": "The ratio of prelogits to be reduce"
        }
    )

def get_args():
    """
    Parser all the arguments.
    """
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    args, unk = parser.parse_known_args()
    if args.order is None:
        args.order = list(dataset_classes[args.dataset].keys())
    else:
        args.order = args.order.split()

    if args.epochs_list is None:
        args.epochs_list = [args.epochs] * len(args.order)
    else:
        args.epochs_list = [int(x) for x in args.epochs_list.split()]
    assert len(args.order) == len(args.epochs_list)

    if args.lr_list is None:
        args.lr_list = [args.lr] * len(args.order)
    else:
        args.lr_list = [float(x) for x in args.lr_list.split()]
    assert len(args.order) == len(args.epochs_list)

    print(args)
    print(f"Unknown args: {unk}")
    return args


def tunable_params_stastic(model):
    # count the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(
        f"Trainable Parameters Ratio: {trainable_params / total_params: .4f}")


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def mahalanobis(querys, mean, cov_inv, norm=2):
    """
    args:
        querys: [n, dim]
        mean: [dim]
        cov_inv: [dim, dim]
    return：
        [n]
    """
    diff = querys - mean
    # [n, dim] = ([n, dim] @ [dim, dim]) * [n, dim] = [n, dim] * [n, dim]
    maha_dis = torch.matmul(diff, cov_inv) * diff

    if norm == 2:
        return maha_dis.sum(dim=1)
    if norm == 1:
        return maha_dis.abs().sqrt().sum(dim=1)
    if norm == 'inf':
        return maha_dis.max(dim=1)


def get_prompts_data(args, config, device, prompt_init=None):
    if prompt_init is None:
        prompt_init = torch.randn

    if args.prompt_mode == "prompt":
        new_prompt_data = prompt_init(
            args.pre_seq_len, config.hidden_size, device=device)
    elif args.prompt_mode == "prefix":
        # simple prefix module
        new_prompt_data = prompt_init(
            args.pre_seq_len, config.num_hidden_layers * config.hidden_size * 2, device=device)
    else:
        raise NotImplementedError
    return new_prompt_data


def mean_pooling(hidden_states, attention_mask):
    pooled_output = torch.sum(hidden_states * attention_mask.unsqueeze(-1),
                              dim=1) / torch.sum(attention_mask, dim=1).unsqueeze(-1)
    return pooled_output


dataset_classes = {
    '5ds': {
        'ag': 4,
        'yelp': 5,
        'amazon': 5,
        'yahoo': 10,
        'dbpedia': 14,
    },
    '20news': {
        '0': 6,
        '1': 5,
        '2': 5,
        '3': 4,
    },
    'WOS11967': {
        '0': 5,
        '1': 3,
        '2': 5,
        '3': 5,
        '4': 5,
        '5': 5,
        '6': 5,
    },
    'WOS5736': {
        '0': 3,
        '1': 4,
        '2': 4,
    },
    'WOS46985': {
        '0': 17,
        '1': 16,
        '2': 19,
        '3': 9,
        '4': 11,
        '5': 53,
        '6': 9,
    }
}
