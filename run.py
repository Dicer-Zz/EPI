import os
import time
import torch
import numpy as np
from transformers import AutoTokenizer
from utils.assets import dataset_classes, get_args, fix_seed, tunable_params_stastic
from utils.read_data import get_dataset


def main(args):
    start_time = time.time()
    if args.method == "epi":
        from trainers.epi import Trainer
    elif args.method in ["baseline", "l2p"]:
        from trainers.baseline import Trainer
    elif args.method == "msp":
        from trainers.msp import Trainer
    elif args.method == "np":
        from trainers.non_parameteric import Trainer

    global dataset_classes
    dataset_classes = dataset_classes[args.dataset]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    train_datasets, val_datasets, test_datasets = get_dataset(args, tokenizer)

    val_acc_table = []
    eval_acc_table = []
    trainer = Trainer(args)
    for idx, dataset in enumerate(args.order):
        print(f"Dataset: {dataset}")
        num_labels = dataset_classes[dataset]
        trainer.new_task(train_datasets[idx], val_datasets[idx], num_labels)
        val_acc = trainer.evaluating_for_datsets(val_datasets[:idx+1]) + [0] * (len(args.order) - idx - 1)
        val_acc_table.append(val_acc)
        eval_acc = trainer.evaluating_for_datsets(test_datasets[:idx+1]) + [0] * (len(args.order) - idx - 1)
        eval_acc_table.append(eval_acc)


    print(f"-" * 50)
    print(f"Validation results (for seleting hyper-parameters)")
    val_acc_table = np.array(val_acc_table)
    print(f"Datasets order: {args.order}")
    print(f"Val acc table: \n{val_acc_table.round(4)}")

    if len(args.order) > 1:
        final_acc = val_acc_table[-1]
        print(f"Final val acc: {final_acc.round(4)}")
        print(f"Avg. Val Acc: {final_acc.mean().round(4)}")
        fgt = val_acc_table[:-1].max(0) - final_acc
        fgt = fgt[:-1]
        print(f"Forggeting: {fgt.round(4)}")
        print(f"Avg. Val Fgt: {fgt.mean().round(4)}")
        print(f"-" * 50)

    print(f"-" * 50)
    print(f"Evaluation results")
    eval_acc_table = np.array(eval_acc_table)
    print(f"Datasets order: {args.order}")
    print(f"Eval acc table: \n{eval_acc_table.round(4)}")

    if len(args.order) > 1:
        final_acc = eval_acc_table[-1]
        print(f"Final acc: {final_acc.round(4)}")
        print(f"Avg. Acc: {final_acc.mean().round(4)}")
        fgt = eval_acc_table[:-1].max(0) - final_acc
        fgt = fgt[:-1]
        print(f"Forggeting: {fgt.round(4)}")
        print(f"Avg. Fgt: {fgt.mean().round(4)}")
        print(f"-" * 50)

    tunable_params_stastic(trainer.model)
    print(f"Total time: {time.time() - start_time:.2f}s")

    if args.save_dir:
        torch.save(trainer.model.state_dict(), os.path.join(args.save_dir, "model.pt"))
        print(f"Model saved to {args.save_dir}.")


if __name__ == '__main__':
    args = get_args()
    fix_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    main(args)
