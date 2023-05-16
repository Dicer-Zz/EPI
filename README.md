# Rehearsal-free Continual Language Learning via Efficient Parameter Isolation

In this repository, we provide the code to reproduce the results presented in our paper, as well as the necessary steps to use it. If you wish to reference our paper, please cite:

*Zhicheng Wang, Yufang Liu, Tao Ji, Xiaoling Wang, Yuanbin Wu, Congcong Jiang, Ye Chao, Zhencong Han, Ling Wang, Xu Shao and Wenqiu Zeng*:
Rehearsal-free Continual Language Learning via Efficient Parameter Isolation, ACL 2023

## Download and pre-processing data

We used the same dataset as [LAMOL](https://github.com/chho33/LAMOL/tree/master) and [IDBR](https://github.com/SALT-NLP/IDBR/tree/main). You can download the raw dataset from [this link](https://drive.google.com/file/d/1rWcgnVcNpwxmBI3c5ovNx-E8XKOEL77S/view).

After downloading, please put the data in `./data` folder, and then run the following command:
```sh
cd ./data
tar -xvzf LAMOL.tar.gz
python ./preprocess.py
```

Regarding the other datasets mentioned in our paper, you can specify them in the command line arguments, and they will be automatically downloaded from Hugging Face.

Data structure after processing the data:
```
./data
├── ag
│   ├── test.csv
│   └── train.csv
├── amazon
│   ├── test.csv
│   └── train.csv
├── dbpedia
│   ├── test.csv
│   └── train.csv
├── preprocess.py
├── yahoo
│   ├── test.csv
│   └── train.csv
└── yelp
    ├── test.csv
    └── train.csv
```

## Requirements

Here is the hardware and software that we used:

```
Ubuntu: 18.04
GPU: RTX 3090 24GB

python: 3.7.11
pytorch: 1.11.0
transformers: 4.23.1
datasets: 2.1.0
```

We recommend that you create a new conda or python environment and install the required packages manually. We also provide a list of all packages and their versions in `./package-list.txt` for checking environment issues.

## Fine-tuning

We provide a run script `./run.py` that you can use by running the following command:

```sh
python run.py \
    --dataset 5ds \
    --method epi \
    --query_mode mahalanobis \
    --prompt_mode prefix \
    --pre_seq_len 16 \
    --n_per_class 2000 \
    --batch_size 32 \
    --lr 0.03 \
    --epochs 5 \
    --gpu_id 0 \
    --seed 42
```

Since prefix-tuning is affected by your environment and machine, we highly recommend that you run the hyperparameter search script in `./scripts` for best performance:

```sh
bash ./scripts/5ds.sh
```
You can modify the corresponding parameters in the script according to your needs, such as base model, task order, training epoch.

For more information on the available arguments, please refer to `./utils/assets.py`.
