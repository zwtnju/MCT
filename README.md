# MCT

## Requirements

The required Python packages for continued pre-training scripts in [`pre_train`](pre_train) 
and fine-tuning scripts in [`downstream_tasks`](downstream_tasks) are listed in [`requirements.txt`](requirements.txt), which can be easily installed via `pip install -r requirements.txt`.

## Datasets, Pre-Trained Models, Fine-tuned Models

All datasets, pre-trained models and fine-tuned models can be downloaded here: 
[Baidu Netdisk](https://pan.baidu.com/s/10Upa_z3UBKo7cJmx0Aw45g?pwd=lhpz).

It contains ptm, dataset, checkpoints folders. 
The ptm contains all the continued pre-trained models for each baseline.
The dataset contains dataset for pre-training, code search, and defect detection.
The checkpoints contain all the fine-tuned models according to tasks and baselines.


## Runs

For pre-train, first run `run.sh` in [`pre_train/dataset`](pre_train/dataset) to download data. 
Then, run `generate_data.py` in [`pre_train`](pre_train) to generate and cache the input data for PTMs.
Last, run `run_mask.py` in [`pre_train`](pre_train) to start continued pre-training. 
You can also use scripts to run these files, and all arguments are located in them, specific whatever you need.

For fine-tuning, some example scripts are as listed in each downstream task directory.
