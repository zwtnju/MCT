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
The checkpoints contains all the fine-tuned models according to tasks and baselines.


## Runs

Run `run_mask.py` to start continued pre-training. 
All arguments are located in it, specific whatever you need.

For fine-tuning, some example scripts are as listed in each downstream task directory.
