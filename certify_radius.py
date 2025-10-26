import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.dataset import load_dataset
from src.models import create_image_classifier
from src.training import (
    train
)
from src.cert import certify
from src.utils import set_random_seed
from experiment import run_experiment
import json
import argparse
from pathlib import Path

def load_votes(path, map_location='cpu'):
    """
    Load votes from a .pt file saved by save_votes.
    Returns: pre_votes (Tensor), votes (Tensor), targets (ndarray), n0,n1
    """
    data = torch.load(path, map_location=map_location)
    pre_votes = data['pre_votes']
    votes = data['votes']
    targets = data['targets']
    n0 = int(data['n0'])
    n1 = int(data['n1'])
    return pre_votes, votes, targets, n0, n1

hparams = {
    "device": "cuda",
    "datatype": "images",
    "dataset_path": "data/images/",
    "checkpoints": "checkpoints/",
    "run_name": "200_epoch",
    "dataset": "CIFAR10",
    "dataset_mean": [0.4914, 0.4822, 0.4465],
    "dataset_std": [0.2023, 0.1994, 0.2010],

    # model
    "arch": "ResNet50",         
    "in_channels": 3,
    "out_channels": 10,
    "ablate": True,
    "protected": True,
    # training
    "batch_size_training": 256,
    "batch_size_inference": 300,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "max_epochs": 200,
    "early_stopping": 40,
    "lr_scheduler": "cosine",
    "logging": True,

    "alpha": 0.01,
    "n0": 1_000,
    "n1": 10_000,

    "smoothing_config" : {
        "mode": "ablation_noise",
        "smoothing_distribution": "patch_smoothing",
        "noise_type":"gaussian",
        "append_indicator": True,
        "k": 20, #number of ablated pixels
        "window_size": 20,
        "std": 0.25,
        "d": 1024
    }
}

pre_votes,votes, targets =  run_experiment(hparams)
#votes_path = "checkpoints/votes/run_votes.pt"
#pre_votes, votes, targets, n0_loaded, n1_loaded = load_votes(votes_path, map_location='cpu')
y_hat = pre_votes.argmax(1)
y = torch.tensor(targets)
correct = (y_hat == y).numpy()
clean_acc = correct.mean()
print(f"Clean ACC on test subset: {clean_acc}")
certificates = certify(correct, votes, pre_votes, hparams)
density = 0.01
max_eps = 1.3
plot_data = {}

r=1
while r in certificates["multiclass"]:
    xticks = np.arange(0,max_eps+density*10, density)

    cert_accs = []
    for x in xticks:
        radius = (r,x)
        certified = np.array(certificates["multiclass"][radius[0]]) >= radius[1]
        cert_acc = (certified * correct).mean()
        cert_accs.append(cert_acc)
    plot_data[r] = cert_accs
    r+=1
sns.set_style("whitegrid")
sns.set_context("notebook")
fig, ax = plt.subplots(1,1)

for radius in plot_data:
    label = f"Perturbed pixels r={radius}"
    plt.plot(xticks, plot_data[radius], label=label,zorder=100-radius)

plt.ylim((0, 0.9))
plt.xlim(0,max_eps)
plt.xticks(np.arange(0,max_eps+0.1, 0.2))
ax.set_xlabel("Perturbation strength $\epsilon$ (L2-distance)")
ax.set_ylabel("Certified Accuracy")
ax.legend()
plt.savefig('plots/certified_radii.png')