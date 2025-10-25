import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.dataset import load_dataset
from src.models import create_image_classifier
from src.training import (
    train,
    smooth_image_classifier,
)
from src.cert import certify
from src.utils import set_random_seed
import json
import argparse
from pathlib import Path


def save_votes(path, pre_votes, votes, targets, n0, n1):
    """
    Save votes to a .pt file to preserve tensor dtypes and shapes.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'pre_votes': pre_votes.detach().cpu(),
        'votes': votes.detach().cpu(),
        'targets': torch.as_tensor(targets).detach().cpu(),
        'n0': int(n0),
        'n1': int(n1)
    }
    torch.save(payload, path)

def run_experiment(hparams):

    seed = 42
    set_random_seed(seed)
    train_data, val_data, test_data_small, test_data = load_dataset(hparams, seed=seed)
    model = create_image_classifier(hparams)
    model = train(model, train_data,val_data, hparams)

    model_dir = os.path.join(hparams['checkpoints'],"models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f"{hparams['run_name']}.pt")
    torch.save({'model_state_dict': model.state_dict()}, model_path)
    print(f"Saved model checkpoint to: {model_path}")

    pre_votes, targets = smooth_image_classifier(hparams, model, test_data_small, hparams["n0"])
    votes, _ = smooth_image_classifier(hparams, model, test_data_small, hparams["n1"])

    votes_dir = os.path.join(hparams['checkpoints'],"votes")
    os.makedirs(votes_dir, exist_ok=True)

    save_votes_path=os.path.join(votes_dir,f"{hparams['run_name']}_votes.pt")
    save_votes(save_votes_path, pre_votes, votes, targets, hparams["n0"], hparams["n1"])
    print(f"Saved votes to: {save_votes_path}")

    return pre_votes,votes, targets 
