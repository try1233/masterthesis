import torch
import numpy as np
from tqdm.auto import tqdm
import torch
import scipy.sparse as sp

import torchvision
from torchvision import transforms


def load_dataset(hparams, seed=42):
    name = hparams["dataset"]
    path = hparams["dataset_path"]

    if hparams["datatype"] == "images":
        return load_image_dataset(hparams, name, path, seed=seed)
   

def load_image_dataset(hparams, name, path, seed=42):
    if name not in ["CIFAR10"]:
        raise Exception("Not implemented")

    torch.multiprocessing.set_sharing_strategy('file_system')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_data = torchvision.datasets.CIFAR10(
        path, download=True, train=True, transform=transform_train)
    test_data = torchvision.datasets.CIFAR10(
        path, download=True, train=False, transform=transforms.ToTensor())

    generator = torch.Generator().manual_seed(seed)
    split = [1000, len(test_data)-1000]
    short_test_data = torch.utils.data.random_split(test_data,
                                                    split,
                                                    generator=generator)[0]
    return train_data, short_test_data, test_data
