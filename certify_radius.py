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

hparams = {
    "device": "cuda",
    "datatype": "images",
    "dataset_path": "data/images/",

    "dataset": "CIFAR10",
    "dataset_mean": [0.4914, 0.4822, 0.4465],
    "dataset_std": [0.2023, 0.1994, 0.2010],

    # model
    "arch": "ResNet50",
    "protected": True,
    "in_channels": 3,
    "out_channels": 10,
    "ablate": True,

    # training
    "batch_size_training": 32,
    "batch_size_inference": 300,
    "lr": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "max_epochs": 1,
    "early_stopping": 400,
    "lr_scheduler": "cosine",
    "logging": True,

    "alpha": 0.01,
    "n0": 1_000,
    "n1": 10_000,

    "smoothing_config" : {
        "mode": "ablation_noise",
        "smoothing_distribution": "ablation_gaussian",
        "append_indicator": True,
        "k": 20, #number of ablated pixels
        "block_size": 5,
        "std": 0.25,
        "d": 1024
    }
}

seed = 42
set_random_seed(seed)
train_data, val_data, test_data_small, test_data = load_dataset(hparams, seed=seed)
model = create_image_classifier(hparams)
model = train(model, train_data,val_data, hparams)


model_path = hparams.get('model_path_override', None) or os.path.join(
    hparams.get('checkpoint_dir', './checkpoints'),
    hparams.get('run_name', 'ResNet50_20251023-213621.pt.pt') if str(hparams.get('run_name','')).endswith('.pt') 
    else f"{hparams.get('run_name','/dfs/is/home/x276198/checkpoints/ResNet50_20251023-213621.pt')}.pt"
)

checkpoint = torch.load(model_path, map_location=hparams.get('device', 'cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(hparams.get('device', 'cpu')).eval()

pre_votes, targets = smooth_image_classifier(hparams, model, test_data_small, hparams["n0"])
votes, _ = smooth_image_classifier(hparams, model, test_data_small, hparams["n1"])

y_hat = pre_votes.argmax(1)
y = torch.tensor(targets)
correct = (y_hat == y).numpy()
clean_acc = correct.mean()
print(f"Clean ACC on test subset: {clean_acc}")
certificates = certify(correct, votes, pre_votes, hparams)
density = 0.01
max_eps = 1.3
plot_data = {}

for r in range(1,5):
    xticks = np.arange(0,max_eps+density*10, density)

    cert_accs = []
    for x in xticks:
        radius = (r,x)
        certified = np.array(certificates["multiclass"][radius[0]]) >= radius[1]
        cert_acc = (certified * correct).mean()
        cert_accs.append(cert_acc)
    plot_data[r] = cert_accs
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