# Provable Robustness for Norm-Bound Patch Attacks
### This repository contains the code used in my Master's thesis on certifying robustness of image classifiers against norm-bounded patch attacks. It implements randomized ablation/patch smoothing, certification, and plotting of certified accuracy across perturbation strengths.
## Quick start
Clone the repo:
```bash
git clone https://github.com/try1233/masterthesis.git
```
```bash
cd masterthesis
```
Install dependencies:
```bash 
pip install -r requirements.txt
```
Run the certification experiment:
```bash 
python certify_radius.py
```
Find generated plots:
plots/
What the scripts do
certify_radius.py
Loads or trains an image classifier (e.g., ResNet-50 on CIFAR-10).
Performs randomized smoothing (ablation/patch smoothing) to obtain votes.
Computes certified accuracy across patch sizes and L2 perturbation strengths.
Saves a figure to plots/certified_radii.png.
Depending on your configuration, the workflow can:
Train a model from scratch or load a pretrained checkpoint.
Save and reuse precomputed votes to avoid recomputation.
