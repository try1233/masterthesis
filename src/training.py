import matplotlib.pyplot as plt
from src.inference import (
    predict_and_certify_ablation_no_noise,
    random_mask_batch_one_sample_ablation_no_noise,
    random_mask_batch_one_sample_ablation_noise,
    random_mask_batch_one_sample_no_noise_window
)
import torch
import random
import torch.nn as nn
import os
import torchvision.transforms as transforms
import torchvision
criterion = nn.CrossEntropyLoss()



def train(model,train_data,val_data,hparams,device='cuda'):
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=hparams['batch_size_training'],
        shuffle=hparams.get('shuffle', True),
        num_workers=hparams.get('num_workers', 1),
        worker_init_fn=hparams.get('seed_worker', None),
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=hparams.get('batch_size_eval', hparams['batch_size_training']),
        shuffle=False,
        num_workers=hparams.get('num_workers', 1),
        worker_init_fn=hparams.get('seed_worker', None),
        pin_memory=True
    )
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=hparams["lr"],
                                momentum=hparams["momentum"],
                                weight_decay=hparams["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=200)

    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2023, 0.1994, 0.2010]
    mode = hparams["smoothing_config"]["mode"]
    noise = hparams["smoothing_config"].get('noise_type', None)
    std = hparams["smoothing_config"].get('std', 0.0)
    block_size = hparams["smoothing_config"].get('window_size', 5)

    if mode == 'ablation':
        util_func = random_mask_batch_one_sample_ablation_no_noise
    elif mode == 'ablation_noise':
        util_func = random_mask_batch_one_sample_ablation_noise
    elif mode == 'window':
        util_func = random_mask_batch_one_sample_no_noise_window

    normalized_cifar = NormalizeLayer(means=cifar_mean, stds=cifar_std)

    monitor = hparams.get('monitor', 'val_acc')  
    mode = 'max' if monitor == 'val_acc' else 'min'
    best_score = -float('inf') if mode == 'max' else float('inf')
    patience = hparams.get('early_stopping', float('inf'))
    epochs_since_improve = 0

    for epoch in range(hparams['max_epochs']):
        ######### Train #########
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if util_func is not None:
                inputs = util_func(
                    inputs,
                    block_size,
                    reuse_noise=hparams.get('reuse_noise', False),
                    device=device,
                    sigma=std,
                    noise_type=noise,
                    normalizer=normalizer if hparams.get('normalize_cifar', False) else None
                )
            elif hparams.get('normalize_cifar', False):
                inputs = normalizer(inputs)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            correct += (logits.argmax(1) == targets).sum().item()
            total += targets.size(0)

        loss_train = running_loss / max(1, len(train_loader))
        acc_train = correct / max(1, total)

        if hparams['lr_scheduler']:
            if hparams['lr_scheduler'] == 'cosine':
                scheduler.step()
            else:
                scheduler.step(loss_train)

        #########Val##########
        model.eval()
        val_loss_sum, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                if util_func is not None:
                    inputs = util_func(
                        inputs,
                        block_size,
                        reuse_noise=hparams.get('reuse_noise', False),
                        device=device,
                        sigma=std,
                        noise_type=noise,
                        normalizer=normalizer if hparams.get('normalize_cifar', False) else None
                    )
                elif hparams.get('normalize_cifar', False):
                    inputs = normalizer(inputs)
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
                val_loss_sum += loss.item()
                val_correct += (logits.argmax(1) == targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_loss_sum / max(1, len(val_loader))
        val_acc = val_correct / max(1, val_total)



        current = val_acc if monitor == 'val_acc' else val_loss
        improved = (current > best_score) if mode == 'max' else (current < best_score)

        if improved:
            best_score = current
            epochs_since_improve = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            epochs_since_improve += 1

        if hparams.get('logging', True):
            print(f"Epoch {epoch:3d} | "
                  f"train_loss {loss_train:.4f} acc {acc_train:.4f} | "
                  f"val_loss {val_loss:.4f} acc {val_acc:.4f}")

        if epochs_since_improve > patience:
            if hparams.get('logging', True):
                print(f"Early stopping at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    return model.eval()

    
       

        
def test_nominal(model,epoch, block_size, sigma, end_epoch = 50,mode = 'ablation', noise_type = "gaussian", testloader = None, device = 'cuda', checkpoint_dir = 'master_thesis/checkpoints', checkpoint_file = 'master_thesis/checkpoints/ckpt_epoch_{}.pth'):
    print('\nEpoch: %d' % epoch)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2023, 0.1994, 0.2010]
    if mode == 'ablation':
        util_func = random_mask_batch_one_sample_ablation_no_noise
    elif mode == 'ablation_noise':
        util_func = random_mask_batch_one_sample_ablation_noise
    elif mode == 'window':
        util_func = random_mask_batch_one_sample_no_noise_window
    normalized_cifar = NormalizeLayer(means = cifar_mean, stds = cifar_std)
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        #inputs = normalized_cifar(inputs, batched=True)
        with torch.no_grad():
            masked_input = util_func(inputs, block_size, reuse_noise=False, device = device, sigma = sigma,noise_type=noise_type, normalizer=normalized_cifar)
            outputs = model(masked_input)
            loss = criterion(outputs, targets)

            total += targets.size(0)
            test_loss += loss.item() * targets.size(0)

            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()

            test_loss += loss.item()
            total += targets.size(0)

    wrong = total - correct
    acc = (100.0 * correct) / max(1, total)
    test_loss = test_loss / max(1, total)
    
    cert_correct_pct = (100.0 * cert_correct) / max(1, total)
    cert_incorrect_pct = (100.0 * cert_incorrect) / max(1, total)

    
    print(f"Using block size {block_size} with threshold {threshold}")
    print(f"Total images: {total}")
    print(f"Loss: {test_loss:.4f}")
    print(f"Correct: {correct} ({acc:.3f}%)")
    print(f"Wrongly classified: {wrong} ({100.0 - acc:.3f}%)")
   
            
    
    
def test(model, block_size, testloader = None, device = 'cuda'):
    correct = 0
    cert_correct = 0
    cert_incorrect = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            predictions,  certyn = predict_and_certify_ablation_no_noise(inputs, model,block_size,size_to_certify = 5, num_classes = 10,threshold =  0.0)

            correct += (predictions.eq(targets)).sum().item()
            cert_correct += (predictions.eq(targets) & certyn).sum().item()
            cert_incorrect += (~predictions.eq(targets) & certyn).sum().item()


          #  progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d) Cert: %.3f%% (%d/%d)'
           #     %  ((100.*correct)/total, correct, total, (100.*cert_correct)/total, cert_correct, total))
    print('Using block size ' + str(block_size) + ' with threshhold ' + str(0))
    print('Total images: ' + str(total))
    print('Correct: ' + str(correct) + ' (' + str((100.*correct)/total)+'%)')
    print('Certified Correct class: ' + str(cert_correct) + ' (' + str((100.*cert_correct)/total)+'%)')
    print('Certified Wrong class: ' + str(cert_incorrect) + ' (' + str((100.*cert_incorrect)/total)+'%)')

from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
from .smoothing import *


def train_image_classifier(model, train_data, hparams):
    batch_size = hparams['batch_size_training']
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               worker_init_fn=seed_worker,
                                               num_workers=1)

    if 'early_stopping' in hparams:
        early_stopping = hparams['early_stopping']
    else:
        early_stopping = np.inf

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=hparams["lr"],
                                momentum=hparams["momentum"],
                                weight_decay=hparams["weight_decay"])

    scheduler = CosineAnnealingLR(optimizer, T_max=200)

    best_acc = -np.inf
    best_epoch = 0
    best_state = {}

    for epoch in tqdm(range(hparams["max_epochs"])):
        model.train()
        loss_train = 0
        correct = 0
        total = 0
        for (input, y) in tqdm(train_loader):

            if hparams['protected']:
                inputs = []
                # smooth each image during training
                for i in range(input.shape[0]):
                    img = input[i].clone().squeeze()
                    inputs.append(smooth_image(img, hparams).squeeze())
                input = torch.stack(inputs)

            x, y = input.to(hparams["device"]), y.to(hparams["device"])
            optimizer.zero_grad()
            logits = model(x)
            correct += (logits.argmax(1) == y).sum()
            total += len(y)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            loss_train += loss
            optimizer.step()

        loss_train /= len(train_loader)
        acc_train = correct/total

        if hparams['lr_scheduler']:
            if hparams['lr_scheduler'] == 'cosine':
                scheduler.step()
            else:
                scheduler.step(loss_train)

        if acc_train > best_acc:
            best_acc = acc_train
            best_epoch = epoch
            best_state = {key: value.cpu()
                          for key, value in model.state_dict().items()}

            if hparams["logging"]:
                print(f'Epoch {epoch:4}: '
                      f'loss_train: {loss_train.item():.5f}, '
                      f'acc_train: {acc_train.item():.5f} ')

        if epoch - best_epoch > early_stopping:
            if hparams["logging"]:
                print(f"early stopping at epoch {epoch}")
            break

    if hparams["logging"]:
        print('best_epoch', best_epoch)
    model.load_state_dict(best_state)
    return model.eval()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
