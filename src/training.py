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

def load_image_dataset(name, path, seed=42):
    if name not in ["CIFAR10"]:
        raise Exception("Not implemented")

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
    short_train_data  = torch.utils.data.random_split(train_data,
                                                    [10000, len(train_data)- 10000],
                                                    generator=generator)[0]
    split = [1000, len(test_data)-1000]
    short_test_data = torch.utils.data.random_split(test_data,
                                                    split,
                                                    generator=generator)[0]
    return train_data, short_train_data, short_test_data, test_data



def train(net, epoch, optimizer, block_size, sigma, noise_type = "gaussian", device='cuda', trainloader=None, mode='ablation'):
    net.train()
    train_loss = 0
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

    normalized_cifar = NormalizeLayer(means=cifar_mean, stds=cifar_std)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        #inputs = normalized_cifar(inputs, batched=True)
        optimizer.zero_grad()
        #outputs = net(inputs)
        pictures = util_func(inputs, block_size, reuse_noise=False, device = device, sigma = sigma,noise_type=noise_type, normalizer = normalized_cifar)#.permute(0, 2, 3, 1)
        #pictures = normalized_cifar(pictures, batched=True)
       
        random_number = random.random()
        if random_number >1: #following for printing images, set 1 to 0 to print Image
            plt.imshow(pictures.cpu().permute(0, 2, 3, 1)[2][:,:,0:3])
            plt.axis('off')  # Turn off axis labels
            plt.show()

        outputs = net(pictures) #add utils. before random_mask
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total += targets.size(0)

        train_loss += loss.item()

        
def test_nominal(net,epoch, block_size, sigma, end_epoch = 50,mode = 'ablation', noise_type = "gaussian", testloader = None, device = 'cuda', checkpoint_dir = 'master_thesis/checkpoints', checkpoint_file = 'master_thesis/checkpoints/ckpt_epoch_{}.pth'):
    print('\nEpoch: %d' % epoch)
    net.eval()
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
            outputs = net(util_func(inputs, block_size, reuse_noise=False, device = device, sigma = sigma,noise_type=noise_type, normalizer=normalized_cifar))
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            test_loss += loss.item()
            total += targets.size(0)
            
    # Save checkpoint.
    if (epoch % 10 == 0):
        acc = 100.*correct/total
        print(acc)
    if (epoch == end_epoch):
        acc = 100.*correct/total
        print(acc)
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch
        }
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        torch.save(state, checkpoint_file.format(epoch))
    if (epoch % 10 == 0):
        acc = 100.*correct/total
        print(acc)
        return acc
    
def test(net, block_size, testloader = None, device = 'cuda'):
    correct = 0
    cert_correct = 0
    cert_incorrect = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            total += targets.size(0)
            predictions,  certyn = predict_and_certify_ablation_no_noise(inputs, net,block_size,size_to_certify = 5, num_classes = 10,threshold =  0.0)

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
