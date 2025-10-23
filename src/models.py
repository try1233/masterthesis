'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy
import numpy as np
import scipy.stats
import itertools
import statsmodels.stats.proportion
from scipy.special import comb
import random
import matplotlib.pyplot as plt

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    #ablate if we ablate pixels, i.e. use 6 channels instead of 3
    def __init__(self, block, num_blocks, num_classes=10, ablate= False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        if ablate:
            self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(ablate = False):
    return ResNet(BasicBlock, [2,2,2,2], ablate = ablate)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50(ablate=False):
    return ResNet(Bottleneck, [3,4,6,3], ablate = ablate)

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def create_image_classifier(hparams):
    arch = hparams['arch']
    if arch == "ResNet50":
        model = ResNet(Bottleneck,
                       num_blocks=[3, 4, 6, 3],
                       in_channels=hparams["in_channels"],
                       out_channels=hparams["out_channels"])
    elif arch == "ResNet18":
        model = ResNet(Bottleneck,
                       num_blocks=[2, 2, 2, 2],
                       in_channels=hparams["in_channels"],
                       out_channels=hparams["out_channels"])
    else:
        raise Exception("Not implemented")

    if not hparams['protected']:
        normalize = NormalizeLayer(hparams["dataset_mean"],
                                   hparams["dataset_std"])
        model = torch.nn.Sequential(normalize, model)
    return model.to(hparams["device"])




def smoothing_func(mask, sigma=0.5, device='cpu', noise_type='gaussian'):
    """Assuming masking part = 1 and non-masking part = 0,
       masking part gets noise added, non-masking part stays the same"""
    dtype = torch.float32
    if noise_type == "gaussian":
        noise = torch.tensor(np.random.normal(loc=0, scale=sigma, size=mask.shape), dtype=dtype).to(device)
    else:
        noise = torch.tensor(np.random.uniform(low=-sigma, high=sigma, size=mask.shape), dtype=dtype).to(device)

    noise_mask = torch.mul(mask, noise).to(device)
    return noise#noise_mask

def predict_and_certify(inpt, net, block_size, size_to_certify, num_classes, threshold=0.0):
    predictions = torch.zeros(inpt.size(0), num_classes).type(torch.int).cuda()
    batch = inpt.permute(0, 2, 3, 1)  # color channel last

    for pos in range(batch.shape[2]):
        out_c1 = torch.zeros(batch.shape).cuda()
        out_c2 = torch.zeros(batch.shape).cuda()

        if (pos + block_size > batch.shape[2]):
            out_c1[:, :, pos:] = batch[:, :, pos:]
            out_c2[:, :, pos:] = 1. - batch[:, :, pos:]

            out_c1[:, :, :pos + block_size - batch.shape[2]] = batch[:, :, :pos + block_size - batch.shape[2]]
            out_c2[:, :, :pos + block_size - batch.shape[2]] = 1. - batch[:, :, :pos + block_size - batch.shape[2]]
        else:
            out_c1[:, :, pos:pos + block_size] = batch[:, :, pos:pos + block_size]
            out_c2[:, :, pos:pos + block_size] = 1. - batch[:, :, pos:pos + block_size]

        out_c1 = out_c1.permute(0, 3, 1, 2)
        out_c2 = out_c2.permute(0, 3, 1, 2)
        out = torch.cat((out_c1, out_c2), 1)
        softmx = torch.nn.functional.softmax(net(out), dim=1)
        predictions += (softmx >= threshold).type(torch.int).cuda()

   
    predinctionsnp = predictions.cpu().numpy()
    idxsort = numpy.argsort(-predinctionsnp, axis=1, kind='stable')
    valsort = -numpy.sort(-predinctionsnp, axis=1, kind='stable')
    val = valsort[:, 0]
    idx = idxsort[:, 0]
    valsecond = valsort[:, 1]
    idxsecond = idxsort[:, 1]
    num_affected_classifications = (size_to_certify + block_size - 1)
    cert = torch.tensor(((val - valsecond > 2 * num_affected_classifications) | ((val - valsecond == 2 * num_affected_classifications) & (idx < idxsecond)))).cuda()
   



#no_noise_window: window of size block_size in which no noise, around noise added
def random_mask_batch_one_sample_no_noise_window(batch_s, block_size, reuse_noise=False, sigma=0.5, noise_type = "gaussian", device = 'cuda:0'):
    """here window without noise, rest gets noise added"""
    batch_s = batch_s.permute(0, 2, 3, 1)  # color channel last
    batch = torch.zeros(batch_s.shape).to(device)
    out_c1 = torch.ones(batch_s.shape).to(device)
    out_c2 = torch.ones(batch_s.shape).to(device)
    if reuse_noise:
        xcorner = random.randint(0, batch.shape[1] - 1)
        ycorner = random.randint(0, batch.shape[2] - 1)
        if xcorner + block_size > batch.shape[1]:
            if ycorner + block_size > batch.shape[2]:
                out_c1[:, xcorner:, ycorner:] = batch[:, xcorner:, ycorner:]
                out_c2[:, xcorner:, ycorner:] = 1. - batch[:, xcorner:, ycorner:]

                out_c1[:, :xcorner + block_size - batch.shape[1], ycorner:] = batch[:, :xcorner + block_size - batch.shape[1], ycorner:]
                out_c2[:, :xcorner + block_size - batch.shape[1], ycorner:] = 1. - batch[:, :xcorner + block_size - batch.shape[1], ycorner:]

                out_c1[:, xcorner:, :ycorner + block_size - batch.shape[2]] = batch[:, xcorner:, :ycorner + block_size - batch.shape[2]]
                out_c2[:, xcorner:, :ycorner + block_size - batch.shape[2]] = 1. - batch[:, xcorner:, :ycorner + block_size - batch.shape[2]]

                out_c1[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]] = batch[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]]
                out_c2[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]] = 1. - batch[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]]
            else:
                out_c1[:, xcorner:, ycorner: ycorner + block_size] = batch[:, xcorner:, ycorner: ycorner + block_size]
                out_c2[:, xcorner:, ycorner: ycorner + block_size] = 1. - batch[:, xcorner:, ycorner: ycorner + block_size]

                out_c1[:, :xcorner + block_size - batch.shape[1], ycorner: ycorner + block_size] = batch[:, :xcorner + block_size - batch.shape[1], ycorner: ycorner + block_size]
                out_c2[:, :xcorner + block_size - batch.shape[1], ycorner: ycorner + block_size] = 1. - batch[:, :xcorner + block_size - batch.shape[1], ycorner: ycorner + block_size]
        else:
            if ycorner + block_size > batch.shape[2]:
                out_c1[:, xcorner: xcorner + block_size, ycorner:] = batch[:, xcorner: xcorner + block_size, ycorner:]
                out_c2[:, xcorner: xcorner + block_size, ycorner:] = 1. - batch[:, xcorner: xcorner + block_size, ycorner:]

                out_c1[:, xcorner: xcorner + block_size, :ycorner + block_size - batch.shape[2]] = batch[:, xcorner: xcorner + block_size, :ycorner + block_size - batch.shape[2]]
                out_c2[:, xcorner: xcorner + block_size, :ycorner + block_size - batch.shape[2]] = 1. - batch[:, xcorner: xcorner + block_size, :ycorner + block_size - batch.shape[2]]
            else:
                out_c1[:, xcorner: xcorner + block_size, ycorner: ycorner + block_size] = batch[:, xcorner: xcorner + block_size, ycorner: ycorner + block_size]
                out_c2[:, xcorner: xcorner + block_size, ycorner: ycorner + block_size] = 1. - batch[:, xcorner: xcorner + block_size, ycorner: ycorner + block_size]

    else:
        for i in range(batch.shape[0]):
            xcorner = random.randint(0, batch.shape[1] - 1)
            ycorner = random.randint(0, batch.shape[2] - 1)
            if xcorner + block_size > batch.shape[1]:
                if ycorner + block_size > batch.shape[2]:
                    out_c1[i, xcorner:, ycorner:] = batch[i, xcorner:, ycorner:]
                    out_c2[i, xcorner:, ycorner:] = 1. - batch[i, xcorner:, ycorner:]

                    out_c1[i, :xcorner + block_size - batch.shape[1], ycorner:] = batch[i, :xcorner + block_size - batch.shape[1], ycorner:]
                    out_c2[i, :xcorner + block_size - batch.shape[1], ycorner:] = 1. - batch[i, :xcorner + block_size - batch.shape[1], ycorner:]

                    out_c1[i, xcorner:, :ycorner + block_size - batch.shape[2]] = batch[i, xcorner:, :ycorner + block_size - batch.shape[2]]
                    out_c2[i, xcorner:, :ycorner + block_size - batch.shape[2]] = 1. - batch[i, xcorner:, :ycorner + block_size - batch.shape[2]]

                    out_c1[i, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]] = batch[i, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]]
                    out_c2[i, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]] = 1. - batch[i, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[i, xcorner:, ycorner: ycorner + block_size] = batch[i, xcorner:, ycorner: ycorner + block_size]
                    out_c2[i, xcorner:, ycorner: ycorner + block_size] = 1. - batch[i, xcorner:, ycorner: ycorner + block_size]

                    out_c1[i, :xcorner + block_size - batch.shape[1], ycorner: ycorner + block_size] = batch[i, :xcorner + block_size - batch.shape[1], ycorner: ycorner + block_size]
                    out_c2[i, :xcorner + block_size - batch.shape[1], ycorner: ycorner + block_size] = 1. - batch[i, :xcorner + block_size - batch.shape[1], ycorner: ycorner + block_size]
            else:
                if ycorner + block_size > batch.shape[2]:
                    out_c1[i, xcorner: xcorner + block_size, ycorner:] = batch[i, xcorner: xcorner + block_size, ycorner:]
                    out_c2[i, xcorner: xcorner + block_size, ycorner:] = 1. - batch[i, xcorner: xcorner + block_size, ycorner:]

                    out_c1[i, xcorner: xcorner + block_size, :ycorner + block_size - batch.shape[2]] = batch[i, xcorner: xcorner + block_size, :ycorner + block_size - batch.shape[2]]
                    out_c2[i, xcorner: xcorner + block_size, :ycorner + block_size - batch.shape[2]] = 1. - batch[i, xcorner: xcorner + block_size, :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[i, xcorner: xcorner + block_size, ycorner: ycorner + block_size] = batch[i, xcorner: xcorner + block_size, ycorner: ycorner + block_size]
                    out_c2[i, xcorner: xcorner + block_size, ycorner: ycorner + block_size] = 1. - batch[i, xcorner: xcorner + block_size, ycorner: ycorner + block_size]

    out_c1 = out_c1.permute(0, 3, 1, 2)
    out_c2 = out_c2.permute(0, 3, 1, 2)
    out_c2 = 1 - out_c1

    out1 = smoothing_func(out_c1, sigma, device,noise_type = noise_type)
    
    batch_s = batch_s.permute(0,3,1,2)
    #out = torch.cat((out_c1, out_c2), 1)

    return out1 + batch_s

def predict_and_certify_no_noise_window(inpt, net, block_size, size_to_certify, num_classes, threshold=0.0, device='cuda:0'):
    predictions = torch.zeros(inpt.size(0), num_classes).type(torch.int).to(device)
    batch = torch.zeros(inpt.shape).to(device)
    batch = batch.permute(0, 2, 3, 1)  # color channel last
    out_c1 = torch.ones(batch.shape).to(device)
    for xcorner in range(batch.shape[1]):
        for ycorner in range(batch.shape[2]):

            out_c1 = torch.ones(batch.shape).to(device)
            
            if (xcorner + block_size > batch.shape[1]):
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[:, xcorner:, ycorner:] = batch[:, xcorner:, ycorner:]
               
                    out_c1[:, :xcorner + block_size - batch.shape[1], ycorner:] = \
                        batch[:, :xcorner + block_size - batch.shape[1], ycorner:]
                    
                    out_c1[:, xcorner:, :ycorner + block_size - batch.shape[2]] = \
                        batch[:, xcorner:, :ycorner + block_size - batch.shape[2]]
                   
                    out_c1[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]] = \
                        batch[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[:, xcorner:, ycorner:ycorner + block_size] = batch[:, xcorner:, ycorner:ycorner + block_size]

                    out_c1[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size] = \
                        batch[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size]
                   
            else:
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[:, xcorner:xcorner + block_size, ycorner:] = batch[:, xcorner:xcorner + block_size, ycorner:]

                    out_c1[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]] = \
                        batch[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]]
                   
                else:
                    out_c1[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = \
                        batch[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size]
                   
            out_c1 = out_c1.permute(0, 3, 1, 2)
            
            softmx = torch.nn.functional.softmax(net(out_c1), dim=1)
            predictions += (softmx >= threshold).type(torch.int).cuda()

    predinctionsnp = predictions.cpu().numpy()
    idxsort = np.argsort(-predinctionsnp, axis=1, kind='stable')
    valsort = -np.sort(-predinctionsnp, axis=1, kind='stable')
    val = valsort[:, 0]
    idx = idxsort[:, 0]
    valsecond = valsort[:, 1]
    idxsecond = idxsort[:, 1]
    num_affected_classifications = (size_to_certify + block_size - 1) * (size_to_certify + block_size - 1)
    cert = torch.tensor(
        ((val - valsecond > 2 * num_affected_classifications) | (
                    (val - valsecond == 2 * num_affected_classifications) & (idx < idxsecond)))).cuda()
    return torch.tensor(idx).cuda(), cert

#standard ablation of everything except window of size block_size
def random_mask_batch_one_sample_ablation_no_noise(batch, block_size, reuse_noise=False, sigma = None, noise_type=None, device = 'cuda:0', normalizer = None):
    batch = batch.permute(0, 2, 3, 1)  # color channel last
    out_c1 = torch.zeros(batch.shape).to(device)
    out_c2 = torch.zeros(batch.shape).to(device)

    if reuse_noise:
            xcorner = random.randint(0, batch.shape[1] - 1)
            ycorner = random.randint(0, batch.shape[2] - 1)

            if (xcorner + block_size > batch.shape[1]):
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[:, xcorner:, ycorner:] = batch[:, xcorner:, ycorner:]
                    out_c1[:, :xcorner + block_size - batch.shape[1], ycorner:] = batch[:, :xcorner + block_size - batch.shape[1], ycorner:]
                    out_c1[:, xcorner:, :ycorner + block_size - batch.shape[2]] = batch[:, xcorner:, :ycorner + block_size - batch.shape[2]]
                    out_c1[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]] = batch[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[:, xcorner:, ycorner:ycorner + block_size] = batch[:, xcorner:, ycorner:ycorner + block_size]
                    out_c1[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size] = batch[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size]
            else:
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[:, xcorner:xcorner + block_size, ycorner:] = batch[:, xcorner:xcorner + block_size, ycorner:]
                    out_c1[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]] = batch[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = batch[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size]

    else:
        for i in range(batch.shape[0]):
            xcorner = random.randint(0, batch.shape[1] - 1)
            ycorner = random.randint(0, batch.shape[2] - 1)

            if (xcorner + block_size > batch.shape[1]):
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[i, xcorner:, ycorner:] = batch[i, xcorner:, ycorner:]
                    out_c1[i, :xcorner + block_size - batch.shape[1], ycorner:] = batch[i, :xcorner + block_size - batch.shape[1], ycorner:]
                    out_c1[i, xcorner:, :ycorner + block_size - batch.shape[2]] = batch[i, xcorner:, :ycorner + block_size - batch.shape[2]]
                    out_c1[i, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]] = batch[i, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[i, xcorner:, ycorner:ycorner + block_size] = batch[i, xcorner:, ycorner:ycorner + block_size]
                    out_c1[i, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size] = batch[i, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size]
            else:
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[i, xcorner:xcorner + block_size, ycorner:] = batch[i, xcorner:xcorner + block_size, ycorner:]
                    out_c1[i, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]] = batch[i, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[i, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = batch[i, xcorner:xcorner + block_size, ycorner:ycorner + block_size]

    out_c1 = out_c1.permute(0, 3, 1, 2)
    if normalizer is not None:
            out_c1 = normalizer(out_c1, batched=True)
    out_c2 = 1 - out_c1
    out = torch.cat((out_c1, out_c2), 1)

    return out

def predict_and_certify_ablation_no_noise(inpt, net, block_size, size_to_certify, num_classes, threshold=0.0, device='cuda:0', normalizer = None):
    predictions = torch.zeros(inpt.size(0), num_classes).type(torch.int).to(device)
    if normalizer is not None:
        inpt = normalizer(inpt, batched=True)
    batch = inpt.permute(0, 2, 3, 1)  # color channel last
    for xcorner in range(batch.shape[1]):
        for ycorner in range(batch.shape[2]):

            out_c1 = torch.zeros(batch.shape).to(device)
            out_c2 = torch.zeros(batch.shape).to(device)
            if (xcorner + block_size > batch.shape[1]):
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[:, xcorner:, ycorner:] = batch[:, xcorner:, ycorner:]
                    out_c2[:, xcorner:, ycorner:] = 1. - batch[:, xcorner:, ycorner:]

                    out_c1[:, :xcorner + block_size - batch.shape[1], ycorner:] = \
                        batch[:, :xcorner + block_size - batch.shape[1], ycorner:]
                    out_c2[:, :xcorner + block_size - batch.shape[1], ycorner:] = \
                        1. - batch[:, :xcorner + block_size - batch.shape[1], ycorner:]

                    out_c1[:, xcorner:, :ycorner + block_size - batch.shape[2]] = \
                        batch[:, xcorner:, :ycorner + block_size - batch.shape[2]]
                    out_c2[:, xcorner:, :ycorner + block_size - batch.shape[2]] = \
                        1. - batch[:, xcorner:, :ycorner + block_size - batch.shape[2]]

                    out_c1[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]] = \
                        batch[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]]
                    out_c2[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]] = \
                        1. - batch[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[:, xcorner:, ycorner:ycorner + block_size] = batch[:, xcorner:, ycorner:ycorner + block_size]
                    out_c2[:, xcorner:, ycorner:ycorner + block_size] = 1. - batch[:, xcorner:, ycorner:ycorner + block_size]

                    out_c1[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size] = \
                        batch[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size]
                    out_c2[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size] = \
                        1. - batch[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size]
            else:
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[:, xcorner:xcorner + block_size, ycorner:] = batch[:, xcorner:xcorner + block_size, ycorner:]
                    out_c2[:, xcorner:xcorner + block_size, ycorner:] = 1. - batch[:, xcorner:xcorner + block_size, ycorner:]

                    out_c1[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]] = \
                        batch[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]]
                    out_c2[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]] = \
                        1. - batch[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = \
                        batch[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size]
                    out_c2[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = \
                        1. - batch[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size]

            out_c1 = out_c1.permute(0, 3, 1, 2)
            out_c2 = out_c2.permute(0, 3, 1, 2)
            out = torch.cat((out_c1, out_c2), 1)
            softmx = torch.nn.functional.softmax(net(out), dim=1)
            predictions += (softmx >= threshold).type(torch.int).cuda()

    predinctionsnp = predictions.cpu().numpy()
    idxsort = np.argsort(-predinctionsnp, axis=1, kind='stable')
    valsort = -np.sort(-predinctionsnp, axis=1, kind='stable')
    val = valsort[:, 0]
    idx = idxsort[:, 0]
    valsecond = valsort[:, 1]
    idxsecond = idxsort[:, 1]
    num_affected_classifications = (size_to_certify + block_size - 1) * (size_to_certify + block_size - 1)
    cert = torch.tensor(
        ((val - valsecond > 2 * num_affected_classifications) | (
                    (val - valsecond == 2 * num_affected_classifications) & (idx < idxsecond)))).cuda()
    return torch.tensor(idx).cuda(), cert

#standard ablation but with noise added onto window that is not ablated
def random_mask_batch_one_sample_ablation_noise(batch, block_size, reuse_noise=False, sigma = 0.5, device = 'cuda:0', noise_type = "gaussian", normalizer = None):
      # color channel last
    #256,3,32,32
    batch = batch.permute(0, 2, 3, 1)
    #256,32,32,3
    out_c1 = torch.zeros(batch.shape).cuda()
    out_c2 = torch.zeros(batch.shape).cuda()
    batch = batch.permute(0, 3, 1, 2)
    #256,3,32,32
    if reuse_noise:
            
            xcorner = random.randint(0, batch.shape[1] - 1)
            ycorner = random.randint(0, batch.shape[2] - 1)
            noise = smoothing_func(batch[:,:,:],sigma, device, noise_type = noise_type)
            batch[:,:,:] += noise
            if normalizer is not None: #first add noise, then normalize and then ablate
                batch = normalizer(batch, batched=True) 
            batch = batch.permute(0, 2, 3, 1)
            if (xcorner + block_size > batch.shape[1]):
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[:, xcorner:, ycorner:] = batch[:, xcorner:, ycorner:]
                    out_c1[:, :xcorner + block_size - batch.shape[1], ycorner:] = batch[:, :xcorner + block_size - batch.shape[1], ycorner:]
                    out_c1[:, xcorner:, :ycorner + block_size - batch.shape[2]] = batch[:, xcorner:, :ycorner + block_size - batch.shape[2]]
                    out_c1[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]] = batch[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[:, xcorner:, ycorner:ycorner + block_size] = batch[:, xcorner:, ycorner:ycorner + block_size]
                    out_c1[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size] = batch[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size]
            else:
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[:, xcorner:xcorner + block_size, ycorner:] = batch[:, xcorner:xcorner + block_size, ycorner:]
                    out_c1[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]] = batch[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = batch[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size]

    else:

        noise = smoothing_func(batch[:,:,:],sigma, device, noise_type = noise_type)
    
        batch[:,:,:] += noise
        #256,3,32,32
        if normalizer is not None: #first add noise, then normalize and then ablate
                batch = normalizer(batch, batched=True)
        batch = batch.permute(0, 2, 3, 1)
        #256,32,32,3
        for i in range(batch.shape[0]):
            xcorner = random.randint(0, batch.shape[1] - 1)
            ycorner = random.randint(0, batch.shape[2] - 1)
            
            if (xcorner + block_size > batch.shape[1]):
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[i, xcorner:, ycorner:] = batch[i, xcorner:, ycorner:]
                    out_c1[i, :xcorner + block_size - batch.shape[1], ycorner:] = batch[i, :xcorner + block_size - batch.shape[1], ycorner:]
                    out_c1[i, xcorner:, :ycorner + block_size - batch.shape[2]] = batch[i, xcorner:, :ycorner + block_size - batch.shape[2]]
                    out_c1[i, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]] = batch[i, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[i, xcorner:, ycorner:ycorner + block_size] = batch[i, xcorner:, ycorner:ycorner + block_size]
                    out_c1[i, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size] = batch[i, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size]
            else:
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[i, xcorner:xcorner + block_size, ycorner:] = batch[i, xcorner:xcorner + block_size, ycorner:]
                    out_c1[i, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]] = batch[i, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[i, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = batch[i, xcorner:xcorner + block_size, ycorner:ycorner + block_size]

    


    out_c1 = out_c1.permute(0, 3, 1, 2) #permutate back

    
    out_c2 = 1 - out_c1
  
    out = torch.cat((out_c1, out_c2), 1)

    return out

def predict_and_certify_ablation_noise(inpt, net, block_size, size_to_certify, num_classes, threshold=0.0, sigma = 0.5, device='cuda:0', noise_type = "gaussian"):
    predictions = torch.zeros(inpt.size(0), num_classes).type(torch.int).to(device)
    batch = inpt.permute(0, 2, 3, 1)  # color channel last
    for xcorner in range(batch.shape[1]):
        for ycorner in range(batch.shape[2]):

            out_c1 = torch.zeros(batch.shape).to(device)
            out_c2 = torch.zeros(batch.shape).to(device)
            if (xcorner + block_size > batch.shape[1]):
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[:, xcorner:, ycorner:] = batch[:, xcorner:, ycorner:]
                    out_c2[:, xcorner:, ycorner:] = 1. - batch[:, xcorner:, ycorner:]

                    out_c1[:, :xcorner + block_size - batch.shape[1], ycorner:] = \
                        batch[:, :xcorner + block_size - batch.shape[1], ycorner:]
                    out_c2[:, :xcorner + block_size - batch.shape[1], ycorner:] = \
                        1. - batch[:, :xcorner + block_size - batch.shape[1], ycorner:]

                    out_c1[:, xcorner:, :ycorner + block_size - batch.shape[2]] = \
                        batch[:, xcorner:, :ycorner + block_size - batch.shape[2]]
                    out_c2[:, xcorner:, :ycorner + block_size - batch.shape[2]] = \
                        1. - batch[:, xcorner:, :ycorner + block_size - batch.shape[2]]

                    out_c1[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]] = \
                        batch[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]]
                    out_c2[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]] = \
                        1. - batch[:, :xcorner + block_size - batch.shape[1], :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[:, xcorner:, ycorner:ycorner + block_size] = batch[:, xcorner:, ycorner:ycorner + block_size]
                    out_c2[:, xcorner:, ycorner:ycorner + block_size] = 1. - batch[:, xcorner:, ycorner:ycorner + block_size]

                    out_c1[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size] = \
                        batch[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size]
                    out_c2[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size] = \
                        1. - batch[:, :xcorner + block_size - batch.shape[1], ycorner:ycorner + block_size]
            else:
                if (ycorner + block_size > batch.shape[2]):
                    out_c1[:, xcorner:xcorner + block_size, ycorner:] = batch[:, xcorner:xcorner + block_size, ycorner:]
                    out_c2[:, xcorner:xcorner + block_size, ycorner:] = 1. - batch[:, xcorner:xcorner + block_size, ycorner:]

                    out_c1[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]] = \
                        batch[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]]
                    out_c2[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]] = \
                        1. - batch[:, xcorner:xcorner + block_size, :ycorner + block_size - batch.shape[2]]
                else:
                    out_c1[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = \
                        batch[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size]
                    out_c2[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size] = \
                        1. - batch[:, xcorner:xcorner + block_size, ycorner:ycorner + block_size]

            out_c1 = out_c1.permute(0, 3, 1, 2)
            out_c2 = out_c2.permute(0, 3, 1, 2)
            #This part is added: add noise to the window
            mask = out_c1 != 0
            mask = mask.float()
            noise_tensor = smoothing_func(mask, sigma, device, noise_type=noise_type)
            out_c1 += noise_tensor
            out_c2 += noise_tensor
            out = torch.cat((out_c1, out_c2), 1)
            softmx = torch.nn.functional.softmax(net(out), dim=1)
            predictions += (softmx >= threshold).type(torch.int).cuda()

    predinctionsnp = predictions.cpu().numpy()
    idxsort = np.argsort(-predinctionsnp, axis=1, kind='stable')
    valsort = -np.sort(-predinctionsnp, axis=1, kind='stable')
    val = valsort[:, 0]
    idx = idxsort[:, 0]
    valsecond = valsort[:, 1]
    idxsecond = idxsort[:, 1]
    num_affected_classifications = (size_to_certify + block_size - 1) * (size_to_certify + block_size - 1)
    cert = torch.tensor(
        ((val - valsecond > 2 * num_affected_classifications) | (
                    (val - valsecond == 2 * num_affected_classifications) & (idx < idxsecond)))).cuda()
    return torch.tensor(idx).cuda(), cert


def batch_choose(n,k,batches, device = 'cuda:0'):

	out = torch.zeros((batches,k), dtype=torch.long).to(device)
	for i in range(k):
		out[:,i] = torch.randint(0,n-i, (batches,))
		if (i != 0):
			last_boost = torch.zeros(batches, dtype=torch.long).to(device)
			boost = (out[:,:i] <=(out[:,i]+last_boost).unsqueeze(0).t()).sum(dim=1)
			while (boost.eq(last_boost).sum() != batches):
				last_boost = boost
				boost = (out[:,:i] <=(out[:,i]+last_boost).unsqueeze(0).t()).sum(dim=1)
			out[:,i]  += boost

	return out


class NormalizeLayer(torch.nn.Module):
    """
      Normalization layer, code adapted from (Cohen et al., 2019):
      https://github.com/locuslab/smoothing
    """

    def __init__(self, means, stds):
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means)
        self.stds = torch.tensor(stds)

    def forward(self, input, batched=True):
        if batched:
            (batch_size, num_channels, height, width) = input.shape
            means = self.means.repeat(
                (batch_size, height, width, 1)).permute(0, 3, 1, 2)
            stds = self.stds.repeat(
                (batch_size, height, width, 1)).permute(0, 3, 1, 2)
        else:
            (num_channels, height, width) = input.shape
            means = self.means.repeat((height, width, 1)).permute(2, 0, 1)
            stds = self.stds.repeat((height, width, 1)).permute(2, 0, 1)
        return (input - means.to(input.device)) / stds.to(input.device)

