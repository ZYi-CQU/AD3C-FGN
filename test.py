from __future__ import print_function
import sys
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import loader
import net
import util
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits
from matplotlib import cm
import matplotlib.pyplot as plt
from time import time
from collections import Counter
from sklearn import preprocessing
from sklearn.cluster import KMeans

# sys.path.append('/home/yangwanli/GAN-DA')

parser = argparse.ArgumentParser()

# Experiment Specification
# environment
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--manualSeed', type=int)
parser.add_argument('--version', default='ganda')
# setting
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--batch_size_stg2', type=int, default=64)
parser.add_argument('--nepoch', type=int, default=100)
parser.add_argument('--epoch_cls', type=int, default=50,
                    help='the eopch of trainging classifier')
parser.add_argument('--epoch_da', type=int, default=100,
                    help='the epoch of training domain')
parser.add_argument('--G_model_path', type=str, default="./models/AD3C_G_CUB.pkl", help='absolute path to .pth netG model')
parser.add_argument('--C_model_path', type=str, default="./models/AD3C_cls_CUB.pkl", help='absolute path to .pth netC model')

# Dataset
# base info
parser.add_argument('--log_path', default='./logda.xls')
parser.add_argument('--data_root', default='/home/data/')
parser.add_argument('--dataset', default='CUB')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--resSize', type=int, default=2048)
parser.add_argument('--attSize', type=int, default=312)
parser.add_argument('--hidSize', type=int, default=512)
parser.add_argument('--anchors', type=int, default=10,
                    help='number of anchors')
# detail
parser.add_argument('--nclass', type=int, default=200)
parser.add_argument('--syn_num ', type=int, default=300)
# process
parser.add_argument('--validation', action='store_true', default=False)
parser.add_argument('--preprocessing', action='store_true', default=True)
parser.add_argument('--standardization', action='store_true', default=False)


args = parser.parse_args()
print(args)

# random setting
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
print("Random Seed: ", args.manualSeed)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

# loading data
data = loader.DataLoader(args)

# initialize net
print('load model......')
netG = torch.load(args.G_model_path)

netC = torch.load(args.C_model_path)

print('starting')

# initialize tensor
test_res = torch.FloatTensor(data.ntest, args.resSize)
test_label = torch.LongTensor(data.ntest)
test_seen_res = torch.FloatTensor(data.test_seen_label.size(0), args.resSize)
test_seen_label = torch.LongTensor(data.test_seen_label.size(0))


if args.cuda:
    netG.cuda()
    netC.cuda()

    test_res = test_res.cuda()
    test_label = test_label.cuda()
    test_seen_res = test_seen_res.cuda()
    test_seen_label = test_seen_label.cuda()


def sample_seen_test():
    batch_res, batch_label = data.seen_sample()
    test_seen_res.copy_(batch_res)
    test_seen_label.copy_(batch_label) 


def sample_test():
    batch_res, batch_label = data.unseen_sample()
    test_res.copy_(batch_res)
    test_label.copy_(batch_label)


def compute_per_class_acc(true_label, predicted_label, target_classes):
    """Compute mean of class-wise accuracy over all given classes.

    For all the class in target classes (of type integer), this method will compute accuracy for each class first and then return the mean of all computed accuracy. Note that all three arguments' elements are of same python data type.

    Args:
        true_label[torch.LongTensor]: ground truth
        predict_label[torch.LongTensor]: the predicted result of model
        target_classes[torch.LongTensor]: the classes for which accuracy is computed

    Returns:
        acc_per_class: the mean of class-wise accuracy
    """
    acc_per_class = 0.0
    for _class in target_classes:
        # _class: torch.LongTensor/tensor(1)
        idx = (true_label == _class)  # idx: torch.BoolTensor
        acc_per_class += torch.sum(true_label[idx] ==
                                   predicted_label[idx]).item() / torch.sum(idx).item()
    acc_per_class /= target_classes.size(0)

    return acc_per_class


######################################################## test ##############################################
netC.eval()
with torch.no_grad():
    # test classifier on unseen features
    sample_test()
    output, _ = netC(test_res)
    # get predict label from torch.max()
    _, pred_label = torch.max(output, 1)
    classes = torch.from_numpy(np.unique(test_label.cpu().numpy()))
    acc_unseen = compute_per_class_acc(test_label, pred_label, classes)
    print('acc on unseen: %.4f' % (acc_unseen))

    sample_seen_test()
    output, _ = netC(test_seen_res)
    # get predict label from torch.max()
    _, pred_label = torch.max(output, 1)
    classes = torch.from_numpy(np.unique(test_seen_label.cpu().numpy()))
    acc_seen = compute_per_class_acc(test_seen_label, pred_label, classes)
    print('acc on seen %.4f' % (acc_seen))

    h = 2*acc_seen*acc_unseen/(acc_unseen+acc_seen)
    print('H:%.4f' % (h))

