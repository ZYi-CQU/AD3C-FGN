from __future__ import print_function
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn import preprocessing
from collections import Counter
from time import time
import matplotlib.pyplot as plt
from matplotlib import cm
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
from numpy import *
# import xlwt
# import xlrd
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
matplotlib.use('Agg')
sys.path.append('/home/yangwanli/GAN-DA')

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
parser.add_argument('--epoch_cls', type=int, default=200,
                    help='the eopch of trainging classifier')
parser.add_argument('--epoch_da', type=int, default=100,
                    help='the epoch of training domain')

# Dataset
# base info
parser.add_argument('--log_path', default='./logda.xls')
parser.add_argument('--data_root', default='/home/yangwanli/data/')
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

# Network
parser.add_argument('--ndh', type=int, default=4096)
parser.add_argument('--ngh', type=int, default=4096)
parser.add_argument('--nmh', type=int, default=4096)
parser.add_argument('--ndph', type=int, default=1024)
parser.add_argument('--nz', type=int, default=312)

# Train
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate to train gan')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--ap_lambda', type=float, default=10)
parser.add_argument('--critic_iter', type=int, default=5)
parser.add_argument('--lambda1', type=float, default=10,
                    help='gradient penalty regularizer')
parser.add_argument('--syn_num', type=int, default=200,
                    help='the number of each class of syn_feature')
parser.add_argument('--cls_lr', type=float, default=0.001,
                    help='learning rate to train cls')
parser.add_argument('--gama', type=float, default=0.1,
                    help='the weight of domain adaptation')
parser.add_argument('--delta', type=float, default=0.1,
                    help='the weight cls inwgan-trans')

print(torch.cuda.is_available())
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

print('starting')


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


def syn_sample(train_x, train_y):
    idx = torch.randperm(train_y.size(0))[0: args.batch_size]
    res = train_x[idx]
    label = train_y[idx]
    m_label = map_label(label, data.unseenclasses)
    input_res.copy_(res)
    input_label.copy_(m_label)


def sample_test():
    batch_res, batch_label = data.unseen_sample()
    label = map_label(batch_label, data.unseenclasses)
    test_res.copy_(batch_res)
    test_label.copy_(label)


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, args.resSize)
    syn_label = torch.LongTensor(nclass*num)
    syn_att = torch.FloatTensor(num, args.attSize)
    syn_noise = torch.FloatTensor(num, args.nz)
    '''
    if args.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    '''
        
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(syn_noise, syn_att)
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label


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


def test_G():
    syn_feature, syn_label = generate_syn_feature(
        netG, data.unseenclasses, data.attribute, args.syn_num)
    for i in range(10):
        # generate features
        # sample from syn_feature
        syn_sample(syn_feature, syn_label)
        output, _ = netC(input_res)
        # compute loss
        cls_loss = criterion(output, input_label)
        cls_loss.backward()
        optC.step()
    # test classifier on unseen features
    sample_test()
    netC.eval()
    output, _ = netC(test_res)
    # get predict label from torch.max()
    _, pred_label = torch.max(output, 1)
    classes = torch.from_numpy(np.unique(test_label.cpu().numpy()))
    acc_unseen = compute_per_class_acc(test_label, pred_label, classes)
    print('acc on unseen: %.4f' % (acc_unseen))

def test_C():
    # test classifier on unseen features
    sample_test()
    netC.eval()
    output, _ = netC(test_res)
    # get predict label from torch.max()
    _, pred_label = torch.max(output, 1)
    classes = torch.from_numpy(np.unique(test_label.cpu().numpy()))
    acc_unseen = compute_per_class_acc(test_label, pred_label, classes)
    print('acc on unseen: %.4f' % (acc_unseen))

def plot_with_labels(embeddings, labels, path, truth):
    print('plt...')
    colors = ['#f6aba5', '#ffb99e', '#ffce90', '#fbe68f', '#d8df92', '#9cd9ac',
              '#7eccc1', '#79baca', '#83a7c8', '#a29fc7', '#b89ab8', '#daa0b3',
              '#e7d5d4', '#e9d5cf', '#f6e3ce', '#efe6c6', '#e6e9c6', '#c4e0cb',
              '#bfe0d9', '#c6dde2', '#c2ccd5', '#c9cad5', '#d0c8d1', '#e4d5d9']
    colors = cm.rainbow(np.linspace(0, 1, 5))
    print(embeddings.shape)

    X, Y = embeddings[:, 0], embeddings[:, 1]
    for x, y, s, t in zip(X, Y, labels, truth):
        if t == 1:
            shape = 'o'
            size = 10
        else:
            shape = 'x'
            size = 10
        
        plt.scatter(x, y, color=colors[s], marker=shape, s=size, edgecolors='k', linewidths=0.1)
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)


    plt.savefig(path, dpi=500, bbox_inches='tight')
    plt.cla()


def plot_dis(embeddings, labels, path, truth):
    X, Y = embeddings[:, 0], embeddings[:, 1]
    plt.scatter(X, Y)
    label = np.unique(labels)
    for i in range(len(label)):
        ind = np.where(labels == (label[i]))[0][0]
        plt.annotate(labels[ind], xy=(X[ind], Y[ind]), xytext=(X[ind], Y[ind]))
    plt.savefig('./test.png', dpi=500, bbox_inches='tight')


# init
netG = torch.load('./models/%s_G_CUB.pkl' % args.version, map_location='cpu')
input_res = torch.FloatTensor(args.batch_size, args.resSize)
input_att = torch.FloatTensor(args.batch_size, args.attSize)
input_label = torch.LongTensor(args.batch_size)
test_res = torch.FloatTensor(data.ntest, args.resSize)
test_label = torch.LongTensor(data.ntest)
netC = torch.load('./models/%s_cls_CUB.pkl' % args.version, map_location='cpu')
# netC = net.Classifier(args.resSize, data.ntest_class)
optC = optim.Adam(netC.parameters(), lr=args.cls_lr, betas=(args.beta1, 0.999))
criterion = nn.NLLLoss()

'''
# cuda
netC.cuda()
netG.cuda()
criterion.cuda()
input_res = input_res.cuda()
input_att = input_att.cuda()
input_label = input_label.cuda()
test_res = test_res.cuda()
test_label = test_label.cuda()
'''

# test G
netG.eval()
test_C()

######################################################## plt ##################################################
print('TSNE...')


############################################## tsne unseenclass #######################################
# tsne cluste
# reses, labels = data.next_seen_one_class()
# for i in range(39):
#     res, label = data.next_seen_one_class()
#     reses = torch.cat((reses, res), 0)
#     labels = torch.cat((labels, label), 0)

# labels = map_label(labels, data.seenclasses).numpy()
# truth = torch.ones(len(labels))
# tsne1 = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000)
# embeddings = tsne1.fit_transform(reses.numpy())

# ind = np.where(labels == (np.unique(labels)[0]))[0][0]
# plt_res = embeddings[ind:ind+500, :]
# plt_lab = labels[ind:ind+500]
# for i in range(1, 10, 1): 
#     ind = np.where(labels == (np.unique(labels)[i]))[0][0]
#     plt_res = np.concatenate((plt_res, embeddings[ind:ind+500, :]),axis=0)
#     plt_lab = np.concatenate((plt_lab, labels[ind:ind+500]))

# ind = np.where(labels == (np.unique(labels)[20]))[0][0]
# plt_res = embeddings[0:ind, :]
# plt_lab = labels[0:ind]

# plot_with_labels(plt_res, plt_lab, './test.pdf', truth)

# plot_with_labels(plt_res, plt_lab, './fig/%s_cub/base.pdf' %
#                  args.version, truth)

# data.index_in_epoch = 0

######################################## tsne unseen class and its synthesis samples ###################################
res_re, label_re = data.next_unseen_one_class()
truth_re = torch.ones(label_re.size(0))
res_syn, label_syn = generate_syn_feature(
    netG, torch.unique(label_re), data.attribute, label_re.size(0))
truth_syn = torch.zeros(label_syn.size(0))
reses = torch.cat((res_re, res_syn), 0)
labels = torch.cat((label_re, label_syn), 0)
truth = torch.cat((truth_re, truth_syn), 0)

for i in range(4):
    res_re, label_re = data.next_unseen_one_class()
    # truth_one denotes real
    truth_re = torch.ones(label_re.size(0))
    res_syn, label_syn = generate_syn_feature(
        netG, torch.unique(label_re), data.attribute, label_re.size(0))
    # truth_zero denotes generative
    truth_syn = torch.zeros(label_syn.size(0))
    reses = torch.cat((reses, res_re, res_syn), 0)
    labels = torch.cat((labels, label_re, label_syn), 0)
    truth = torch.cat((truth, truth_re, truth_syn), 0)

tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000)
embeddings = tsne.fit_transform(reses.numpy())
labels = map_label(labels, data.unseenclasses).numpy()
truth = truth.numpy()


#plot_with_labels(embeddings, labels, './fig/%s_cub.pdf' %
#                 args.version, truth)

colors = cm.rainbow(np.linspace(0, 1, 5))
names = ['Albatross', 'Ani', 'Auklet', 'Cormorant', 'Blackbird']

dists = []
label_list = []
for l in np.unique(labels):
    index = np.argwhere(labels==l)
    index = index.reshape(len(index))
    res = embeddings[index]
    tr = truth[index]
    
    index0 = np.argwhere(tr == 0)
    index0 = index0.reshape(len(index0))
    res_gen = res[index0,:]
    plt.scatter(res_gen[:,0], res_gen[:,1], color=colors[l], marker='x', s=10)
    res_gen_mean = np.mean(res_gen, axis=0)
    
    index1 = np.argwhere(tr == 1)
    index1 = index1.reshape(len(index1))
    res_real = res[index1,:]
    p = plt.scatter(res_real[:,0], res_real[:,1], color=colors[l], marker='o', s=10, label=names[l])
    label_list.append(p)
    res_real_mean = np.mean(res_real,axis=0)
    
    d = np.linalg.norm(res_gen_mean-res_real_mean)
    dists.append(d)

l1 = plt.legend(label_list, names)

p1 = plt.scatter(res_real[-1,0], res_real[-1,1], color=colors[l], marker='o', s=10, label='Real')
p2 = plt.scatter(res_gen[-1,0], res_gen[-1,1], color=colors[l], marker='x', s=10, label='Gen')

plt.legend([p1,p2],['Real','Gen'],loc='lower right')

plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
x = -23.2
y = 5

box = {
'facecolor' : '.99',
'edgecolor' : 'k',
'boxstyle' : 'round'
}

dist_mean = mean(dists)
plt.text(x,y,'Mean Distance:%.2f' % dist_mean,bbox=box)

plt.gca().add_artist(l1)
plt.savefig('./fig/%s_cub.pdf' % args.version, dpi=500, bbox_inches='tight')
plt.cla()

