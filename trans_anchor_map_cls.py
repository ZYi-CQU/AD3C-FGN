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
parser.add_argument('--alpha', type=float, default=1,
                    help='the weight of anchor')
parser.add_argument('--beta', type=float, default=1,
                    help='the weight of semantic anchor')
parser.add_argument('--gamma', type=float, default=1,
                    help='the weight cls loss')


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

args.anchors = data.ntrain_class
# initialize net
netG = net.MLP_G(args)

netD = net.MLP_CRITIC(args)

netC = net.Classifier(args.resSize, data.ntest_class+data.ntrain_class)

netUD = net.MLP_uncond_CRITIC(args)

netM = net.Feature_Mapper(args)

netL = net.Location_Mapper(args)

print('starting')

# initialize tensor
input_res = torch.FloatTensor(args.batch_size, args.resSize)
input_att = torch.FloatTensor(args.batch_size, args.attSize)
noise = torch.FloatTensor(args.batch_size, args.nz)
z = torch.FloatTensor(args.batch_size, args.nz)
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
input_label = torch.LongTensor(args.batch_size)
test_res = torch.FloatTensor(data.ntest, args.resSize)
test_label = torch.LongTensor(data.ntest)
test_seen_res = torch.FloatTensor(data.test_seen_label.size(0), args.resSize)
test_seen_label = torch.LongTensor(data.test_seen_label.size(0))
input_res2 = torch.FloatTensor(args.batch_size, args.resSize)
input_att2 = torch.FloatTensor(args.batch_size, args.attSize)
input_label2 = torch.LongTensor(args.batch_size)
seen_label = torch.LongTensor(args.batch_size)
unseen_label = torch.LongTensor(args.batch_size)

# unmapping labels
label_unseen = torch.LongTensor(args.batch_size)
label_seen = torch.LongTensor(args.batch_size)

batch_locat_label = torch.empty(args.batch_size, args.anchors)

optD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optC = optim.Adam(netC.parameters(), lr=args.cls_lr, betas=(args.beta1, 0.999))
optUD = optim.Adam(netUD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optM = optim.Adam(netM.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optL = optim.Adam(netL.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
criterion = nn.NLLLoss()
dis_criterion = nn.NLLLoss()
map_criterion = nn.NLLLoss()

if args.cuda:
    netD.cuda()
    netG.cuda()
    netC.cuda()
    netUD.cuda()
    netM.cuda()
    netL.cuda()
    criterion.cuda()
    dis_criterion.cuda()
    map_criterion.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    input_label = input_label.cuda()
    test_res = test_res.cuda()
    test_label = test_label.cuda()
    input_att2 = input_att2.cuda()
    input_res2 = input_res2.cuda()
    input_label2 = input_label2.cuda()
    batch_locat_label = batch_locat_label.cuda()
    seen_label = seen_label.cuda()
    unseen_label = unseen_label.cuda()
    test_seen_res = test_seen_res.cuda()
    test_seen_label = test_seen_label.cuda()
    label_seen = label_seen.cuda()
    label_unseen = label_unseen.cuda()


def unseen_sample():
    batch_res, batch_label, batch_att = data.next_unseen_batch(args.batch_size)
    label = map_label(batch_label, data.unseenclasses)
    input_res2.copy_(batch_res)


def sample_seen_test():
    batch_res, batch_label = data.seen_sample()
    test_seen_res.copy_(batch_res)
    test_seen_label.copy_(batch_label) 


def sample_unseen_att():
    batch_res, batch_label, batch_att = data.next_unseen_batch(args.batch_size)
    unseen_label.copy_(batch_label)
    label = map_label(batch_label, data.unseenclasses)
    input_label2.copy_(label)
    input_att2.copy_(batch_att)
    label_unseen.copy_(batch_label)


def syn_sample(train_x, train_y):
    idx = torch.randperm(train_y.size(0))[0: args.batch_size]
    res = train_x[idx]
    label = train_y[idx]
    m_label = map_label(label, data.unseenclasses)
    input_res.copy_(res)
    input_label.copy_(m_label)


def seen_syn_sample(train_x, train_y):
    idx = torch.randperm(train_y.size(0))[0: args.batch_size]
    res = train_x[idx]
    label = train_y[idx]
    m_label = map_label(label, data.seenclasses)
    input_res.copy_(res)
    input_label.copy_(m_label)


def sample_test():
    # batch_res, batch_label = data.unseen_sample()
    # label = map_label(batch_label, data.unseenclasses)
    # test_res.copy_(batch_res)
    # test_label.copy_(label)
    batch_res, batch_label = data.unseen_sample()
    test_res.copy_(batch_res)
    test_label.copy_(batch_label)


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i
    return mapped_label


def sample():
    batch_feature, batch_label, batch_att = data.next_seen_batch(
        args.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    seen_label.copy_(batch_label)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))
    label_seen.copy_(batch_label)


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, args.resSize)
    syn_label = torch.LongTensor(nclass*num)
    syn_att = torch.FloatTensor(num, args.attSize)
    syn_noise = torch.FloatTensor(num, args.nz)
    if args.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        output = netG(syn_noise, syn_att)
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    alpha = torch.rand(args.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if args.cuda:
        alpha = alpha.cuda()
    # get the x` between the distribution of real and fake res
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad = True
    if args.cuda:
        interpolates = interpolates.cuda()

    disc_interpolates = netD(interpolates, input_att)

    ones = torch.ones(disc_interpolates.size())
    if args.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1)
                        ** 2).mean() * args.lambda1
    return gradient_penalty


def UN_calc_gradient_penalty(netUD, real_data, fake_data):
    alpha = torch.rand(args.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if args.cuda:
        alpha = alpha.cuda()
    # get the x` between the distribution of real and fake res
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates.requires_grad = True
    if args.cuda:
        interpolates = interpolates.cuda()

    disc_interpolates = netUD(interpolates)

    ones = torch.ones(disc_interpolates.size())
    if args.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1)
                        ** 2).mean() * args.lambda1
    return gradient_penalty


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


def dist_softmax(fake_res,anchors):
    res = fake_res[0]
    reses = res.expand(args.anchors, args.resSize)
    reses = reses.cuda()
    dists = nn.functional.pairwise_distance(reses, anchors, p=2)
    dist_soft = nn.functional.log_softmax(-dists,
                                          dim=0).reshape(1, args.anchors)
    for i in range(1, args.batch_size):
        res = fake_res[i]
        reses = res.expand(args.anchors, args.resSize)
        reses = reses.cuda()
        dists = nn.functional.pairwise_distance(reses, anchors, p=2)
        soft = nn.functional.log_softmax(-dists,
                                         dim=0).reshape(1, args.anchors)
        dist_soft = torch.cat((dist_soft, soft), 0)
    return dist_soft


def map_softmax(fake_att):
    att = fake_att[0]
    atts = att.expand(data.allclsnum,args.attSize)
    dists = nn.functional.pairwise_distance(atts, semantics, p=2)
    dist_soft = nn.functional.log_softmax(-dists,
                                          dim=0).reshape(1, data.allclsnum)
    for i in range(1, fake_att.size(0)):
        att = fake_att[i]
        atts = att.expand(data.allclsnum,args.attSize)
        dists = nn.functional.pairwise_distance(atts, semantics, p=2)
        soft = nn.functional.log_softmax(-dists,
                                            dim=0).reshape(1, data.allclsnum)
        dist_soft = torch.cat((dist_soft,soft), 0) 
    return dist_soft

semantics = data.attribute
semantics=semantics.cuda()
results = []

############################################## the first training stage ###################################
for epoch in range(args.nepoch):

    # establish anchors
    res, label = data.next_seen_one_class()
    anchors = res[np.random.randint(low=0, high=label.size(0), size=1)]
    for i in range(data.ntrain_class-1):
        res, label = data.next_seen_one_class()
        anchor = res[np.random.randint(low=0, high=label.size(0), size=1)]
        anchors = torch.cat((anchors, anchor), 0)
    anchors = anchors.cuda()
    data.index_in_epoch = 0

    for i in range(0, data.ntrain, args.batch_size):
        # set netD's parameters
        for p in netD.parameters():
            p.requires_grad = True
        for p in netUD.parameters():
            p.requires_grad = True
        # train Discriminator
        for iter_d in range(args.critic_iter):
            sample()
            netD.zero_grad()
            # train with real
            criticD_real = netD(input_res, input_att)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)
            # train with fake
            noise.normal_(0, 1)
            fake = netG(noise, input_att)
            # here's the detach() that separate the fake_res from generator to avoid compute the gradient of generator
            criticD_fake = netD(fake.detach(), input_att)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)
            # gradien penalty
            gradient_penalty = calc_gradient_penalty(
                netD, input_res, fake.data, input_att)
            gradient_penalty.backward()
            # loss value
            wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty
            optD.step()
            # train UNCOND_D
            unseen_sample()
            netUD.zero_grad()
            # train real
            criticUD_real = netUD(input_res2)
            criticUD_real = criticUD_real.mean()
            criticUD_real.backward(mone)
            # train fake
            sample_unseen_att()
            noise.normal_(0, 1)
            fake2 = netG(noise, input_att2)
            criticUD_fake = netUD(fake2.detach())
            criticUD_fake = criticUD_fake.mean()
            criticUD_fake.backward(one)
            # loss value
            gradient_penalty = UN_calc_gradient_penalty(
                netUD, input_res2, fake.data)
            gradient_penalty.backward()
            optUD.step()

        # train Generator
        # reset netD's parameters
        for p in netD.parameters():
            p.requires_grad = False
        for p in netUD.parameters():
            p.requires_grad = False
        netG.zero_grad()
        netC.zero_grad()
        netM.zero_grad()
        # get fake
        noise.normal_(0, 1)
        sample()
        fake1 = netG(noise, input_att)
        criticG_fake = netD(fake1, input_att)
        criticG_fake = criticG_fake.mean()
        # get unseen fake
        noise.normal_(0, 1)
        sample_unseen_att()
        fake2 = netG(noise, input_att2)
        criticG_fake_UD = netUD(fake2)
        criticG_fake_UD = criticG_fake_UD.mean()
        # find min_res and minpooling
        # dist_soft = dist_softmax(fake1)
        dist_soft = dist_softmax(fake1,anchors)
        dis_loss = dis_criterion(dist_soft, input_label)
        # map
        syn_res = torch.cat((fake1, fake2), 0)
        map_labels = torch.cat((seen_label, unseen_label), 0)
        fake_att = netM(syn_res)
        map_soft = map_softmax(fake_att)
        map_loss = map_criterion(map_soft, map_labels)
        # compute cls loss
        # output, _ = netC(fake2)
        # cls_loss = criterion(output, input_label2)
        syn_res = torch.cat((input_res, fake2))
        labels = torch.cat((label_seen, label_unseen), 0)
        output, _ = netC(syn_res)
        cls_loss = criterion(output, labels)
        # compute loss
        G_cost = (-(criticG_fake + criticG_fake_UD)) + \
            args.gamma * cls_loss + args.alpha * (dis_loss + args.beta * map_loss)
        G_cost.backward()
        optM.step()
        optG.step()
        optC.step()

    print(args.version, 'epoch: %d/%d Loss_G %.4f dis_loss: %.4f cls: %.4f map_loss: %.4f'
          % (epoch, args.nepoch, G_cost.item(), dis_loss.item(), cls_loss.item(), map_loss.item()))

######################################################## test ##############################################
    netC.eval()
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

    result = [h,acc_seen,acc_unseen]
    results.append(result)
    netC.train()

    max_res = max(results)
    # print(max_res)
    if acc_unseen == max_res[2]:
        print('save model......')
        torch.save(netG, './models/%s_G_%s.pkl' % (args.version, args.dataset))
        torch.save(netC, './models/%s_cls_%s.pkl' % (args.version, args.dataset))
        print('acc on unseen: %.4f, seen: %.4f, h: %.4f' % (acc_unseen, acc_seen, h))
    
print(args.dataset)