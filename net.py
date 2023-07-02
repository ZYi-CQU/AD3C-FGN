import torch
import torch.nn as nn
from reverse import ReverseLayerF


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class MLP_CRITIC(nn.Module):
    def __init__(self, args):
        super(MLP_CRITIC, self).__init__()
        self.fc1 = nn.Linear(args.resSize + args.attSize, args.ndh)
        self.fc2 = nn.Linear(args.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.fc2(h)
        return h


class MLP_uncond_CRITIC(nn.Module):
    def __init__(self, args):
        super(MLP_uncond_CRITIC, self).__init__()
        self.fc1 = nn.Linear(args.resSize, args.ndh)
        self.fc2 = nn.Linear(args.ndh, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()

        self.apply(weights_init)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        h = self.fc2(h)
        return h


class MLP_G(nn.Module):
    def __init__(self, args):
        super(MLP_G, self).__init__()
        self.fc1 = nn.Linear(args.attSize + args.nz, args.ngh)
        self.fc2 = nn.Linear(args.ngh, args.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, noise, att):
        h = torch.cat((noise, att), 1)
        h1 = self.lrelu(self.fc1(h))
        h2 = self.relu(self.fc2(h1))
        return h2


class Classifier(nn.Module):
    def __init__(self, in_size, out_size):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_size, out_size)
        self.logic = nn.LogSoftmax(dim=1)
        self.soft = nn.Softmax(dim=1)

        self.apply(weights_init)

    def forward(self, x):
        h = self.logic(self.fc(x))
        h1 = self.soft(self.fc(x))
        return h, h1


class Feature_Mapper(nn.Module):
    def __init__(self, args):
        super(Feature_Mapper, self).__init__()
        self.fc1 = nn.Linear(args.resSize, args.nmh)
        self.fc2 = nn.Linear(args.nmh, args.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

        self.apply(weights_init)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        h1 = self.relu(self.fc2(h))
        return h1


class Location_Mapper(nn.Module):

    def __init__(self, args):
        super(Location_Mapper, self).__init__()
        self.fc1 = nn.Linear(args.resSize, args.nmh)
        self.fc2 = nn.Linear(args.nmh, args.anchors)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        h1 = self.relu(self.fc2(h))
        return h1
