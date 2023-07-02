import torch
import h5py
import numpy as np
import scipy.io as sio
from sklearn import preprocessing


class DataLoader(object):
    """Load dataset from *.mat file.

    Attributes:
        attribute:
        train_feature:
        train_label:
        test_unseen_feature:
        test_unseen_label:
        test_seen_feature:
        test_seen_label:
    """

    def __init__(self, args):
        self.args = args

        if args.dataset == 'imageNet1K':
            self.read_matimagenet(args)
        else:
            self.read_matdataset(args)

        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, args):
        data = sio.loadmat(args.data_root + "/" +
                           args.dataset + "/" + args.image_embedding + ".mat")
        feature = data['features'].T
        label = data['labels'].astype(int).squeeze() - 1

        split = sio.loadmat(args.data_root + "/" + args.dataset +
                            "/" + args.class_embedding + "_splits.mat")
        trainval_loc = split['trainval_loc'].squeeze() - 1
        train_loc = split['train_loc'].squeeze() - 1
        val_unseen_loc = split['val_loc'].squeeze() - 1
        test_seen_loc = split['test_seen_loc'].squeeze() - 1
        test_unseen_loc = split['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(split['att'].T).float()

        if args.validation:
            # Train Data
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            # Test Unseen Data
            self.test_unseen_feature = torch.from_numpy(
                feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(
                label[val_unseen_loc]).long()
        elif args.preprocessing:  # no validation, but preprocessing
            if args.standardization:
                scaler = preprocessing.StandardScaler()
            else:
                scaler = preprocessing.MinMaxScaler()

            _train_feature = scaler.fit_transform(feature[trainval_loc])
            _test_seen_feature = scaler.transform(feature[test_seen_loc])
            _test_unseen_feature = scaler.transform(feature[test_unseen_loc])

            # Train Data
            self.train_feature = torch.from_numpy(_train_feature).float()
            mx = self.train_feature.max()
            self.train_feature.mul_(1/mx)
            self.train_label = torch.from_numpy(label[trainval_loc]).long()

            # Test Unseen Data
            self.test_unseen_feature = torch.from_numpy(
                _test_unseen_feature).float()
            self.test_unseen_feature.mul_(1/mx)
            self.test_unseen_label = torch.from_numpy(
                label[test_unseen_loc]).long()

            # Test Seen Data
            self.test_seen_feature = torch.from_numpy(
                _test_seen_feature).float()
            self.test_seen_feature.mul_(1/mx)
            self.test_seen_label = torch.from_numpy(
                label[test_seen_loc]).long()
        else:  # no validation, no preprocessing
            # Train Data
            self.train_feature = torch.from_numpy(
                feature[trainval_loc]).float()
            self.train_label = torch.from_numpy(label[trainval_loc]).long()
            # Test Unseen Data
            self.test_unseen_feature = torch.from_numpy(
                feature[test_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(
                label[test_unseen_loc]).long()
            # Test Seen Data
            self.test_seen_feature = torch.from_numpy(
                feature[test_seen_loc]).float()
            self.test_seen_label = torch.from_numpy(
                label[test_seen_loc]).long()

        self.ntrain = self.train_feature.size(0)
        self.ntest = self.test_unseen_label.size(0)
        self.nunseen_test = self.test_unseen_feature.size(0)

        self.seenclasses = torch.from_numpy(
            np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(
            np.unique(self.test_unseen_label.numpy()))

        self.ntrain_class = self.seenclasses.size(0)  # seen class num
        self.ntest_class = self.unseenclasses.size(0)  # unseen class num
        
        self.allclasses = torch.cat((self.unseenclasses, self.seenclasses),0)
        self.allclsnum = self.allclasses.size(0)

    # TODO rewrite according to read_matdataset()
    def read_matimagenet(self, args):
        """Read Imagenet1K in *.mat file"""
        if args.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()

            matcontent = h5py.File(
                args.data_root + "/" + args.dataset + "/" + args.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(
                np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(
                int).squeeze() - 1
            matcontent.close()

            matcontent = h5py.File(
                '/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(
                matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(
                args.data_root + "/" + args.dataset + "/" + args.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(
                int).squeeze() - 1
            matcontent.close()

        matcontent = sio.loadmat(
            args.data_root + "/" + args.dataset + "/" + args.class_embedding + ".mat")
        self.attribute = torch.from_numpy(matcontent['w2v']).float()
        self.train_feature = torch.from_numpy(feature).float()
        self.train_label = torch.from_numpy(label).long()
        self.test_seen_feature = torch.from_numpy(feature_val).float()
        self.test_seen_label = torch.from_numpy(label_val).long()
        self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        self.test_unseen_label = torch.from_numpy(label_unseen).long()

        self.ntrain = self.train_feature.size(0)
        self.ntest = self.test_unseen_label.size(0)
        self.seenclasses = torch.from_numpy(
            np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(
            np.unique(self.test_unseen_label.numpy()))
        self.seenclasses = torch.from_numpy(
            np.unique(self.train_label.numpy()))
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.allclasses = torch.cat(self.unseenclasses, self.seenclasses)

    def unseen_sample(self):
        batch_res = self.test_unseen_feature
        batch_label = self.test_unseen_label

        return batch_res, batch_label
    def seen_sample(self):
        batch_res = self.test_seen_feature
        batch_label = self.test_seen_label
        
        return batch_res,batch_label

    def next_unseen_one_class(self):
        iclass = self.unseenclasses[self.index_in_epoch]
        idx = self.test_unseen_label.eq(iclass).nonzero().squeeze()
        iclass_feature = self.test_unseen_feature[idx]
        iclass_label = self.test_unseen_label[idx]
        self.index_in_epoch += 1

        return iclass_feature, iclass_label

    def next_seen_one_class(self):
        iclass = self.seenclasses[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1

        return iclass_feature, iclass_label

    def next_seen_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]

        batch_vf = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]

        return batch_vf, batch_label, batch_att

    def next_unseen_batch(self, batch_size):
        idx = torch.randperm(self.nunseen_test)[0:batch_size]

        batch_vf = self.test_unseen_feature[idx]
        batch_label = self.test_unseen_label[idx]
        batch_att = self.attribute[batch_label]

        return batch_vf, batch_label, batch_att

    # not testing
    def next_batch_one_class(self, batch_size):
        """
        """
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.seenclasses[perm] = self.seenclasses[perm]

        iclass = self.seenclasses[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1

        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]]

    def next_label_class(self, label):
        iclass = self.seenclasses[label]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]

        return iclass_feature, iclass_label

    # not testing
    def next_batch_uniform_class(self, batch_size):
        """Select a batch samples by randomly drawing batch_size classes"""
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.seenclasses[idx]

        batch_feature = torch.FloatTensor(
            batch_size, self.train_feature.size(1))
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]]

        return batch_feature, batch_label, batch_att
