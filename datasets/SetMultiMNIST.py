"""
Set-MultiMNIST dataset
Mostly copy-and-paste from https://github.com/Cyanogenoid/dspn and https://github.com/shaohua0116/MultiDigitMNIST
"""
import os
import numpy as np
import random

import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from .MultiMNIST import MultiMNIST


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


class SetMultiMNIST(torch.utils.data.Dataset):
    def __init__(self, digits=None, threshold=0.0, split='train',
                 root="cache/multimnist/", cache_dir=None, mnist_root="cache/mnist/",
                 sample_size=400):
        self.root = root
        self.split = split
        self.digits = digits
        self.in_tr_sample_size = None
        self.in_te_sample_size = None
        self.threshold = threshold
        self.subdirs = None
        self.scale = None
        self.random_subsample = None
        self.input_dim = 2
        self.sample_size = sample_size
        self.cache_dir = cache_dir
        if cache_dir is None:
            self.cache_dir = root
        self.mnist_path = os.path.join(mnist_root, "MNIST/raw")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        if self.split == 'train':
            self.train_multimnist = MultiMNIST(train=True, transform=transform, mnist_path=self.mnist_path, root=self.root)
            self.imgsize = self.train_multimnist.data.shape[1]
            self.train_points = self._process_cache(self.train_multimnist, self.sample_size)
            self._filter(self.digits)
            print(f"Total number of data: {self.train_multimnist.data.shape[0]} -> {len(self.train_multimnist)}")
        else:
            self.test_multimnist = MultiMNIST(train=False, transform=transform, mnist_path=self.mnist_path, root=self.root)
            self.imgsize = self.test_multimnist.data.shape[1]
            self.test_points = self._process_cache(self.test_multimnist, self.sample_size)
            self._filter(self.digits)
            print(f"Total number of data: {self.test_multimnist.data.shape[0]} -> {len(self.test_multimnist)}")
        print("Cardinality: %d" % self.sample_size)

    def image_to_set(self, image):
        xy = (image.squeeze(0) > self.threshold).nonzero().float()  # [M, 2]
        xy = xy[torch.randperm(xy.size(0)), :]  # [M, 2]
        xy = xy + torch.zeros_like(xy).uniform_(0., 1.)
        c = xy.size(0)
        pad = torch.zeros(self.max - c, 2)
        xy = torch.cat([xy.float(), pad], dim=0) / float(self.imgsize)  # scale [0, 1]
        mask = torch.ones(self.max).bool()  # mask of which elements are invalid
        mask[:c].fill_(False)
        return xy, mask

    def _process_cache(self, dataset, sample_size):
        cache_path = os.path.join(self.cache_dir, f"multimnist_{self.split}_{self.threshold}.pth")
        if os.path.exists(cache_path):
            return torch.load(cache_path)
        os.makedirs(self.cache_dir, exist_ok=True)
        np.random.seed(321)
        print("Processing dataset... (random seed fixed to 321)")
        data = []
        idx = 0
        for datapoint in dataset:
            img, label, coord = datapoint
            label = torch.tensor(label)
            coord = torch.tensor(coord)
            s, s_mask = self.image_to_set(img)
            s = s[~s_mask]
            if len(s) < sample_size:
                continue
            s = s[np.random.choice(len(s), sample_size)]
            train_points = s if self.split == 'train' else None
            test_points = s if self.split != 'train' else None
            m = torch.tensor([0., 0., 0.])  # dummy denormalization
            s = torch.tensor([1., 1., 1.])  # dummy denormalization
            cate_idx = label
            sid = None
            mid = None
            data.append({
                'idx': idx,
                'train_points': train_points,
                'test_points': test_points,
                'mean': m, 'std': s, 'cate_idx': cate_idx,
                'sid': sid, 'mid': mid
            })
            idx += 1
        random.Random(42).shuffle(data)
        torch.save(data, cache_path)
        print("Done!")
        return data

    def _filter(self, digits):
        if digits is not None:
            print(f"Use digits {digits}")
            if self.split == 'train':
                self.train_points = [tr for tr in self.train_points if tr['cate_idx'] in digits]
            else:
                self.test_points = [te for te in self.test_points if te['cate_idx'] in digits]
        if self.split == 'train':
            tr_sample_size = max([(~tr['mask']).long().sum() for tr in self.train_points])
            return tr_sample_size
        else:
            te_sample_size = max([(~te['mask']).long().sum() for te in self.test_points])
            return te_sample_size

    @staticmethod
    def get_pc_stats(idx):
        return 0., 1.

    def renormalize(mean, std):
        pass

    def save_statistics(self, save_dir):
        pass

    def __len__(self):
        return len(self.train_points) if self.split == 'train' else len(self.test_points)

    def __getitem__(self, idx):
        return self.train_points[idx] if self.split=='train' else self.test_points[idx]


def collate_fn(batch):
    ret = dict()
    for k, v in batch[0].items():
        ret.update({k: [b[k] for b in batch]})

    s = torch.stack(ret['set'], dim=0)  # [B, N, 2]
    mask = torch.stack(ret['mask'], dim=0).bool()  # [B, N]
    labels = torch.stack(ret['cate_idx'], dim=0)  # [B, N]
    coords = torch.stack(ret['xyxy'], dim=0)  # [B, N]
    cardinality = (~mask).long().sum(dim=-1)  # [B,]

    ret.update({'set': s, 'set_mask': mask, 'cardinality': cardinality,
                'cate_idx':labels, 'xyxy':coords,
                'mean': 0., 'std': 1.})
    return ret


def build(args):
    train_dataset = SetMultiMNIST(digits=args.digits,
                                  split='train',
                                  threshold=args.threshold,
                                  root=args.multimnist_data_dir,
                                  mnist_root=args.mnist_data_dir,
                                  cache_dir=args.multimnist_cache)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              pin_memory=True, sampler=train_sampler, drop_last=True, num_workers=args.num_workers,
                              collate_fn=collate_fn, worker_init_fn=init_np_seed)

    val_dataset = SetMultiMNIST(digits=args.digits,
                                split='val',
                                threshold=args.threshold,
                                root=args.multimnist_data_dir,
                                mnist_root=args.mnist_data_dir,
                                cache_dir=args.multimnist_cache)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, drop_last=False, num_workers=args.num_workers,
                            collate_fn=collate_fn, worker_init_fn=init_np_seed)

    return train_dataset, val_dataset, train_loader, val_loader
