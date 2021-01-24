import numpy as np
import torch
from torch.utils import data

from .ShapeNet import ShapeNet15kPointClouds, _get_MN10_datasets_, _get_MN40_datasets_
from .SetMultiMNIST import SetMultiMNIST, build

def get_datasets(args):
    if args.dataset_type == 'shapenet15k':
        tr_dataset = ShapeNet15kPointClouds(
            categories=args.cates, split='train',
            tr_sample_size=args.tr_max_sample_points,
            te_sample_size=args.te_max_sample_points,
            scale=args.dataset_scale, root_dir=args.data_dir,
            standardize_per_shape=args.standardize_per_shape,
            normalize_per_shape=args.normalize_per_shape,
            normalize_std_per_axis=args.normalize_std_per_axis,
            random_subsample=True)
        te_dataset = ShapeNet15kPointClouds(
            categories=args.cates, split='val',
            tr_sample_size=args.tr_max_sample_points,
            te_sample_size=args.te_max_sample_points,
            scale=args.dataset_scale, root_dir=args.data_dir,
            standardize_per_shape=args.standardize_per_shape,
            normalize_per_shape=args.normalize_per_shape,
            normalize_std_per_axis=args.normalize_std_per_axis,
            all_points_mean=tr_dataset.all_points_mean,
            all_points_std=tr_dataset.all_points_std,
        )
    elif args.dataset_type == 'modelnet40_15k':
        tr_dataset, te_dataset = _get_MN40_datasets_(args)
    elif args.dataset_type == 'modelnet10_15k':
        tr_dataset, te_dataset = _get_MN10_datasets_(args)
    elif args.dataset_type == 'multimnist':
        assert args.tr_max_sample_points == args.te_max_sample_points
        return build(args)
    else:
        raise Exception("Invalid dataset type:%s" % args.dataset_type)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(tr_dataset)
    else:
        train_sampler = None

    tr_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True,
        worker_init_fn=init_np_seed)
    te_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False,
        worker_init_fn=init_np_seed)

    return tr_dataset, te_dataset, tr_loader, te_loader, train_sampler


def get_data_loaders(args):
    tr_dataset, te_dataset = get_datasets(args)
    train_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, drop_last=True,
        worker_init_fn=init_np_seed)
    train_unshuffle_loader = data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, drop_last=True,
        worker_init_fn=init_np_seed)
    test_loader = data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, drop_last=False,
        worker_init_fn=init_np_seed)
    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
        'train_unshuffle_loader': train_unshuffle_loader,
    }
    return loaders


def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)
