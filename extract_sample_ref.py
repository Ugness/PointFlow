from datasets import get_datasets
from args import get_args
from models.networks import PointFlow
import os
import torch
import numpy as np
import torch.nn as nn


def get_test_loader(args):
    _, te_dataset = get_datasets(args)
    if args.resume_dataset_mean is not None and args.resume_dataset_std is not None:
        mean = np.load(args.resume_dataset_mean)
        std = np.load(args.resume_dataset_std)
        te_dataset.renormalize(mean, std)
    loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False)
    return loader


def sample_gen(model, args):
    loader = get_test_loader(args)
    all_sample = []
    all_ref = []
    for data in loader:
        idx_b, te_pc = data['idx'], data['test_points']
        te_pc = te_pc.cuda() if args.gpu is None else te_pc.cuda(args.gpu)
        B, N = te_pc.size(0), te_pc.size(1)
        _, out_pc = model.sample(B, N)

        # denormalize
        m, s = data['mean'].float(), data['std'].float()
        m = m.cuda() if args.gpu is None else m.cuda(args.gpu)
        s = s.cuda() if args.gpu is None else s.cuda(args.gpu)
        out_pc = out_pc * s + m
        te_pc = te_pc * s + m

        all_sample.append(out_pc)
        all_ref.append(te_pc)

    sample_pcs = torch.cat(all_sample, dim=0)
    ref_pcs = torch.cat(all_ref, dim=0)
    print("Generation sample size:%s reference size: %s"
          % (sample_pcs.size(), ref_pcs.size()))

    # Save the generative output
    save_dir = os.path.dirname(args.resume_checkpoint)
    np.save(os.path.join(save_dir, "emd_out_smp.npy"), sample_pcs.cpu().detach().numpy())
    np.save(os.path.join(save_dir, "emd_out_ref.npy"), ref_pcs.cpu().detach().numpy())
    print("Done")
    return


def main(args):
    model = PointFlow(args)

    def _transform_(m):
        return nn.DataParallel(m)

    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)

    print("Resume Path:%s" % args.resume_checkpoint)
    checkpoint = torch.load(args.resume_checkpoint)
    model.load_state_dict(checkpoint)
    model.eval()


    with torch.no_grad():
        sample_gen(model, args)



if __name__ == '__main__':
    args = get_args()
    print(args)
    main(args)
