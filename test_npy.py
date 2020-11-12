from datasets import get_datasets, synsetid_to_cate
from argparse import ArgumentParser
from pprint import pprint
from metrics.evaluation_metrics import EMD_CD
from metrics.evaluation_metrics import jsd_between_point_cloud_sets as JSD
from metrics.evaluation_metrics import compute_all_metrics
from collections import defaultdict
from models.networks import PointFlow
import os
import torch
import numpy as np
import torch.nn as nn

def evaluate_gen(sample_pcs, ref_pcs):
    results = compute_all_metrics(sample_pcs, ref_pcs, args.batch_size, accelerated_cd=True)
    results = {k: (v.cpu().detach().item()
                   if not isinstance(v, float) else v) for k, v in results.items()}
    pprint(results)

    sample_pcl_npy = sample_pcs.cpu().detach().numpy()
    ref_pcl_npy = ref_pcs.cpu().detach().numpy()
    jsd = JSD(sample_pcl_npy, ref_pcl_npy)
    print("JSD:%s" % jsd)




if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=str, help='folder which have emd_out_*.npys')
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    ref_pcs = np.load(os.path.join(args.dir, 'emd_out_ref.npy'))
    sample_pcs = np.load(os.path.join(args.dir, 'emd_out_smp.npy'))
    ref_pcs = torch.tensor(ref_pcs).cuda()
    sample_pcs = torch.tensor(sample_pcs).cuda()
    with torch.no_grad():
        evaluate_gen(sample_pcs, ref_pcs)
