import argparse
import numpy as np

from itertools import combinations
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics.cluster import adjusted_rand_score

import expt_utils

parser = argparse.ArgumentParser(
    description="Cell-type-specific subject split experiments."
)
parser.add_argument("--verbose", default=False, type=bool)
parser.add_argument(
    "--cell-type",
    required=True,
    type=str,
    help="Cell type that will be used.",
)
parser.add_argument(
    "--percentile",
    required=False,
    type=int,
    default=95,
    help="Percentile used to determine co-expression threshold.",
)
parser.add_argument(
    "--num-splits",
    required=False,
    type=int,
    default=2,
    help="Number of split for subjects.",
)
parser.add_argument(
    "--num-repeats",
    required=False,
    type=int,
    default=1,
    help="Number of repetition of with different randomization in each.",
)

args = parser.parse_args()
cell_type = args.cell_type
num_splits = args.num_splits
num_repeats = args.num_repeats
percentile = args.percentile
verbose = args.verbose

data_path = expt_utils.get_data_path(cell_type, percentile)
msdyn_nw_details = expt_utils.get_msdyn_nw_details(data_path)
adj_nw_details = expt_utils.get_adjacency_details(data_path)
msdyn_nw = expt_utils.get_msdyn_nw(data_path)

num_sbj, th, n, _ = msdyn_nw.shape

z_muspces_folded = np.empty((num_splits * num_repeats, num_sbj, th, n))
z_pisces_folded = np.empty((num_splits * num_repeats, num_sbj, th, n))
z_static_folded = np.empty((num_splits * num_repeats, num_sbj, th, n))

if num_splits * num_repeats == 1:
    raise ValueError("Computing ARI requires at least two split and/or repetition.")

muspces, pisces, static = expt_utils.get_community_detection_methods(verbose)

alpha_pisces = 0.025 * np.ones((th, 2))
alpha_muspces = 0.025 * np.ones((th, 2))

if num_splits > 1:
    rkf = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats)
    splits = rkf.split(msdyn_nw)
else:
    splits = [([idx for idx in range(num_sbj)], ()) for _ in range(num_repeats)]

fold_splits = []

for idx, split in enumerate(splits):
    fold_splits.append(split)
    expt_utils.log(f"Processing fold-{idx+1}...")
    for sbj in range(num_sbj):
        for t in range(th):
            z_static_folded[idx, sbj, t, :] = static.fit_predict(
                msdyn_nw[sbj, t, :, :], k_max=n // 10
            )
        z_pisces_folded[idx, sbj, :, :] = pisces.fit_predict(
            msdyn_nw[sbj, :, :, :], alpha=alpha_pisces, k_max=n // 10
        )
    for split_idx in split:
        if len(split_idx) > 0:
            beta_muspces = 0.025 * np.ones(len(split_idx))
            z_muspces_folded[idx, split_idx, :, :] = muspces.fit_predict(
                msdyn_nw[split_idx, :, :, :],
                alpha=alpha_muspces,
                beta=beta_muspces,
                k_max=(n // 10),
            )
        else:
            pass

ari_static, ari_pisces, ari_muspces = [], [], []

for idx1, idx2 in list(combinations([i for i in range(len(z_static_folded))], 2)):
    for sbj in range(num_sbj):
        for t in range(th):
            ari_static.append(
                adjusted_rand_score(
                    z_static_folded[idx1, sbj, t, :], z_static_folded[idx2, sbj, t, :]
                )
            )
            ari_pisces.append(
                adjusted_rand_score(
                    z_pisces_folded[idx1, sbj, t, :], z_pisces_folded[idx2, sbj, t, :]
                )
            )
            ari_muspces.append(
                adjusted_rand_score(
                    z_muspces_folded[idx1, sbj, t, :], z_muspces_folded[idx2, sbj, t, :]
                )
            )

expt_utils.log(fold_splits)
expt_utils.log(f"mean(ARI)(static):{np.mean(ari_static)}")
expt_utils.log(f"mean(ARI)(pisces):{np.mean(ari_pisces)}")
expt_utils.log(f"mean(ARI)(muspces):{np.mean(ari_muspces)}")
