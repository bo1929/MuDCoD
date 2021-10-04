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

args = parser.parse_args()
cell_type = args.cell_type
percentile = args.percentile
verbose = args.verbose

data_path = expt_utils.get_data_path(cell_type, percentile)
msdyn_nw_details = expt_utils.get_msdyn_nw_details(data_path)
adj_nw_details = expt_utils.get_adjacency_details(data_path)
msdyn_nw = expt_utils.get_msdyn_nw(data_path)[:,:,:,:]

num_sbj, th, n, _ = msdyn_nw.shape

muspces, _, _ = expt_utils.get_community_detection_methods(verbose)

z_muspces = np.empty((num_sbj, th, n))

alpha_muspces = 0.05 * np.ones((th, 2))
beta_muspces = 0.05 * np.ones(num_sbj)

expt_utils.log(f"Fit and predict running...")
z_muspces[:, :, :] = muspces.fit_predict(
    msdyn_nw[:, :, :, :],
    alpha=alpha_muspces,
    beta=beta_muspces,
    k_max=(n // 20),
)
outfile = expt_utils.get_result_path(cell_type, percentile) / "pred_communities.npy"
np.save(outfile, z_muspces)
expt_utils.log(f"Community detection results for {cell_type} saved.")
