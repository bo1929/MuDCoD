import sys
import copy
import argparse
import logging
import pathlib
import numpy as np
import joblib as jb
import mudcod.sutils as sutils
import mudcod.community_detection as community_detection

logging.captureWarnings(True)

parser = argparse.ArgumentParser(
    description="Cell-type-specific subject split experiments."
)
parser.add_argument("--verbose", default=False, type=bool)
parser.add_argument(
    "--adjacency-matrix-path",
    required=True,
    type=str,
    help="Path to input adjacency matrix (multi-subject & dynamic) in NPY format.",
)
parser.add_argument(
    "--name-suffix",
    required=False,
    default="",
    type=str,
    help="Suffix to use for result report.",
)

if __name__ == "__main__":
    args = parser.parse_args()
    adjacency_matrix_path = pathlib.Path(args.adjacency_matrix_path)
    name_suffix = args.name_suffix
        
    if name_suffix:
        name_suffix = "_-_" + name_suffix

    cell_type = adjacency_matrix_path.stem.split("-")[1]

    mus_dynamic_adj = np.load(adjacency_matrix_path)[:, :, :, :]
    S, T = mus_dynamic_adj.shape[0], mus_dynamic_adj.shape[1]

    mudcod = community_detection.MuDCoD(verbose=True)

    # Some reasonable alpha-beta grids.
    alpha_values = [0.01, 0.05, 0.1]
    beta_values = [0.01, 0.05, 0.1]
    # Reansoble values for number of iterations and maximum model order.
    max_K = 30
    n_iter = 25

    cv_report = f"alpha_values={alpha_values}" + "\n" + f"beta_values={beta_values}" + "\n"
    cv_report += f"max_K={max_K}" + "\n" + f"n_iter={n_iter}" + "\n"
    cv_report += "========================" + "\n"

    obj_max, alpha, beta = -np.inf, 0, 0
    for cv_alpha in alpha_values:
        for cv_beta in beta_values:
            modu, logllh = mudcod.cross_validation(
                mus_dynamic_adj,
                num_folds=5,
                alpha=cv_alpha * np.ones((T, 2)),
                beta=cv_beta * np.ones(S),
                n_iter=n_iter,
                max_K=max_K,
                opt_K="null",
                n_jobs=-1,
                monitor_convergence=True,
            )

            report = (
                f"Cross validation for alpha={cv_alpha} and beta={cv_beta} ~ "
                f"modularity:{modu}, loglikelihood:{logllh}"
            )
            cv_report += report + "\n"
            if logllh > obj_max:
                alpha = cv_alpha
                beta = cv_beta
                obj_max = logllh

    result_report = f"Chosen values: alpha={alpha} and beta={beta}"
    cv_report += result_report

    with open(f"cv_report{name_suffix}.txt", "w") as f:
        f.write(cv_report)
