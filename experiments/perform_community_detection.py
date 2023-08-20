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
    "--output-path",
    required=True,
    type=str,
    help="Path to output results.",
)
parser.add_argument(
    "--name-suffix",
    required=False,
    default="",
    type=str,
    help="Suffix to use for result filename.",
)
parser.add_argument(
    "--method-name",
    type=str,
    required=True,
    default="mudcod",
    choices=["mudcod", "pisces"],
    help="Method for performing community detection.",
)


def log(*args):
    sutils.log(*args)


if __name__ == "__main__":
    args = parser.parse_args()
    adjacency_matrix_path = pathlib.Path(args.adjacency_matrix_path)
    output_path = pathlib.Path(args.adjacency_matrix_path)
    name_suffix = args.name_suffix
    method_name = args.method_name

    if name_suffix:
        name_suffix = "_-_" + name_suffix

    msdyn_nw = np.load(MSDYN_NW_PATH)
    S, T, n, _ = msdyn_nw.shape

    if method_name == "mudcod":
        z_mudcod = np.empty((S, T, n))

        alpha_mudcod = 0.075 * np.ones((T, 2))
        beta_mudcod = 0.1 * np.ones(S)

        log("Fit and predict running...")
        mudcod = community_detection.MuDCoD(verbose=True)
        z_mudcod[:, :, :] = mudcod.fit_predict(
            msdyn_nw,
            alpha=alpha_mudcod,
            beta=beta_mudcod,
            max_K=30,
            n_iter=50,
            opt_K="null",
            monitor_convergence=True,
        )
        outfile_z = output_path / "mudcod_pred_communities{name_suffix}.npy"
        np.save(outfile_z, z_mudcod)
        outfile_obj = output_path / "mudcod_model{name_suffix}.npy"
        jb.dump(mudcod, outfile_model)

    elif method_name == "pisces":
        pisces_list = []
        z_pisces = np.empty((S, T, n))

        alpha_pisces = 0.075 * np.ones((T, 2))

        log("Fit and predict running...")
        pisces = community_detection.PisCES(verbose=True)
        for sbj in range(S):
            z_pisces[sbj, :, :] = pisces.fit_predict(
                msdyn_nw[i],
                alpha=alpha_pisces,
                max_K=30,
                n_iter=50,
                opt_K="null",
                monitor_convergence=True,
            )
            pisces_list.append(copy.deepcopy(pisces))
        outfile_z = output_path / "pisces_pred_communities{name_suffix}.npy"
        np.save(outfile_z, z_pisces)
        outfile_obj = output_path / "pisces_model{name_suffix}.npy"
        jb.dump(pisces_list, outfile_model)
    else:
        raise NotImplementedError

    log(f"Community detection results for {adjacency_matrix_path} saved.")
