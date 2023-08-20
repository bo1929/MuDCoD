from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

MAIN_PATH = Path("./")
NETWORK_NAME = "XYZ"

NAME_EXPRESSION = f"corr-{NETWORK_NAME}-*-*.csv"

THRESHOLD_PERCENTILE = 95

NETWORK_PATH_EXPRESSION = MAIN_PATH / NAME_EXPRESSION

if __name__ == "__main__":
    network_path_dict = defaultdict(dict)

    for file_path in map(Path, glob(str(NETWORK_PATH_EXPRESSION))):
        _, NETWORK_NAME, donor, time_point = file_path.stem.split("-")
        network_path_dict[donor][time_point] = file_path

    donors = list(network_path_dict.keys())
    mus_dynamic_adj_list = []

    for donor in tqdm(donors):
        dynamic_network_paths_dict = network_path_dict[donor]
        dynamic_adj_list = []
        for time_point, network_path in dynamic_network_paths_dict.items():
            coexp_df = pd.read_csv(network_path).drop(EXCLUDE)
            coexp_df = coexp_df.drop(columns=coexp_df.columns[EXCLUDE])
            coexp = np.abs(coexp_df.to_numpy()[:, :].astype(float))

            n = coexp.shape[0]

            if THRESHOLD_PERCENTILE == 0:
                adj = coexp
                adj[np.diag_indices_from(adj)] = 0
            else:
                coexp_off_diag = coexp[~np.eye(n, dtype=bool)].reshape(-1)
                q_threshold = np.percentile(coexp_off_diag, THRESHOLD_PERCENTILE)
                adj = np.where(coexp > q_threshold, 1, 0).astype(np.bool8)
                adj[np.diag_indices_from(adj)] = False
                for n1, n2 in enumerate(np.argsort(coexp, axis=0)[-2]):
                    adj[n1, n2] = True
                    adj[n2, n1] = True
            dynamic_adj_list.append(adj)

        mus_dynamic_adj_list.append(np.stack(dynamic_adj_list, axis=0))

    mus_dynamic_adj = np.stack(mus_dynamic_adj_list, axis=0)

    if THRESHOLD_PERCENTILE == 0:
        np.save(
            MAIN_PATH / "mus_dynamic_adj-{NETWORK_NAME}-weighted.npy",
            mus_dynamic_adj,
        )
    else:
        np.save(
            MAIN_PATH / f"mus_dynamic_adj-{NETWORK_NAME}-p{THRESHOLD_PERCENTILE}.npy",
            mus_dynamic_adj,
        )
