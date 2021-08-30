import sys
import numpy as np

from pathlib import Path

MAIN_DIR = Path(__file__).absolute().parent.parent
SIMULATION_DIR = MAIN_DIR / "simulations"
RESULT_DIR = MAIN_DIR / "results-aug20"
NW_COMPARE_DIR = RESULT_DIR / "network_comparison"
sys.path.append(str(MAIN_DIR))


import dypoces.dcbm as dcbm  # noqa: E402

from dypoces.utils import sutils  # noqa: E402
from dypoces.nw import Similarity  # noqa: E402

sutils.safe_create_dir(NW_COMPARE_DIR)


def compare_rtime_MuSDynamicDCBM(
    n, k, p_in, p_out, th, r_time_grid, ns, r_subject, case
):
    membership_distances = np.zeros((len(r_time_grid), th, th))
    for i, r_time in enumerate(r_time_grid):
        ms_dyn_dcbm = dcbm.MuSDynamicDCBM(
            n=n,
            k=k,
            p_in=p_in,
            p_out=p_out,
            time_horizon=th,
            num_subjects=ns,
            r_time=r_time,
            r_subject=r_subject,
        )
        adj_ms_series, z_series = ms_dyn_dcbm.simulate_ms_dynamic_dcbm(case=case)
        for ti in range(th):
            for tj in range(ti, th):
                membership_dist_total = 0
                for si in range(ns):
                    membership_dist_total += Similarity.community_membership_distance(
                        z_series[si, ti, :], z_series[si, tj, :]
                    )
                membership_distances[i, ti, tj] = membership_dist_total / ns
        membership_distances[i, :, :] = (
            np.triu(membership_distances[i, :, :], 1)
            + np.triu(membership_distances[i, :, :], 1).T
        )
    return membership_distances


def compare_rsubject_MuSDynamicDCBM(
    n, k, p_in, p_out, th, r_time, ns, r_subject_grid, case
):
    membership_distances = np.zeros((len(r_subject_grid), ns, ns))
    for i, r_subject in enumerate(r_subject_grid):
        ms_dyn_dcbm = dcbm.MuSDynamicDCBM(
            n=n,
            k=k,
            p_in=p_in,
            p_out=p_out,
            time_horizon=th,
            num_subjects=ns,
            r_time=r_time,
            r_subject=r_subject,
        )
        adj_ms_series, z_series = ms_dyn_dcbm.simulate_ms_dynamic_dcbm(case=case)
        for si in range(ns):
            for sj in range(si, ns):
                membership_dist_total = 0
                for ti in range(th):
                    membership_dist_total += Similarity.community_membership_distance(
                        z_series[si, ti, :], z_series[sj, ti, :]
                    )
                membership_distances[i, si, sj] = membership_dist_total / th

        membership_distances[i, :, :] = (
            np.triu(membership_distances[i, :, :], 1)
            + np.triu(membership_distances[i, :, :], 1).T
        )
    return membership_distances
