import os
import itertools
import numpy as np

from pathlib import Path
from numpy.random import default_rng


_eps = 10 ** (-5)


class DCBM:
    def __init__(self, n=None, k=None, p_in=None, p_out=None, seed=0):
        assert isinstance(n, int) and n > 1
        assert isinstance(k, int) and k > 1
        assert isinstance(p_in, tuple) and isinstance(p_out, tuple)
        assert all(isinstance(p, float) for p in p_in) and len(p_in) == 2
        assert all(isinstance(p, float) for p in p_out) and len(p_in) == 2
        assert all(p > 0 for p in p_in) and all(p > 0 for p in p_out)
        assert (p_out[1] < p_in[0]) and (p_out[1] + p_in[1] <= 1)
        assert (p_in[0] <= p_in[1]) and (p_out[0] <= p_out[1])

        self.n = n
        self.k = k
        self.p_in = p_in
        self.p_out = p_out
        self.rng = default_rng(seed)

    def _get_random_z(self):
        rng = self.rng
        k = self.k
        n = self.n
        z_init = np.nonzero(rng.multinomial(1, [1 / k] * k, size=n))[1]
        return z_init

    def _evolve_z(self, z_prev, r):
        assert z_prev.shape[0] == self.n
        assert isinstance(r, float) and abs(r) <= 1.0
        rng = self.rng
        n = self.n
        k = self.k
        e_r = rng.binomial(1, r, size=n)
        z_tn = np.nonzero(rng.multinomial(1, [1 / k] * k, size=n))[1]
        z_new = np.where(e_r == 1, z_tn, z_prev)
        return z_new

    def get_adjacency(self, z, connectivity_matrix, psi):
        assert z.shape[0] == self.n
        assert connectivity_matrix.shape[0] == connectivity_matrix.shape[1] == self.k
        rng = self.rng
        n = self.n

        adj = np.zeros((n, n), dtype=int)
        p = np.empty((n, n))

        bz = connectivity_matrix[np.ix_(z[:], z[:])]
        p[:, :] = psi[np.newaxis, :] * psi[:, np.newaxis] * bz
        adj_triu = rng.binomial(1, p)
        adj[:, :] = np.triu(adj_triu, 1) + np.triu(adj_triu, 1).T

        return adj

    def simulate_dcbm(self, z, psi=None):
        assert z.shape[0] == self.n
        rng = self.rng
        n = self.n
        k = self.k
        p_in, p_out = self.p_in, self.p_out

        connectivity_matrix = np.full((k, k), rng.uniform(p_out[0], p_out[1], 1))
        connectivity_matrix[np.diag_indices(k)] = rng.uniform(p_in[0], p_in[1], k)

        if psi is None:
            gamma1 = 1
            gamma0 = 0.5
            # gamma0 = np.sqrt(1 / max(p_in)) - 1 - _eps
            psi = gamma1 * (np.random.permutation(n) / n) + gamma0
        else:
            assert all((psi**2 * max(p_in)) < 1) and all(psi > 0)

        adj = self.get_adjacency(z, connectivity_matrix, psi)

        return adj, z


class DynamicDCBM(DCBM):
    def __init__(
        self,
        n=None,
        k=None,
        p_in=None,
        p_out=None,
        time_horizon=None,
        r_time=None,
        seed=0,
    ):
        assert isinstance(r_time, float) and (r_time < 1) and (r_time >= 0)
        assert isinstance(time_horizon, int) and (time_horizon > 1)
        self.time_horizon = time_horizon
        self.r_time = r_time
        super().__init__(n, k, p_in, p_out)

    def simulate_dynamic_dcbm(self):
        n = self.n
        th = self.time_horizon

        adj_dynamic = np.empty((th, n, n), dtype=int)
        z_dynamic = np.empty((th, n), dtype=int)

        z_init = self._get_random_z()
        adj_dynamic[0, :, :], z_dynamic[0, :] = self.simulate_dcbm(z_init)

        for t in range(1, th):
            z_dynamic[t, :] = self._evolve_z(z_dynamic[t - 1, :], self.r_time)
            adj_dynamic[t, :, :], _ = self.simulate_dcbm(z_dynamic[t, :])

        return adj_dynamic, z_dynamic


class MuSDynamicDCBM(DynamicDCBM):
    def __init__(
        self,
        n=None,
        k=None,
        p_in=None,
        p_out=None,
        time_horizon=None,
        r_time=None,
        num_subjects=None,
        r_subject=None,
        seed=0,
    ):
        assert isinstance(r_subject, float) and (r_subject < 1) and (r_subject >= 0)
        assert isinstance(num_subjects, int) and (num_subjects > 1)
        self.num_subjects = num_subjects
        self.r_subject = r_subject
        super().__init__(n, k, p_in, p_out, time_horizon, r_time)

    def simulate_mus_dynamic_dcbm(self, setting=3):
        n = self.n
        th = self.time_horizon
        num_sbj = self.num_subjects
        adj_mus_dynamic = np.empty((num_sbj, th, n, n), dtype=int)
        z_mus_dynamic = np.empty((num_sbj, th, n), dtype=int)

        # Totally independent subjects, evolve independently.
        if setting == 0:
            for sbj in range(num_sbj):
                (
                    adj_mus_dynamic[sbj, :, :, :],
                    z_mus_dynamic[sbj, :, :],
                ) = self.simulate_dynamic_dcbm()
        # Subjects are siblings at time 0, then they evolve independently.
        elif setting == 1:
            z_init = self._get_random_z()
            adj_mus_dynamic[0, 0, :, :], z_mus_dynamic[0, 0, :] = self.simulate_dcbm(
                z_init
            )
            for sbj in range(1, num_sbj):
                z_mus_dynamic[sbj, 0, :] = self._evolve_z(z_init, self.r_subject)
                adj_mus_dynamic[sbj, 0, :, :], _ = self.simulate_dcbm(
                    z_mus_dynamic[sbj, 0, :]
                )
            for sbj in range(0, num_sbj):
                for t in range(1, th):
                    z_mus_dynamic[sbj, t, :] = self._evolve_z(
                        z_mus_dynamic[sbj, t - 1, :], self.r_time
                    )
                    adj_mus_dynamic[sbj, t, :, :], _ = self.simulate_dcbm(
                        z_mus_dynamic[sbj, t, :]
                    )
        # Subjects are siblings at each time point.
        elif setting == 2:
            (
                adj_mus_dynamic[0, :, :, :],
                z_mus_dynamic[0, :, :],
            ) = self.simulate_dynamic_dcbm()
            for sbj in range(1, num_sbj):
                for t in range(0, th):
                    z_mus_dynamic[sbj, t, :] = self._evolve_z(
                        z_mus_dynamic[0, t, :], self.r_subject
                    )
                    adj_mus_dynamic[sbj, t, :, :], _ = self.simulate_dcbm(
                        z_mus_dynamic[sbj, t, :]
                    )
        # Subjects are parents of each other at time 0, then they evolve independently.
        elif setting == 3:
            z_init = self._get_random_z()
            adj_mus_dynamic[0, 0, :, :], z_mus_dynamic[0, 0, :] = self.simulate_dcbm(
                z_init
            )
            for sbj in range(1, num_sbj):
                z_mus_dynamic[sbj, 0, :] = self._evolve_z(
                    z_mus_dynamic[sbj - 1, 0, :], self.r_subject
                )
                adj_mus_dynamic[sbj, 0, :, :], _ = self.simulate_dcbm(
                    z_mus_dynamic[sbj, 0, :]
                )
            for sbj in range(0, num_sbj):
                for t in range(1, th):
                    z_mus_dynamic[sbj, t, :] = self._evolve_z(
                        z_mus_dynamic[sbj, t - 1, :], self.r_time
                    )
                    adj_mus_dynamic[sbj, t, :, :], _ = self.simulate_dcbm(
                        z_mus_dynamic[sbj, t, :]
                    )
        # Subjects are parents of each other at each time point.
        elif setting == 4:
            (
                adj_mus_dynamic[0, :, :, :],
                z_mus_dynamic[0, :, :],
            ) = self.simulate_dynamic_dcbm()
            for sbj in range(1, num_sbj):
                for t in range(0, th):
                    z_mus_dynamic[sbj, t, :] = self._evolve_z(
                        z_mus_dynamic[sbj - 1, t, :], self.r_subject
                    )
                    adj_mus_dynamic[sbj, t, :, :], _ = self.simulate_dcbm(
                        z_mus_dynamic[sbj, t, :]
                    )
        else:
            raise ValueError(f"Given setting number {setting} is not defined.")

        return adj_mus_dynamic, z_mus_dynamic


if __name__ == "__main__":
    SIMULATION_DATA_PATH = "../simulation_data/"

    r_subject_values = [0.0, 0.2, 0.5]
    r_time_values = [0.0, 0.2, 0.5]
    time_horizon_values = [2, 4, 8]
    num_subject_values = [16]
    mus_dynamic_dcmb_settings = [1, 2]

    num_instances = 1

    nw_parameters = [
        {"n": 100, "k": 2, "p_in": (0.2, 0.25), "p_out": (0.1, 0.1)},
        {"n": 500, "k": 8, "p_in": (0.25, 0.4), "p_out": (0.1, 0.1)},
    ]

    setting_name = {1: "SSoT", 2: "SSoS"}

    for setting, nw_param in itertools.product(
        mus_dynamic_dcmb_settings, nw_parameters
    ):
        for r_subject, r_time in itertools.product(r_subject_values, r_time_values):
            for time_horizon, num_subjects in itertools.product(
                time_horizon_values, num_subject_values
            ):
                n = nw_param["n"]
                k = nw_param["k"]
                p_in = nw_param["p_in"]
                p_out = nw_param["p_out"]

                def str_decimal(num):
                    return str(num % 1).split(".")[-1]

                def p_path(p):
                    return f"{str_decimal(p[0])}btw{str_decimal(p[1])}"

                scenario_dir_path = Path(SIMULATION_DATA_PATH)
                scenario_dir_path /= f"setting{setting_name[setting]}"
                scenario_dir_path /= Path(
                    f"n{n}-k{k}-p_in{p_path(p_in)}-p_out{p_path(p_out)}"
                )
                scenario_dir_path /= (
                    f"r_time{str_decimal(r_time)}-time_horizon{time_horizon}"
                    + f"-r_subject{str_decimal(r_subject)}-num_subjects{num_subjects}"
                )

                if not scenario_dir_path.exists():
                    scenario_dir_path.mkdir(parents=True)

                for i in range(num_instances):
                    adj_mus_dynamic, z_mus_dynamic = MuSDynamicDCBM(
                        n=n,
                        k=k,
                        p_in=p_in,
                        p_out=p_out,
                        time_horizon=time_horizon,
                        r_time=r_time,
                        num_subjects=num_subjects,
                        r_subject=r_subject,
                        seed=i,
                    ).simulate_mus_dynamic_dcbm(setting=setting)

                    network_dir_path = scenario_dir_path / str(i)

                    if not network_dir_path.exists():
                        network_dir_path.mkdir(parents=True)

                    np.save(network_dir_path / "adj_mus_dynamic.npy", adj_mus_dynamic)
                    np.save(network_dir_path / "z_mus_dynamic.npy", z_mus_dynamic)
