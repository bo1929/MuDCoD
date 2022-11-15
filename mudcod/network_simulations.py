import itertools
from pathlib import Path

import numpy as np
from numpy.random import default_rng

_eps = 10 ** (-5)


class DCBM:
    def __init__(self, n=None, model_order_K=None, p_in=None, p_out=None, seed=0):
        assert isinstance(n, int) and n > 1
        assert isinstance(model_order_K, int) and model_order_K > 1
        assert isinstance(p_in, tuple) and isinstance(p_out, tuple)
        assert all(isinstance(p, float) for p in p_in) and len(p_in) == 2
        assert all(isinstance(p, float) for p in p_out) and len(p_in) == 2
        assert all(p > 0 for p in p_in) and all(p > 0 for p in p_out)
        assert (p_out[1] < p_in[0]) and (p_out[1] + p_in[1] <= 1)
        assert (p_in[0] <= p_in[1]) and (p_out[0] <= p_out[1])

        self.n = n
        self.model_order_K = model_order_K
        self.p_in = p_in
        self.p_out = p_out
        self.rng = default_rng(seed)

    def _get_random_z(self):
        rng = self.rng
        model_order_K = self.model_order_K
        n = self.n
        z_init = np.nonzero(
            rng.multinomial(1, [1 / model_order_K] * model_order_K, size=n)
        )[1]
        return z_init

    def _evolve_z(self, z_prev, r):
        assert z_prev.shape[0] == self.n
        assert isinstance(r, float) and abs(r) <= 1.0
        rng = self.rng
        n = self.n
        model_order_K = self.model_order_K
        e_r = rng.binomial(1, r, size=n)
        z_tn = np.nonzero(
            rng.multinomial(1, [1 / model_order_K] * model_order_K, size=n)
        )[1]
        z_new = np.where(e_r == 1, z_tn, z_prev)
        return z_new

    def get_adjacency(self, z, connectivity_matrix, psi):
        assert z.shape[0] == self.n
        assert (
            connectivity_matrix.shape[0]
            == connectivity_matrix.shape[1]
            == self.model_order_K
        )
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
        model_order_K = self.model_order_K
        p_in, p_out = self.p_in, self.p_out

        connectivity_matrix = np.full(
            (model_order_K, model_order_K), rng.uniform(p_out[0], p_out[1], 1)
        )
        connectivity_matrix[np.diag_indices(model_order_K)] = rng.uniform(
            p_in[0], p_in[1], model_order_K
        )

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
        model_order_K=None,
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
        super().__init__(n, model_order_K, p_in, p_out)

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
        model_order_K=None,
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
        super().__init__(n, model_order_K, p_in, p_out, time_horizon, r_time)

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
