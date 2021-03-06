import warnings
import numpy as np

from sklearn.cluster import KMeans
from numpy.linalg import inv, eigvals
from scipy.sparse.linalg import eigs
from scipy.linalg import sqrtm
from copy import deepcopy

from mudcod.nw import Loss, Similarity
from mudcod.utils.sutils import timeit, log
from mudcod.spectral import SpectralClustering


warnings.filterwarnings(action="ignore", category=np.ComplexWarning)

_eps = 10 ** (-10)
CONVERGENCE_CRITERIA = 10 ** (-5)


class MuSPCES(SpectralClustering):
    def __init__(self, verbose=False):
        super().__init__("muspces", verbose=verbose)
        self.convergence_monitor = []

    def fit(
        self,
        adj,
        alpha=None,
        beta=None,
        k_max=None,
        k_opt="empirical",
        n_iter=30,
        degree_correction=True,
        monitor_convergence=False,
    ):
        """
        Parameters
        ----------
        adj
            adjacency matrices with dimention (ns,th,n,n), where
            n is the number of nodes, th is the number of time time steps,
            and ns is the total number of subjects.
        alpha
            smoothing tuning parameter, along time axis, default is
            0.05J(th,2).
        beta
            smoothing tuning parameter, among the subjects, default is
            0.01J(ns).
        k_max
            maximum number of communities, default is n/10.
        n_iter
            number of iteration of pisces, default is 30.
        degree_correction
            Laplacianized, default is 'True'.
            'True' for degree correction.
        monitor_convergence
            if True, method reports convergence based on ||U_{t} - U_{t-1}||
            and |obj_t - obj_{t-1}> at each iteration.

        Returns
        -------
        embeddings: computed spectral embeddings, with shape (ns, th, n, k)
        """
        assert type(adj) in [np.ndarray, np.memmap] and adj.ndim == 4

        self.adj = adj.astype(float)
        self.adj_shape = self.adj.shape
        self.n = self.adj_shape[2]  # or adj_shape[3]
        self.time_horizon = self.adj_shape[1]
        self.num_subjects = self.adj_shape[0]
        self.degree_correction = degree_correction
        self.degree = deepcopy(adj)

        ns = self.num_subjects
        th = self.time_horizon
        n = self.n
        adj = self.adj
        degree = self.degree

        if alpha is None:
            alpha = 0.05 * np.ones((th, 2))
            if self.verbose:
                log(f"alpha is not provided, alpha set to 0.05J({th},2) by default.")
        if beta is None:
            beta = 0.01 * np.ones(ns)
            if self.verbose:
                log(f"beta is not provided, alpha set to 0.01J({ns}) by default.")
        if k_max is None:
            k_max = np.ceil(self.n / 10).astype(int)
            if self.verbose:
                log(f"k_max is not provided, default value is ceil({n}/10).")
        if th < 2:
            raise ValueError(
                "Time horizon must be at least 2, otherwise use static spectral "
                "clustering."
            )
        assert alpha.shape == (th, 2)
        assert beta.shape == (ns,)
        assert k_max > 0

        if self.verbose:
            log(
                "MuSPCES-fit ~ "
                f"#nodes:{self.n}, "
                f"#time:{self.time_horizon}, "
                f"#num-sbj:{self.num_subjects}, "
                f"alpha:{alpha[0,0]}, "
                f"beta:{beta[0]}, "
                f"k_max:{k_max}, "
                f"n_iter:{n_iter}"
            )

        k = np.zeros((ns, th)).astype(int) + k_max
        v_col = np.zeros((ns, th, n, k_max))
        objective = np.zeros((n_iter))
        self.convergence_monitor = []
        diffU = 0

        if self.degree_correction:
            for t in range(self.time_horizon):
                for sbj in range(self.num_subjects):
                    adj_t = self.adj[sbj, t, :, :]
                    dg = np.diag(np.sum(np.abs(adj_t), axis=0) + _eps)
                    sqinv_degree = sqrtm(inv(dg))
                    self.adj[sbj, t, :, :] = sqinv_degree @ adj_t @ sqinv_degree
                    self.degree[sbj, t, :, :] = dg
        else:
            for t in range(self.time_horizon):
                for sbj in range(self.num_subjects):
                    self.degree[sbj, t, :, :] = np.eye(self.n)

        # Initialization of k, v_col.
        for t in range(th):
            for sbj in range(ns):
                adj_t = adj[sbj, t, :, :]
                k[sbj, t] = self.choose_k(
                    adj_t, adj_t, degree[sbj, t, :, :], k_max, opt=k_opt
                )
                _, v_col[sbj, t, :, : k[sbj, t]] = eigs(adj_t, k=k[sbj, t], which="LM")
                if monitor_convergence:
                    diffU = diffU + (
                        Similarity.hamming_distance(
                            adj[sbj, t, :, :],
                            v_col[sbj, t, :, : k[sbj, t]]
                            @ v_col[sbj, t, :, : k[sbj, t]].T,
                        )
                    )
        self.convergence_monitor.append((-np.inf, diffU))

        total_itr = 0
        for itr in range(n_iter):
            total_itr += 1
            v_col_pv = deepcopy(v_col)
            diffU = 0
            for t in range(th):
                v_col_t = v_col_pv[:, t, :, :]
                swp_v_col_t = np.swapaxes(v_col_t, 1, 2)
                mu_u_t = np.mean(v_col_t @ swp_v_col_t, axis=0)
                for sbj in range(ns):
                    if t == 0:
                        adj_t = adj[sbj, t, :, :]
                        v_col_ktn = v_col_pv[sbj, t + 1, :, : k[sbj, t + 1]]
                        s_t = (
                            adj_t
                            + alpha[t, 1] * v_col_ktn @ v_col_ktn.T
                            + beta[sbj] * mu_u_t
                        )
                        k[sbj, t] = self.choose_k(
                            s_t, adj_t, degree[sbj, t, :, :], k_max, opt=k_opt
                        )
                        _, v_col[sbj, t, :, : k[sbj, t]] = eigs(
                            adj_t, k=k[sbj, t], which="LM"
                        )

                    elif t == th - 1:
                        adj_t = adj[sbj, t, :, :]
                        v_col_ktp = v_col_pv[sbj, t - 1, :, : k[sbj, t - 1]]
                        s_t = (
                            adj_t
                            + alpha[t, 0] * v_col_ktp @ v_col_ktp.T
                            + beta[sbj] * mu_u_t
                        )
                        k[sbj, t] = self.choose_k(
                            s_t, adj_t, degree[sbj, t, :, :], k_max, opt=k_opt
                        )
                        _, v_col[sbj, t, :, : k[sbj, t]] = eigs(
                            s_t, k=k[sbj, t], which="LM"
                        )

                    else:
                        adj_t = adj[sbj, t, :, :]
                        v_col_ktp = v_col_pv[sbj, t - 1, :, : k[sbj, t - 1]]
                        v_col_ktn = v_col_pv[sbj, t + 1, :, : k[sbj, t + 1]]
                        s_t = (
                            adj_t
                            + (alpha[t, 0] * v_col_ktp @ v_col_ktp.T)
                            + (alpha[t, 1] * v_col_ktn @ v_col_ktn.T)
                            + beta[sbj] * mu_u_t
                        )
                        k[sbj, t] = self.choose_k(
                            s_t, adj_t, degree[sbj, t, :, :], k_max, opt=k_opt
                        )
                        _, v_col[sbj, t, :, : k[sbj, t]] = eigs(
                            s_t, k=k[sbj, t], which="LM"
                        )
                    eig_val = eigvals(
                        v_col[sbj, t, :, : k[sbj, t]].T
                        @ v_col_pv[sbj, t, :, : k[sbj, t]]
                    )
                    objective[itr] = objective[itr] + np.sum(np.abs(eig_val), axis=0)

                    if monitor_convergence:
                        diffU = diffU + (
                            Similarity.hamming_distance(
                                v_col[sbj, t, :, : k[sbj, t]]
                                @ v_col[sbj, t, :, : k[sbj, t]].T,
                                v_col_pv[sbj, t, :, : k[sbj, t]]
                                @ v_col_pv[sbj, t, :, : k[sbj, t]].T,
                            )
                        )
            if monitor_convergence:
                self.convergence_monitor.append((objective[itr], diffU))

            if self.verbose:
                log(
                    f"Value of objective funciton: {objective[itr]}, at iteration {itr+1}."
                )

            if itr >= 1:
                diff_obj = objective[itr] - objective[itr - 1]
                if abs(diff_obj) < CONVERGENCE_CRITERIA:
                    break

        if (
            (total_itr > 0)
            and (total_itr == n_iter)
            and (objective[-1] - objective[-2] >= CONVERGENCE_CRITERIA)
        ):
            warnings.warn("MuSPCES does not converge!", RuntimeWarning)
            if self.verbose:
                log(
                    f"MuSPCES does not converge for alpha={alpha[0, 0]}, beta={beta[0]}."
                )

        self.embeddings = v_col
        self.model_order_k = k

        return self.embeddings

    def predict(
        self,
    ):
        """
        Parameters
        ----------

        Returns
        -------
        z_series: community prediction for each time point, with shape (ns, th, n).
        """
        ns = self.num_subjects
        th = self.time_horizon
        n = self.n

        if self.verbose:
            log("MuSPCES-predict ~ ")

        z = np.empty((ns, th, n), dtype=int)
        for t in range(th):
            for sbj in range(ns):
                kmeans = KMeans(n_clusters=self.model_order_k[sbj, t])
                z[sbj, t, :] = kmeans.fit_predict(
                    self.embeddings[sbj, t, :, : self.model_order_k[sbj, t]]
                )
        return z

    def fit_predict(
        self,
        adj,
        degree_correction=True,
        alpha=None,
        beta=None,
        k_max=None,
        k_opt="empirical",
        n_iter=30,
        monitor_convergence=False,
    ):
        self.fit(
            adj,
            alpha=alpha,
            beta=beta,
            k_max=k_max,
            k_opt=k_opt,
            n_iter=n_iter,
            degree_correction=degree_correction,
            monitor_convergence=monitor_convergence,
        )
        return self.predict()

    @timeit
    def cross_validation(
        self,
        n_splits=5,
        alpha=None,
        beta=None,
        k_max=None,
        n_iter=30,
        n_jobs=1,
        verbose=False,
    ):
        """
        This is a function for cross validation of PisCES method
        Parameters
        ----------
        n_splits
                number of folds in cross validation
        alpha
            smoothing tuning parameter, along time axis, default is
            0.05J(th,2).
        beta
            smoothing tuning parameter, among the subjects, default is
            0.01J(ns).
        k_max
                maximum number of communities, default is n/10
        n_iter
                number of iteration of pisces, default is 10
        n_jobs
                number of parallel joblib jobs

        Returns
        -------
        modu
                total modularity value for alpha, beta pair
        logllh
                total log-likelihood value for alpha, beta pair
        """
        ns = self.num_subjects
        th = self.time_horizon
        n = self.n
        adj = self.adj

        if self.degree_correction:
            raise ValueError("Adjacency matrix must be unlaplacianized.")
        if alpha is None:
            alpha = 0.05 * np.ones((th, 2))
            if self.verbose:
                log(f"alpha is not provided, default value is 0.05J({th},2).")
        if beta is None:
            beta = 0.01 * np.ones(ns)
            if self.verbose:
                log(f"beta is not provided, default value is 0.01J({ns}).")
        if k_max is None:
            k_max = np.ceil(self.n / 10).astype(int)
            if self.verbose:
                log(f"k_max is not provided, default value is ceil({n}/10).")
        if th < 2:
            raise ValueError(
                "Time horizon must be at least 2, otherwise use static spectral "
                "clustering."
            )
        assert alpha.shape == (th, 2)
        assert beta.shape == (ns,)
        assert k_max > 0

        idx_n = np.arange(n)
        idx = np.c_[np.repeat(idx_n, idx_n.shape), np.tile(idx_n, idx_n.shape)]
        r = np.random.choice(n ** 2, size=n ** 2, replace=False)

        muspces_kwargs = {
            "alpha": alpha,
            "beta": beta,
            "k_max": k_max,
            "n_iter": n_iter,
        }

        def compute_for_split(adj, idx_test, n, th, ns, muspces_kwargs={}):
            cvidx = np.empty((ns, th, n, n))
            adj_train_imputed = np.zeros((ns, th, n, n))

            for t in range(th):
                for sbj in range(ns):
                    cvidx_t = np.zeros((n, n))
                    cvidx_t[idx_test[sbj, t, :, 0], idx_test[sbj, t, :, 1]] = 1
                    cvidx_t = np.triu(cvidx_t) + np.triu(cvidx_t).T
                    cvidx[sbj, t, :, :] = cvidx_t

                    adj_t = deepcopy(adj[sbj, t, :, :])
                    adj_t[idx_test[sbj, t, :, 0], idx_test[sbj, t, :, 1]] = 0
                    adj_t = np.triu(adj_t) + np.triu(adj_t).T
                    adj_train_imputed[sbj, t, :, :] = self.eigen_complete(
                        adj_t, cvidx_t, 10, 10
                    )

            z = self.__class__(verbose=False).fit_predict(
                deepcopy(adj_train_imputed[:, :, :, :]),
                degree_correction=True,
                **muspces_kwargs,
            )

            modu_val, logllh_val = 0, 0
            for t in range(th):
                for sbj in range(ns):
                    modu_val = modu_val + Loss.modularity(
                        adj[sbj, t, :, :],
                        adj_train_imputed[sbj, t, :, :],
                        z[sbj, t, :],
                        cvidx[sbj, t, :, :],
                    )
                    logllh_val = logllh_val + Loss.loglikelihood(
                        adj[sbj, t, :, :],
                        adj_train_imputed[sbj, t, :, :],
                        z[sbj, t, :],
                        cvidx[sbj, t, :, :],
                    )
            return modu_val, logllh_val

        modu = 0
        logllh = 0

        def split_idx_test(split):
            psplit = n ** 2 // n_splits
            test = r[split * psplit : (split + 1) * psplit]
            idx_test = idx[test, :]
            return np.tile(idx_test, (ns, th, 1, 1))

        if n_jobs > 1:
            from joblib import Parallel, delayed

            with Parallel(n_jobs=n_jobs) as parallel:  ## prefer="processes"
                loss_zipped = parallel(
                    delayed(compute_for_split)(
                        adj,
                        split_idx_test(split),
                        n,
                        th,
                        ns,
                        muspces_kwargs=muspces_kwargs,
                    )
                    for split in range(n_splits)
                )
                modu_split, logllh_split = map(np.array, zip(*loss_zipped))
                modu = sum(modu_split)
                logllh = sum(logllh_split)
        else:
            for split in range(n_splits):
                modu_split, logllh_split = compute_for_split(
                    adj, split_idx_test(split), n, th, ns, muspces_kwargs=muspces_kwargs
                )
                modu = modu + modu_split
                logllh = logllh + logllh_split

        if self.verbose:
            log(
                f"Cross validation(alpha={alpha[0,0]},beta={beta[0]}) ~ "
                f"modularity:{modu}, loglikelihood:{logllh}"
            )

        return modu, logllh
