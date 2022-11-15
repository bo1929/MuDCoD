import warnings
from copy import deepcopy

import numpy as np
from numpy.linalg import eigvals, inv, svd
from scipy.linalg import sqrtm
from scipy.sparse.linalg import eigs
from scipy.special import comb
from sklearn.cluster import KMeans

from mudcod.network_distances import Distances

_eps = 10 ** (-10)
CONVERGENCE_CRITERIA = 10 ** (-5)
warnings.filterwarnings(action="ignore", category=np.ComplexWarning)


class CommunityDetectionMixin:
    def __init__(self, method, verbose=False):
        self.verbose = verbose
        self.method = method
        assert type(method) == str and method.lower() in [
            "PisCES".lower(),
            "MuDCoD".lower(),
            "StaticSpectralCoD".lower(),
        ]
        self._embeddings = None
        self._model_order_K = None

    @property
    def embeddings(self):
        if self._embeddings is None:
            raise ValueError("Embeddings are not computed yet, run 'fit' first.")
        return self._embeddings

    @property
    def model_order_K(self):
        if self._model_order_K is None:
            raise ValueError("Model order K is not computed yet, run 'fit' first.")
        return self._model_order_K

    @embeddings.setter
    def embeddings(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("Embeddings must be instance 'np.ndarray'.")
        else:
            self._embeddings = value

    @model_order_K.setter
    def model_order_K(self, value):
        if self.method in ["PisCES", "MuDCoD"] and not isinstance(value, np.ndarray):
            raise ValueError("Model order K must be instance of 'np.ndarray'.")
        elif self.method in ["StaticSpectralCoD"] and not isinstance(value, int):
            raise ValueError("Model order K must be instance of 'int'.")
        else:
            self._model_order_K = value

    @staticmethod
    def eigen_complete(adj, cv_idx, epsilon, k):
        m = np.zeros(adj.shape)
        while True:
            adj_cv = (adj * (1 - cv_idx)) + (m * cv_idx)
            u, s, vh = svd(adj_cv)
            s[k:] = 0
            m2 = u @ np.diag(s) @ vh
            m2 = np.where(m2 < 0, 0, m2)
            m2 = np.where(m2 > 1, 1, m2)
            dn = np.sqrt(np.sum((m - m2) ** 2))
            if dn < epsilon:
                break
            else:
                m = m2
        return m

    @staticmethod
    def choose_model_order_K(reprs, degrees, max_K, opt="empirical"):
        """
        Predicts number of communities/modules in a network.

        Parameters
        ----------
        reprs : `np.ndarray` of shape (n, n)
            Laplacianized adjacency matrix representation with dimension (n,n).

        degrees : `np.ndarray` of shape (n)
            Diagonal matrix of degree values with dimension (n).

        max_K : `int`
            Determines the maximum number of communities to predict.

        opt_K : `string`, default='empirical'
            Chooses the technique to estimate K, i.e., number of communities.

        Returns
        -------
        model_order_pred : `int`
            Number of modules to consider.

        """
        opt_list = ["null", "empirical", "full"]

        if opt == opt_list[2]:
            model_order_pred = max_K - 1
        else:
            n = reprs.shape[0]
            sorted_eigenvalues = np.sort(eigvals(np.eye(n) - reprs))
            gaprw = np.diff(sorted_eigenvalues)[1:max_K]

            assert isinstance(gaprw, np.ndarray) and gaprw.ndim == 1

            if opt == opt_list[0]:
                pin = np.sum(degrees) / comb(n, 2) / 2
                threshold = 3.5 / pin ** (0.58) / n ** (1.15)
                idx = np.nonzero(gaprw > threshold)[0]
                if idx.shape[0] == 0:
                    model_order_pred = 1
                else:
                    model_order_pred = np.max(idx) + 1
            elif opt == opt_list[1]:
                if (gaprw == 0).any():
                    idx = -2
                else:
                    idx = -1
                model_order_pred = np.argsort(gaprw, axis=0)[idx] + 1
            else:
                raise ValueError(
                    f"Unkown option {opt} is given for predicting number of communities.\n"
                    f"Use one of {opt_list}."
                )

        return int(model_order_pred) + 1

    @staticmethod
    def modularity(adj_test, adj_train, z_pred, cv_idx, resolution=1):
        """
        Calculates modularity given imputed training adjacency matrix,
        community membership predictions and the actual training matrix.

        Parameters
        ----------
        adj_test : `np.ndarray` of shape (n, n)
            Test ajdacency matrix of shape (n,n), corresponding to the actual network.

        adj_train : `np.ndarray` of shape (n, n)
            Training adjacency matrix of shape (n,n); whose edges are removed
            and then imputed, where n is the number of vertices.

        z_pred : np.ndarray of shape (n)
            Predicted community membership label vector of size (n) of each
            vertex to compute loss.

        cv_idx : `np.ndarray` of shape (n, n)
            A marix of size (n,n), indicating the index of removed edges. Value
            of the entry is 1 for test and 0 for training edges.

        Returns
        -------
        modu : `float`
            Modularity value computed based on the predicted community
            memberships.

        """
        modu = 0
        np.fill_diagonal(cv_idx, 0)

        d_out_train = np.sum(adj_train, axis=0)
        d_in_train = np.sum(adj_train, axis=1)
        m = np.sum(cv_idx * adj_test) / 2
        norm = 1 / (np.sum(adj_train) / 2) ** 2

        for i, j in zip(*np.nonzero(cv_idx.astype(bool))):
            if (cv_idx[i, j] > 0) and (z_pred[i] == z_pred[j]) and (i != j):
                c = z_pred[i] = z_pred[j]
                out_degree_sum = np.sum(d_out_train[z_pred == c])
                in_degree_sum = np.sum(d_in_train[z_pred == c])
                modu += adj_test[i, j] / m - (
                    (resolution * out_degree_sum * in_degree_sum * norm)
                    / np.sum(cv_idx[z_pred == c][:, z_pred == c])
                    / 2
                )
        return modu / 2

    @staticmethod
    def loglikelihood(adj_test, adj_train, z_pred, cv_idx):
        """
        Calcualtes loglikelihood given imputed training adjacency matrix,
        community membership predictions and the actual training matrix.

        Parameters
        ----------
        adj_test : `np.ndarray` of shape (n, n)
            Test ajdacency matrix of shape (n,n), corresponding to the actual network.

        adj_train : `np.ndarray` of shape (n, n)
            Training adjacency matrix of shape (n,n); whose edges are removed
            and then imputed, where n is the number of vertices.

        z_pred : np.ndarray of shape (n)
            Predicted community membership label vector of size (n) of each
            vertex to compute loss.

        cv_idx : `np.ndarray` of shape (n, n)
            A marix of size (n,n), indicating the index of removed edges. Value
            of the entry is 1 for test and 0 for training edges.

        Returns
        -------
        logllh : `float`
            DCBM loglikelihood value computed based on the predicted community
            memberships.

        """
        logllh = 0
        num_communities = np.max(z_pred) + 1

        d_out_train = np.sum(adj_train[:, :], axis=0)
        # dp_out_train = d_out_train[:, np.newaxis] @ d_out_train[np.newaxis, :]

        # hatB = np.zeros((num_communities, num_communities), dtype=int)
        hat0 = np.zeros((num_communities, num_communities), dtype=int)

        for comm_k in range(num_communities):
            for comm_l in range(num_communities):
                z_ix_kl = np.ix_(z_pred == comm_k, z_pred == comm_l)
                total_degree = np.sum(adj_train[z_ix_kl])
                # sum_degree_product = np.sum(dp_out_train[z_ix_kl])
                # hatB[comm_k, comm_l] = total_degree / sum_degree_product
                hat0[comm_k, comm_l] = total_degree

        for i, j in zip(*np.nonzero(cv_idx.astype(bool))):
            # prob = d_out_train[i] * d_out_train[j] * hatB[z_pred[i], z_pred[j]]
            prob = (
                d_out_train[i]
                / np.sum(hat0[z_pred[i], :])
                * d_out_train[j]
                / np.sum(hat0[z_pred[j], :])
                * hat0[z_pred[i], z_pred[j]]
                * 0.8
            )

            if prob == 0 or np.isnan(prob):
                prob = _eps
            elif prob >= 1:
                prob = 1 - _eps
            else:
                pass

            logllh = (
                logllh
                + np.log(prob) * (adj_test[i, j] >= _eps)
                + np.log(1 - prob) * (adj_test[i, j] < _eps)
            )

        # return logllh
        return logllh / np.sum(adj_train)


class StaticSpectralCoD(CommunityDetectionMixin):
    def __init__(self, verbose=False):
        super().__init__("StaticSpectralCoD", verbose=verbose)

    def fit(self, adj, max_K=None, opt_K="empirical"):
        """
        Computes the spectral embeddings from the adjacency matrix of the
        network.

        Parameters
        ----------
        adj : `np.ndarray` of shape (n, n)
            Adjacency matrices of size (n, n), n is the number of vertices.

        max_K : `int`, default=n/10
            Determines the maximum number of communities to predict.

        opt_K : `string`, default='empirical'
            Chooses the technique to estimate k, i.e., number of communities.

        Returns
        -------
        embeddings : `np.ndarray` of shape (n, max_K)
            Computed spectral embedding of the adjacency matrix.

        """
        self.adj = adj.astype(float)
        self.num_vertices = self.adj.shape[0]

        self.degrees = np.empty(self.adj.shape[:-1])
        self.lapl_adj = np.empty_like(self.adj)

        if max_K is None:
            max_K = np.ceil(self.num_vertices / 10).astype(int)

        assert type(self.adj) in [np.ndarray, np.memmap] and self.adj.ndim == 2
        assert self.adj.shape[0] == self.adj.shape[1]

        n = self.num_vertices

        self.degrees = np.sum(np.abs(self.adj), axis=0) + _eps
        sqinv_degree = sqrtm(inv(np.diag(self.degrees)))
        self.lapl_adj = sqinv_degree @ self.adj @ sqinv_degree

        v_col = np.zeros((n, max_K))
        k = self.choose_model_order_K(self.lapl_adj, self.degrees, max_K, opt=opt_K)
        _, v_col[:, :k] = eigs(self.lapl_adj, k=k, which="LM")

        self.embeddings = v_col
        self.model_order_K = k

        return self.embeddings

    def predict(self):
        """
        Predicts community memberships of vertices.

        Parameters
        ----------

        Returns
        -------
        z_pred : np.ndarray of shape (n)
            Predicted community membership labels of each vertex.

        """
        kmeans = KMeans(n_clusters=self.model_order_K)
        z_pred = kmeans.fit_predict(self.embeddings[:, : self.model_order_K])
        return z_pred

    def fit_predict(self, adj, max_K=None, opt_K="empirical"):
        """
        Predicts community memberships of vertices given the adjacency matrix
        of the network.

        Parameters
        ----------
        adj : `np.ndarray` of shape (n, n)
            Adjacency matrices of size (n, n), n is the number of vertices.

        max_K : `int`, default=n/10
            Determines the maximum number of communities to predict.

        opt_K : `string`, default='empirical'
            Chooses the technique to estimate k, i.e., number of communities.

        Returns
        -------
        z_pred : np.ndarray of shape (n)
            Predicted community membership labels of each vertex.

        """
        self.fit(adj, max_K=max_K, opt_K=opt_K)
        return self.predict()


class PisCES(CommunityDetectionMixin):
    def __init__(self, verbose=False):
        super().__init__("PisCES", verbose=verbose)
        self.convergence_monitor = []

    def fit(
        self,
        adj,
        alpha=None,
        n_iter=30,
        max_K=None,
        opt_K="empirical",
        monitor_convergence=False,
    ):
        """
        Computes the spectral embeddings from given time series of matrices of
        the dynamic network.

        Parameters
        ----------
        adj : `np.ndarray` of shape (th, n, n)
            Time series of adjacency matrices of size (th,n,n), where n is the
            number of vertices, and th is the time horizon.

        alpha : `np.ndarray` of shape (th, 2), default=0.05J(th,2)
            Tuning parameter for smoothing along the time axis.

        n_iter : `int`, default=30
            Determines the number of iterations to run PisCES.

        max_K : `int`, default=n/10
            Determines the maximum number of communities to predict.

        opt_K : `string`, default='empirical'
            Chooses the technique to estimate k, i.e., number of communities.

        monitor_convergence : `bool`, default='False'
            Controls if method saves ||U_{t} - U_{t-1}|| values and |obj_t -
            obj_{t-1}> at each iteration to monitor convergence.

        Returns
        -------
        embeddings : `np.ndarray` of shape (th, n, max_K)
            Computed and smoothed spectral embeddings of the time series of the
            adjacency matrices, with shape (th, n, max_K).

        """
        self.adj = adj.astype(float)
        self.num_vertices = self.adj.shape[1]
        self.time_horizon = self.adj.shape[0]

        self.degrees = np.empty(self.adj.shape[:-1])
        self.lapl_adj = np.empty_like(self.adj)

        if alpha is None:
            alpha = 0.05 * np.ones((self.time_horizon, 2))
        if max_K is None:
            max_K = np.ceil(self.num_vertices / 10).astype(int)

        if self.time_horizon < 2:
            raise ValueError(
                "Time horizon must be at least 2, otherwise use static spectral clustering."
            )
        assert type(self.adj) in [np.ndarray, np.memmap] and self.adj.ndim == 3
        assert self.adj.shape[1] == self.adj.shape[2]
        assert isinstance(alpha, np.ndarray) and alpha.shape == (self.time_horizon, 2)
        assert max_K > 0

        th = self.time_horizon
        n = self.num_vertices

        u = np.zeros((th, n, n))
        v_col = np.zeros((th, n, max_K))
        k = np.zeros(th).astype(int) + max_K
        objective = np.zeros(n_iter)
        self.convergence_monitor = []
        diffU = 0

        for t in range(th):
            adj_t = self.adj[t, :, :]
            self.degrees[t, :] = np.sum(np.abs(adj_t), axis=0) + _eps
            sqinv_degree = sqrtm(inv(np.diag(self.degrees[t, :])))
            self.lapl_adj[t, :, :] = sqinv_degree @ adj_t @ sqinv_degree

        # Initialization of k, v_col.
        for t in range(th):
            lapl_adj_t = self.lapl_adj[t, :, :]
            k[t] = self.choose_model_order_K(
                lapl_adj_t, self.degrees[t, :], max_K, opt=opt_K
            )
            _, v_col[t, :, : k[t]] = eigs(lapl_adj_t, k=k[t], which="LM")
            u[t, :, :] = v_col[t, :, : k[t]] @ v_col[t, :, : k[t]].T

            if monitor_convergence:
                diffU = diffU + (
                    Distances.hamming_distance(
                        self.lapl_adj[t, :, :],
                        v_col[t, :, : k[t]] @ v_col[t, :, : k[t]].T,
                    )
                )

        if monitor_convergence:
            self.convergence_monitor.append((-np.inf, diffU))

        total_itr = 0
        for itr in range(n_iter):
            if self.verbose:
                print(f"Iteration {itr}/{n_iter} is running.")
            total_itr += 1
            diffU = 0
            v_col_pv = deepcopy(v_col)
            for t in range(th):
                # reprs = u[t, :, :]
                reprs = self.lapl_adj[t, :, :]
                if t == 0:
                    v_col_pv_ktn = v_col_pv[t + 1, :, : k[t + 1]]
                    reprs_bar = reprs + alpha[t, 1] * (v_col_pv_ktn @ v_col_pv_ktn.T)
                elif t == th - 1:
                    v_col_pv_ktp = v_col_pv[t - 1, :, : k[t - 1]]
                    reprs_bar = reprs + alpha[t, 0] * (v_col_pv_ktp @ v_col_pv_ktp.T)
                else:
                    v_col_pv_ktp = v_col_pv[t - 1, :, : k[t - 1]]
                    v_col_pv_ktn = v_col_pv[t + 1, :, : k[t + 1]]
                    reprs_bar = (
                        reprs
                        + (alpha[t, 0] * (v_col_pv_ktp @ v_col_pv_ktp.T))
                        + (alpha[t, 1] * (v_col_pv_ktn @ v_col_pv_ktn.T))
                    )

                k[t] = self.choose_model_order_K(
                    reprs_bar, self.degrees[t, :], max_K, opt=opt_K
                )
                _, v_col[t, :, : k[t]] = eigs(reprs_bar, k=k[t], which="LM")

                eig_val = eigvals(v_col[t, :, : k[t]].T @ v_col_pv[t, :, : k[t]])
                objective[itr] = objective[itr] + np.sum(np.abs(eig_val), axis=0)

                if monitor_convergence:
                    diffU = diffU + (
                        Distances.hamming_distance(
                            v_col[t, :, : k[t]] @ v_col[t, :, : k[t]].T,
                            v_col_pv[t, :, : k[t]] @ v_col_pv[t, :, : k[t]].T,
                        )
                    )

            if monitor_convergence:
                self.convergence_monitor.append((objective[itr], diffU))

            if itr >= 1:
                diff_obj = objective[itr] - objective[itr - 1]
                if abs(diff_obj) < CONVERGENCE_CRITERIA:
                    break

        if (
            (total_itr > 1)
            and (total_itr == n_iter)
            and (objective[-1] - objective[-2] >= CONVERGENCE_CRITERIA)
        ):
            warnings.warn("PisCES does not converge!", RuntimeWarning)

        self.embeddings = v_col
        self.model_order_K = k

        return self.embeddings

    def predict(self):
        """
        Predicts community memberships of vertices at each time point.

        Parameters
        ----------

        Returns
        -------
        z_pred : np.ndarray of shape (th, n)
            Predicted community membership labels of vertices at each time
            point, where n is the number of vertices and th is the time
            horizon.

        """
        th = self.time_horizon
        n = self.num_vertices
        z_pred = np.empty((th, n), dtype=int)
        for t in range(th):
            kmeans = KMeans(n_clusters=self.model_order_K[t])
            z_pred[t, :] = kmeans.fit_predict(
                self.embeddings[t, :, : self.model_order_K[t]]
            )
        return z_pred

    def fit_predict(
        self,
        adj,
        alpha=None,
        n_iter=30,
        max_K=None,
        opt_K="empirical",
        monitor_convergence=False,
    ):
        """
        Predicts time series of community memberships given the adjacency
        matrices of the dynamic network.

        Parameters
        ----------
        adj : `np.ndarray` of shape (th, n, n)
            Time series of adjacency matrices of size (th,n,n), where n
            is the number of vertices, and th is the time horizon.

        alpha : `np.ndarray` of shape (th, 2), default=0.05J(th,2)
            Tuning parameter for smoothing along the time axis.

        n_iter : `int`, default=30
            Determines the number of iterations to run PisCES.

        max_K : `int`, default=n/10
            Determines the maximum number of communities to predict.

        opt_K : `string`, default='empirical'
            Chooses the technique to estimate k, i.e., number of communities.

        monitor_convergence : `bool`, default='False'
            Controls if method saves ||U_{t} - U_{t-1}|| values and |obj_t -
            obj_{t-1}> at each iteration to monitor convergence.

        Returns
        -------
        z_pred : np.ndarray of shape (th, n)
            Predicted community membership labels of vertices at each time
            point, where n is the number of vertices and th is the time
            horizon.

        """
        self.fit(
            adj,
            alpha=alpha,
            max_K=max_K,
            opt_K=opt_K,
            n_iter=n_iter,
            monitor_convergence=monitor_convergence,
        )
        return self.predict()

    @classmethod
    def cross_validation(
        cls,
        adj,
        num_folds=5,
        alpha=None,
        n_iter=30,
        max_K=None,
        opt_K="empirical",
        n_jobs=1,
    ):
        """
        Performs cross validation to choose the best value for the alpha parameter.

        Parameters
        ----------
        adj : `np.ndarray` of shape (th, n, n)
            Time series of adjacency matrices of size (th,n,n), where n is the
            number of vertices, and th is the time horizon.

        num_folds : `int`, default=5
            Number of folds to perform in the cross validation.

        alpha : `np.ndarray` of shape (th, 2), default=0.05J(th,2)
            Tuning parameter for smoothing along the time axis.

        n_iter : `int`, default=30
            Determines the number of iterations to run PisCES.

        max_K : `int`, default=n/10
            Determines the maximum number of communities to predict.

        opt_K : `string`, default='empirical'
            Chooses the technique to estimate k, i.e., number of communities.

        n_jobs : `int`, default=1
            The number of parallel `joblib` threads.

        Returns
        -------
        modu : `float`
            Sum of the modularity value computed for each fold with respect to
            the given alpha.

        logllh : `float`
            Sum of the log-likelihood value computed for each fold with
            respect to the given alpha.

        """
        adj = adj.astype(float)
        num_vertices = adj.shape[1]
        time_horizon = adj.shape[0]

        if alpha is None:
            alpha = 0.05 * np.ones((time_horizon, 2))
        if max_K is None:
            max_K = np.ceil(num_vertices / 10).astype(int)

        if time_horizon < 2:
            raise ValueError(
                "Time horizon must be at least 2, otherwise use static spectral clustering."
            )
        assert type(adj) in [np.ndarray, np.memmap] and adj.ndim == 3
        assert adj.shape[1] == adj.shape[2]
        assert alpha.shape == (time_horizon, 2)
        assert max_K > 0

        n = num_vertices
        th = time_horizon

        idx_n = np.arange(n)
        idx = np.c_[np.repeat(idx_n, idx_n.shape), np.tile(idx_n, idx_n.shape)]
        r = np.random.choice(n**2, size=n**2, replace=False)

        pisces_kwargs = {
            "alpha": alpha,
            "n_iter": n_iter,
            "max_K": max_K,
            "opt_K": opt_K,
        }

        def compute_for_fold(adj, idx_split, n, th, pisces_kwargs={}):
            cv_idx = np.zeros((th, n, n), dtype=bool)
            adj_train = np.zeros((th, n, n))
            adj_train_imputed = np.zeros((th, n, n))

            for t in range(th):
                idx1, idx2 = idx_split[t, :, 0], idx_split[t, :, 1]

                cv_idx_t = np.zeros((n, n), dtype=bool)
                cv_idx_t[idx1, idx2] = True
                cv_idx_t = np.triu(cv_idx_t) + np.triu(cv_idx_t).T
                cv_idx[t, :, :] = cv_idx_t

                adj_train[t, :, :] = adj[t, :, :]
                adj_train[t, idx1, idx2] = 0
                adj_train[t, :, :] = (
                    np.triu(adj_train[t, :, :]) + np.triu(adj_train[t, :, :]).T
                )
                adj_train_imputed[t, :, :] = cls.eigen_complete(
                    adj_train[t, :, :], cv_idx_t, 10, 10
                )

            z_pred = cls().fit_predict(
                deepcopy(adj_train_imputed[:, :, :]),
                **pisces_kwargs,
            )

            modu_fold, logllh_fold = 0, 0
            for t in range(th):
                modu_fold = modu_fold + cls.modularity(
                    adj[t, :, :],
                    adj_train[t, :, :],
                    z_pred[t, :],
                    cv_idx[t, :, :],
                )
                logllh_fold = logllh_fold + cls.loglikelihood(
                    adj[t, :, :],
                    adj_train[t, :, :],
                    z_pred[t, :],
                    cv_idx[t, :, :],
                )

            return modu_fold, logllh_fold

        modu_total = 0
        logllh_total = 0

        def split_train_test(fold_idx):
            pfold = n**2 // num_folds
            start, end = pfold * fold_idx, (fold_idx + 1) * pfold
            test = r[start:end]
            idx_split = idx[test, :]
            return np.tile(idx_split, (th, 1, 1))

        if n_jobs > 1:
            from joblib import Parallel, delayed

            with Parallel(n_jobs=n_jobs) as parallel:  # prefer="processes"
                loss_zipped = parallel(
                    delayed(compute_for_fold)(
                        adj,
                        split_train_test(fold_idx),
                        n,
                        th,
                        pisces_kwargs=pisces_kwargs,
                    )
                    for fold_idx in range(num_folds)
                )
                modu_fold, logllh_fold = map(np.array, zip(*loss_zipped))
                modu_total = sum(modu_fold)
                logllh_total = sum(logllh_fold)
        else:
            for fold_idx in range(num_folds):
                modu_fold, logllh_fold = compute_for_fold(
                    adj, split_train_test(fold_idx), n, th, pisces_kwargs=pisces_kwargs
                )
                modu_total = modu_total + modu_fold
                logllh_total = logllh_total + logllh_fold

        num_adj = th

        return modu_total / num_adj, logllh_total / num_adj


class MuDCoD(CommunityDetectionMixin):
    def __init__(self, verbose=False):
        super().__init__("MuDCoD", verbose=verbose)
        self.convergence_monitor = []

    def fit(
        self,
        adj,
        alpha=None,
        beta=None,
        n_iter=30,
        max_K=None,
        opt_K="empirical",
        monitor_convergence=False,
    ):
        """
        Computes the spectral embeddings from given multi-subject time series
        of matrices of the dynamic networks of different subjects.

        Parameters
        ----------
        adj : `np.ndarray` of shape (ns, th, n, n)
            Multi-subject time series of adjacency matrices of size (ns,
            th,n,n), where n is the number of vertices, th is the time horizon,
            and ns is the number of subjects.

        alpha : `np.ndarray` of shape (th, 2), default=0.05J(th,2)
            Tuning parameter for smoothing along the time axis.

        beta : `np.ndarray` of shape (ns), default=0.01J(ns)
            Tuning parameter for smoothing along the subject axis.

        n_iter : `int`, default=30
            Determines the number of iterations to run MuDCoD.

        max_K : `int`, default=n/10
            Determines the maximum number of communities to predict.

        opt_K : `string`, default='empirical'
            Chooses the technique to estimate k, i.e., number of communities.

        monitor_convergence : `bool`, default='False'
            Controls if method saves ||U_{t} - U_{t-1}|| values and |obj_t -
            obj_{t-1}> at each iteration to monitor convergence.

        Returns
        -------
        embeddings : `np.ndarray` of shape (ns, th, n, max_K)
            Computed and smoothed spectral embeddings of the multi-subject time
            series of the adjacency matrices, with shape (ns, th, n, max_K).

        """
        self.adj = adj.astype(float)
        self.num_subjects = self.adj.shape[0]
        self.time_horizon = self.adj.shape[1]
        self.num_vertices = self.adj.shape[2]

        self.degrees = np.empty(self.adj.shape[:-1])
        self.lapl_adj = np.empty_like(self.adj)

        if alpha is None:
            alpha = 0.05 * np.ones((self.time_horizon, 2))
        if beta is None:
            beta = 0.01 * np.ones(self.num_subjects)
        if max_K is None:
            max_K = np.ceil(self.num_vertices / 10).astype(int)

        if self.time_horizon < 2:
            raise ValueError(
                "Time horizon must be at least 2, otherwise use static spectral clustering."
            )
        assert type(self.adj) in [np.ndarray, np.memmap] and self.adj.ndim == 4
        assert self.adj.shape[2] == self.adj.shape[3]
        assert isinstance(alpha, np.ndarray) and alpha.shape == (self.time_horizon, 2)
        assert isinstance(beta, np.ndarray) and beta.shape == (self.num_subjects,)
        assert max_K > 0

        ns = self.num_subjects
        th = self.time_horizon
        n = self.num_vertices

        k = np.zeros((ns, th)).astype(int) + max_K
        v_col = np.zeros((ns, th, n, max_K))
        u = np.zeros((ns, th, n, n))
        objective = np.zeros((n_iter))
        self.convergence_monitor = []
        diffU = 0

        for t in range(th):
            for sbj in range(ns):
                adj_t = self.adj[sbj, t, :, :]
                self.degrees[sbj, t, :] = np.sum(np.abs(adj_t), axis=0) + _eps
                sqinv_degree = sqrtm(inv(np.diag(self.degrees[sbj, t, :])))
                self.lapl_adj[sbj, t, :, :] = sqinv_degree @ adj_t @ sqinv_degree

        # Initialization of k, v_col.
        for t in range(th):
            for sbj in range(ns):
                lapl_adj_t = self.lapl_adj[sbj, t, :, :]
                k[sbj, t] = self.choose_model_order_K(
                    lapl_adj_t, self.degrees[sbj, t, :], max_K, opt=opt_K
                )
                _, v_col[sbj, t, :, : k[sbj, t]] = eigs(
                    lapl_adj_t, k=k[sbj, t], which="LM"
                )
                u[sbj, t, :, :] = v_col[sbj, t, :, :] @ v_col[sbj, t, :, :].T

                if monitor_convergence:
                    diffU = diffU + (
                        Distances.hamming_distance(
                            self.lapl_adj[sbj, t, :, :],
                            v_col[sbj, t, :, : k[sbj, t]]
                            @ v_col[sbj, t, :, : k[sbj, t]].T,
                        )
                    )

        if monitor_convergence:
            self.convergence_monitor.append((-np.inf, diffU))

        total_itr = 0
        for itr in range(n_iter):
            if self.verbose:
                print(f"Iteration {itr}/{n_iter} is running.")
            total_itr += 1
            diffU = 0
            v_col_pv = deepcopy(v_col)
            for t in range(th):
                v_col_t = v_col_pv[:, t, :, :]
                swp_v_col_t = np.swapaxes(v_col_t, 1, 2)
                u_bar_t = v_col_t @ swp_v_col_t
                for sbj in range(ns):
                    # reprs = u[sbj, t, :, :]
                    mu_u_bar_t = np.mean(np.delete(u_bar_t, sbj, 0), axis=0)
                    reprs = self.lapl_adj[sbj, t, :, :]
                    if t == 0:
                        v_col_pv_ktn = v_col_pv[sbj, t + 1, :, : k[sbj, t + 1]]
                        reprs_bar = (
                            reprs
                            + alpha[t, 1] * (v_col_pv_ktn @ v_col_pv_ktn.T)
                            + beta[sbj] * mu_u_bar_t
                        )
                    elif t == th - 1:
                        v_col_pv_ktp = v_col_pv[sbj, t - 1, :, : k[sbj, t - 1]]
                        reprs_bar = (
                            reprs
                            + alpha[t, 0] * (v_col_pv_ktp @ v_col_pv_ktp.T)
                            + beta[sbj] * mu_u_bar_t
                        )
                    else:
                        v_col_pv_ktp = v_col_pv[sbj, t - 1, :, : k[sbj, t - 1]]
                        v_col_pv_ktn = v_col_pv[sbj, t + 1, :, : k[sbj, t + 1]]
                        reprs_bar = (
                            reprs
                            + (alpha[t, 0] * (v_col_pv_ktp @ v_col_pv_ktp.T))
                            + (alpha[t, 1] * (v_col_pv_ktn @ v_col_pv_ktn.T))
                            + beta[sbj] * mu_u_bar_t
                        )

                    k[sbj, t] = self.choose_model_order_K(
                        reprs_bar,
                        self.degrees[sbj, t, :],
                        max_K,
                        opt=opt_K,
                    )
                    _, v_col[sbj, t, :, : k[sbj, t]] = eigs(
                        reprs_bar, k=k[sbj, t], which="LM"
                    )

                    eig_val = eigvals(
                        v_col[sbj, t, :, : k[sbj, t]].T
                        @ v_col_pv[sbj, t, :, : k[sbj, t]]
                    )
                    objective[itr] = objective[itr] + np.sum(np.abs(eig_val), axis=0)

                    if monitor_convergence:
                        diffU = diffU + (
                            Distances.hamming_distance(
                                v_col[sbj, t, :, : k[sbj, t]]
                                @ v_col[sbj, t, :, : k[sbj, t]].T,
                                v_col_pv[sbj, t, :, : k[sbj, t]]
                                @ v_col_pv[sbj, t, :, : k[sbj, t]].T,
                            )
                        )

            if monitor_convergence:
                self.convergence_monitor.append((objective[itr], diffU))

            if itr >= 1:
                diff_obj = objective[itr] - objective[itr - 1]
                if abs(diff_obj) < CONVERGENCE_CRITERIA:
                    break

        if (
            (total_itr > 1)
            and (total_itr == n_iter)
            and (objective[-1] - objective[-2] >= CONVERGENCE_CRITERIA)
        ):
            warnings.warn("MuDCoD does not converge!", RuntimeWarning)

        self.embeddings = v_col
        self.model_order_K = k

        return self.embeddings

    def predict(
        self,
    ):
        """
        Predicts community memberships of vertices at each time point for each
        subject.

        Parameters
        ----------

        Returns
        -------
        z_pred : np.ndarray of shape (ns, th, n)
            Predicted community membership labels of vertices at each time
            point for each subject, where n is the number of vertices, th is
            the time horizon, and ns is the number of subjects.

        """
        ns = self.num_subjects
        th = self.time_horizon
        n = self.num_vertices
        z_pred = np.empty((ns, th, n), dtype=int)
        for t in range(th):
            for sbj in range(ns):
                kmeans = KMeans(n_clusters=self.model_order_K[sbj, t])
                z_pred[sbj, t, :] = kmeans.fit_predict(
                    self.embeddings[sbj, t, :, : self.model_order_K[sbj, t]]
                )
        return z_pred

    def fit_predict(
        self,
        adj,
        alpha=None,
        beta=None,
        n_iter=30,
        max_K=None,
        opt_K="empirical",
        monitor_convergence=False,
    ):
        """
        Predicts multi-subject time series of community memberships given the
        adjacency matrices of the dynamic networks for each subject.

        Parameters
        ----------
        adj : `np.ndarray` of shape (th, n, n)
            Time series of adjacency matrices of size (th,n,n), where n is the
            number of vertices, and th is the time horizon.

        alpha : `np.ndarray` of shape (th, 2), default=0.05J(th,2)
            Tuning parameter for smoothing along the time axis.

        n_iter : `int`, default=30
            Determines the number of iterations to run PisCES.

        max_K : `int`, default=n/10
            Determines the maximum number of communities to predict.

        opt_K : `string`, default='empirical'
            Chooses the technique to estimate k, i.e., number of communities.

        monitor_convergence : `bool`, default='False'
            Controls if method saves ||U_{t} - U_{t-1}|| values and |obj_t -
            obj_{t-1}> at each iteration to monitor convergence.

        Returns
        -------
        z_pred : np.ndarray of shape (ns, th, n)
            Predicted community membership labels of vertices at each time
            point for each subject, where n is the number of vertices, th is
            the time horizon, and ns is the number of subjects.

        """
        self.fit(
            adj,
            alpha=alpha,
            beta=beta,
            max_K=max_K,
            opt_K=opt_K,
            n_iter=n_iter,
            monitor_convergence=monitor_convergence,
        )
        return self.predict()

    @classmethod
    def cross_validation(
        cls,
        adj,
        num_folds=5,
        alpha=None,
        beta=None,
        n_iter=30,
        max_K=None,
        opt_K="empirical",
        n_jobs=1,
    ):
        """
        Performs cross validation to choose the best pair of values for the alpha
        and the beta parameters.

        Parameters
        ----------
        num_folds : `int`, default=5
            Number of folds to perform in the cross validation.

        alpha : `np.ndarray` of shape (th, 2), default=0.05J(th,2)
            Tuning parameter for smoothing along the time axis.

        beta : `np.ndarray` of shape (ns), default=0.01J(ns)
            Tuning parameter for smoothing along the subject axis.

        n_iter : `int`, default=30
            Determines the number of iterations to run PisCES.

        max_K : `int`, default=n/10
            Determines the maximum number of communities to predict.

        opt_K : `string`, default='empirical'
            Chooses the technique to estimate k, i.e., number of communities.

        n_jobs : `int`, default=1
            The number of parallel `joblib` threads.

        Returns
        -------
        modu : `float`
            Sum of the modularity value computed for each fold with respect to
            the given alpha.

        logllh : `float`
            Sum of the log-likelihood value computed for each fold with
            respect to the given alpha.

        """
        adj = adj.astype(float)
        num_vertices = adj.shape[2]
        time_horizon = adj.shape[1]
        num_subjects = adj.shape[0]

        if alpha is None:
            alpha = 0.05 * np.ones((time_horizon, 2))
        if beta is None:
            beta = 0.01 * np.ones(num_subjects)
        if max_K is None:
            max_K = np.ceil(num_vertices / 10).astype(int)

        if time_horizon < 2:
            raise ValueError(
                "Time horizon must be at least 2, otherwise use static spectral clustering."
            )
        assert type(adj) in [np.ndarray, np.memmap] and adj.ndim == 4
        assert adj.shape[2] == adj.shape[3]
        assert alpha.shape == (time_horizon, 2)
        assert beta.shape == (num_subjects,)
        assert max_K > 0

        ns = num_subjects
        th = time_horizon
        n = num_vertices

        idx_n = np.arange(n)
        idx = np.c_[np.repeat(idx_n, idx_n.shape), np.tile(idx_n, idx_n.shape)]
        r = np.random.choice(n**2, size=n**2, replace=False)

        mudcod_kwargs = {
            "alpha": alpha,
            "beta": beta,
            "n_iter": n_iter,
            "max_K": max_K,
            "opt_K": opt_K,
        }

        def compute_for_fold(adj, idx_split, n, th, ns, mudcod_kwargs={}):
            cv_idx = np.empty((ns, th, n, n), dtype=bool)
            adj_train = np.zeros((ns, th, n, n))
            adj_train_imputed = np.zeros((ns, th, n, n))

            for t in range(th):
                for sbj in range(ns):
                    idx1, idx2 = idx_split[sbj, t, :, 0], idx_split[sbj, t, :, 1]

                    cv_idx_t = np.zeros((n, n), dtype=bool)
                    cv_idx_t[idx1, idx2] = True
                    cv_idx_t = np.triu(cv_idx_t) + np.triu(cv_idx_t).T
                    cv_idx[sbj, t, :, :] = cv_idx_t

                    adj_train[sbj, t, :, :] = adj[sbj, t, :, :]
                    adj_train[sbj, t, idx1, idx2] = 0
                    adj_train[sbj, t] = (
                        np.triu(adj_train[sbj, t]) + np.triu(adj_train[sbj, t]).T
                    )
                    adj_train_imputed[sbj, t, :, :] = cls.eigen_complete(
                        adj_train[sbj, t], cv_idx_t, 10, 10
                    )

            z_pred = cls().fit_predict(
                deepcopy(adj_train_imputed[:, :, :, :]),
                **mudcod_kwargs,
            )

            modu_fold, logllh_fold = 0, 0
            for t in range(th):
                for sbj in range(ns):
                    modu_fold = modu_fold + cls.modularity(
                        adj[sbj, t, :, :],
                        adj_train[sbj, t, :, :],
                        z_pred[sbj, t, :],
                        cv_idx[sbj, t, :, :],
                    )
                    logllh_fold = logllh_fold + cls.loglikelihood(
                        adj[sbj, t, :, :],
                        adj_train[sbj, t, :, :],
                        z_pred[sbj, t, :],
                        cv_idx[sbj, t, :, :],
                    )
            return modu_fold, logllh_fold

        modu_total = 0
        logllh_total = 0

        def split_train_test(fold_idx):
            pfold = n**2 // num_folds
            start, end = pfold * fold_idx, (fold_idx + 1) * pfold
            test = r[start:end]
            idx_split = idx[test, :]
            return np.tile(idx_split, (ns, th, 1, 1))

        if n_jobs > 1:
            from joblib import Parallel, delayed

            with Parallel(n_jobs=n_jobs) as parallel:  # prefer="processes"
                loss_zipped = parallel(
                    delayed(compute_for_fold)(
                        adj,
                        split_train_test(fold_idx),
                        n,
                        th,
                        ns,
                        mudcod_kwargs=mudcod_kwargs,
                    )
                    for fold_idx in range(num_folds)
                )
                modu_fold, logllh_fold = map(np.array, zip(*loss_zipped))
                modu_total = sum(modu_fold)
                logllh_total = sum(logllh_fold)
        else:
            for fold_idx in range(num_folds):
                modu_fold, logllh_fold = compute_for_fold(
                    adj,
                    split_train_test(fold_idx),
                    n,
                    th,
                    ns,
                    mudcod_kwargs=mudcod_kwargs,
                )
                modu_total = modu_total + modu_fold
                logllh_total = logllh_total + logllh_fold

        num_adj = ns * th

        return modu_total / num_adj, logllh_total / num_adj
