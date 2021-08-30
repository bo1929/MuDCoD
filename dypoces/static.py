import warnings
import numpy as np

from copy import deepcopy
from numpy.linalg import inv
from scipy.sparse.linalg import eigs
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans

from dypoces.spectral import SpectralClustering
from dypoces.utils.sutils import log


warnings.filterwarnings(action="ignore", category=np.ComplexWarning)

_eps = 10 ** (-10)


class Static(SpectralClustering):
    def __init__(self, verbose=False):
        super().__init__("static", verbose=verbose)

    def fit(self, adj, degree_correction=True):
        """
        Parameters
        ----------
        adj
            adjacency matrices with dimention (n,n),
            n is the number of nodes.
        degree_correction
            degree normalization, default is 'True'.
        """
        assert type(adj) in [np.ndarray, np.memmap] and adj.ndim == 2

        self.adj = adj.astype(float)
        self.adj_shape = self.adj.shape
        self.n = self.adj_shape[0]  # or adj_shape[1]
        self.degree = deepcopy(adj)
        self.degree_correction = degree_correction

        if self.verbose:
            log(f"Static-fit ~ #nodes:{self.n}")

        if self.degree_correction:
            dg = np.diag(np.sum(np.abs(adj), axis=0) + _eps)
            sqinv_degree = sqrtm(inv(dg))
            self.adj = sqinv_degree @ adj @ sqinv_degree
            self.degree = dg
        else:
            self.degree = np.eye(self.n)

    def predict(self, k_max=None, n_iter=30):
        """
        Parameters
        ----------
        k_max
            maximum number of communities, default is n/10.

        Returns
        -------
        z_series: community prediction for each time point, with shape (n).
        """
        n = self.n
        adj = self.adj
        degree = self.degree

        if k_max is None:
            k_max = n // 10
            log(f"k_max is not provided, default value is floor({n}/10).")

        if self.verbose:
            log(f"Static-predict ~ k_max:{k_max}")

        v_col = np.zeros((n, k_max))

        # initialization of k, v_col.
        k = self.choose_k(adj, adj, degree, k_max)
        _, v_col[:, :k] = eigs(adj, k=k, which="LM")

        kmeans = KMeans(n_clusters=k)
        z = kmeans.fit_predict(v_col[:, :k])

        return z

    def fit_predict(self, adj, k_max=None, n_iter=30):
        self.fit(adj, degree_correction=True)
        return self.predict(k_max=k_max, n_iter=n_iter)
