import numpy as np
import networkx as nx

from numpy import linalg as LA
from scipy.spatial import distance
from scipy.linalg import eigh
from scipy.sparse import csgraph
from scipy.integrate import quad
from sklearn.metrics.cluster import adjusted_rand_score

_eps = 10 ** (-5)


def inform_about_network(adj):
    G = nx.from_numpy_array(adj)
    info_txt = nx.classes.function.info(G)
    return info_txt


def degree_details(adj):
    G = nx.from_numpy_array(adj)
    hist_degree = nx.classes.function.degree_histogram(G)
    density = nx.classes.function.density(G)
    return hist_degree, density


def _assert_before_distance(adj1, adj2):
    assert adj1.ndim == 2 and adj2.ndim == 2
    assert adj1.shape == adj2.shape
    assert adj1.shape[0] == adj1.shape[1]
    assert adj2.shape[0] == adj2.shape[1]


class Similarity:
    @staticmethod
    def community_membership_distance(z1, z2):
        return 1 - adjusted_rand_score(z1, z2)

    @staticmethod
    def exact_edit_distance(adj1, adj2):
        G1 = nx.from_numpy_array(adj1.astype(int))
        G2 = nx.from_numpy_array(adj2.astype(int))
        dist = nx.graph_edit_distance(G1, G2)
        return dist

    @staticmethod
    def optimized_edit_distance(adj1, adj2):
        G1 = nx.from_numpy_array(adj1.astype(int))
        G2 = nx.from_numpy_array(adj2.astype(int))
        minv = np.inf
        for v in nx.optimize_graph_edit_distance(G1, G2):
            minv = v
            print(minv)
        return minv

    @staticmethod
    def hamming_distance(adj1, adj2, normalize=True):
        _assert_before_distance(adj1, adj2)
        n = adj1.shape[0]
        dist = np.sum(np.abs(adj1 - adj2))
        if normalize:
            dist = dist / (n * (n - 1))
        else:
            pass
        return dist

    @staticmethod
    def jaccard_distance(adj1, adj2):
        _assert_before_distance(adj1, adj2)
        ## dist = (np.sum(np.abs(adj1 - adj2))) / np.sum(np.maximum(adj1, adj2))
        dist = 1 - np.sum(np.minimum(adj1, adj2)) / np.sum(np.maximum(adj1, adj2))
        return dist

    @staticmethod
    def frobenius_distance(adj1, adj2):
        _assert_before_distance(adj1, adj2)
        dist = LA.norm(adj1 - adj2, ord="fro")
        return dist

    @staticmethod
    def communicability_jensenshannon(adj1, adj2):
        """

        author: Brennan Klein
        email: brennanjamesklein@gmail.com
        Submitted as part of the 2019 NetSI Collabathon.

        """
        G1 = nx.from_numpy_array(adj1.astype(int))
        G2 = nx.from_numpy_array(adj2.astype(int))

        N1 = G1.number_of_nodes()
        N2 = G2.number_of_nodes()

        C1 = nx.communicability_exp(G1)
        C2 = nx.communicability_exp(G2)

        Ca1 = np.zeros((N1, N1))
        Ca2 = np.zeros((N2, N2))

        for i in range(Ca1.shape[0]):
            Ca1[i] = np.array(list(C1[i].values()))
        for i in range(Ca2.shape[0]):
            Ca2[i] = np.array(list(C2[i].values()))

        lil_sigma1 = np.triu(Ca1).flatten()
        lil_sigma2 = np.triu(Ca2).flatten()

        big_sigma1 = sum(lil_sigma1[np.nonzero(lil_sigma1)[0]])
        big_sigma2 = sum(lil_sigma2[np.nonzero(lil_sigma2)[0]])

        P1 = lil_sigma1 / big_sigma1
        P2 = lil_sigma2 / big_sigma2
        P1 = np.array(sorted(P1))
        P2 = np.array(sorted(P2))

        dist = distance.jensenshannon(P1, P2)

        return dist

    @staticmethod
    def ipsen_mikhailov_distance(adj1, adj2, hwhm=0.08):
        """

        author: Guillaume St-Onge
        email: guillaume.st-onge.4@ulaval.ca
        Submitted as part of the 2019 NetSI Collabathon.

        Compare the spectrum ot the associated Laplacian matrices.

        The results dictionary also stores a 2-tuple of the underlying
        adjacency matrices in the key `'adjacency_matrices'`.


        Parameters
        ----------

        adj1, adj2
            two adjacency matrices to be compared.

        hwhm (float)
            half with at half maximum of the lorentzian kernel.

        Returns
        -------

        dist (float)
            the distance between G1 and G2.

        """
        _assert_before_distance(adj1, adj2)
        N = adj1.shape[0]
        # get laplacian matrix
        L1 = csgraph.laplacian(adj1, normed=False)
        L2 = csgraph.laplacian(adj2, normed=False)

        # get the modes for the positive-semidefinite laplacian
        w1 = np.sqrt(np.abs(eigh(L1)[0][1:]))
        w2 = np.sqrt(np.abs(eigh(L2)[0][1:]))

        # we calculate the norm for both spectrum
        norm1 = (N - 1) * np.pi / 2 - np.sum(np.arctan(-w1 / hwhm))
        norm2 = (N - 1) * np.pi / 2 - np.sum(np.arctan(-w2 / hwhm))

        # define both spectral densities
        density1 = (
            lambda w: np.sum(hwhm / ((w - w1) ** 2 + hwhm ** 2)) / norm1
        )  # noqa: E731
        density2 = (
            lambda w: np.sum(hwhm / ((w - w2) ** 2 + hwhm ** 2)) / norm2
        )  # noqa: E731

        func = lambda w: (density1(w) - density2(w)) ** 2  # noqa: E731

        return np.sqrt(quad(func, 0, np.inf, limit=100)[0])

    @staticmethod
    def polynomial_dissimilarity(adj1, adj2, k=5, alpha=1):
        """

        author: Jessica T. Davis
        email:
            Submitted as part of the 2019 NetSI Collabathon.

        Compares the polynomials of the eigenvalue decomposition of
        two adjacency matrices.

        Note that the :math:`ij`-th element of where :math:`adj^k`
        corresponds to the number of paths of length :math:`k` between
        nodes :math:`i` and :math:`j`.

        The results dictionary also stores a 2-tuple of the underlying
        adjacency matrices in the key `'adjacency_matrices'`.

        Parameters
        ----------

        adj1, adj2
            two adjacency matrices to be compared.

        k (float)
            maximum degree of the polynomial

        alpha (float)
            weighting factor

        Returns
        -------
        dist (float)
            Polynomial Dissimilarity between `G1`, `G2`

        """

        def similarity_score(adj, k, alpha):
            """
            Calculate the similarity score used in the polynomial dissimilarity
            distance. This uses a polynomial transformation of the eigenvalues of the
            of the adjacency matrix in combination with the eigenvectors of the
            adjacency matrix. See p. 27 of Donnat and Holmes (2018).
            """

            eig_vals, Q = np.linalg.eig(adj)

            n = adj.shape[0]

            def polynomial(kp):
                return eig_vals ** kp / (n - 1) ** (alpha * (kp - 1))

            W = np.diag(sum([polynomial(k) for k in range(1, k + 1)]))
            P_A = np.dot(np.dot(Q, W), Q.T)

            return P_A

        P_adj1 = similarity_score(adj1, k, alpha)
        P_adj2 = similarity_score(adj2, k, alpha)

        dist = np.linalg.norm(P_adj1 - P_adj2, ord="fro") / adj1.shape[0] ** 2

        return dist


class Loss:
    @staticmethod
    def modularity(adj_test, adj_train, z_hat, cvidx):
        """
        Calculate modularity

        Parameters
        ----------
        adj_test
                test matrix with dimention (n,n); training edges with value 0
        adj_train
                training matrix with dimention (n,n); test edges with value 0
        z_hat
                estimated community assignment with dimension (1,n)
        cvidx
                (n,n) marix indicates the index of test edges: 1 for test and 0 for
                training

        Returns
        -------
        modularity
                modularity on test data
        """
        modularity = 0

        kts = np.sum(adj_train, axis=0)
        w = np.sum(kts, axis=0)
        n = z_hat.shape[0]

        row_idx, col_idx = np.nonzero(cvidx == 0)
        hval, _ = np.histogram(col_idx, bins=n)
        ne = np.sum(hval, axis=0)

        for i in range(n):
            for j in range(n):
                if (cvidx[i, j] > 0 and z_hat[i] == z_hat[j]) and i != j:
                    modularity = modularity + (
                        adj_test[i, j] - (kts[i] / hval[i] * kts[j] / hval[j] / w * ne)
                    )

        return modularity / w

    @staticmethod
    def loglikelihood(adj_test, adj_train, z_hat, cvidx):
        """
        Calculate loglikelihood

        Parameters
        ----------
        adj_test
                test matrix with dimention (n,n); training edges with value 0
        adj_train
                training matrix with dimention (n,n); test edges with value 0
        z_hat
                estimated community assignment with dimension (1,n)
        cvidx
                (n,n) marix indicates the index of test edges: 1 for test and 0 for
                training

        Returns
        -------
        loglikelihood
                loglikelihood on test data
        """
        logllh = 0
        k_ = np.max(z_hat) + 1
        n = len(z_hat)

        d_out_train = np.sum(adj_train[:, :], axis=0)
        ## d_out_test = np.sum(adj_test[:, :], axis=0)
        ## td_test = np.sum(d_out_test, axis=0)
        ## d_in_train = np.sum(adj_train[:, :], axis=1)

        hatB = np.zeros((k_, k_), dtype=int)

        for kk in range(k_):
            for ll in range(k_):
                td = np.sum(adj_train[np.ix_(z_hat == kk, z_hat == ll)])
                qd = np.sum(
                    (d_out_train[:, np.newaxis] @ d_out_train[np.newaxis, :])[
                        np.ix_(z_hat == kk, z_hat == ll)
                    ]
                )
                hatB[kk, ll] = td / qd

        for i in range(n):
            for j in range(n):
                if cvidx[i, j] > 0:
                    prob = d_out_train[i] * d_out_train[j] * hatB[z_hat[i], z_hat[j]]

                    if prob == 0 or np.isnan(prob):
                        prob = _eps
                    elif prob >= 1:
                        prob = 1 - _eps
                    else:
                        pass

                    logllh = (
                        logllh
                        + np.log(prob) * (adj_test[i, j])
                        + np.log(1 - prob) * (1 - adj_test[i, j])
                    )

        return logllh
