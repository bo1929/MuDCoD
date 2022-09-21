import numpy as np
import networkx as nx

from numpy import linalg as LA
from scipy.spatial import distance
from scipy.linalg import eigh
from scipy.sparse import csgraph
from scipy.integrate import quad
from sklearn.metrics.cluster import adjusted_rand_score

_eps = 10 ** (-5)


def _assert_before_distance(adj1, adj2):
    assert adj1.ndim == 2 and adj2.ndim == 2
    assert adj1.shape == adj2.shape
    assert adj1.shape[0] == adj1.shape[1]
    assert adj2.shape[0] == adj2.shape[1]


class Distances:
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
        return minv

    @staticmethod
    def hamming_distance(adj1, adj2, normalize=False):
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
        Submitted as part of the 2019 NetSI Collabathon.
        """
        _assert_before_distance(adj1, adj2)
        N = adj1.shape[0]
        # Get the Laplacian matrix.
        L1 = csgraph.laplacian(adj1, normed=False)
        L2 = csgraph.laplacian(adj2, normed=False)

        # Get the modes for the positive-semidefinite Laplacian.
        w1 = np.sqrt(np.abs(eigh(L1)[0][1:]))
        w2 = np.sqrt(np.abs(eigh(L2)[0][1:]))

        # Calculate the norm for both spectrum.
        norm1 = (N - 1) * np.pi / 2 - np.sum(np.arctan(-w1 / hwhm))
        norm2 = (N - 1) * np.pi / 2 - np.sum(np.arctan(-w2 / hwhm))

        # Define both spectral densities.
        density1 = (
            lambda w: np.sum(hwhm / ((w - w1) ** 2 + hwhm**2)) / norm1
        )  # noqa: E731
        density2 = (
            lambda w: np.sum(hwhm / ((w - w2) ** 2 + hwhm**2)) / norm2
        )  # noqa: E731

        func = lambda w: (density1(w) - density2(w)) ** 2  # noqa: E731

        return np.sqrt(quad(func, 0, np.inf, limit=100)[0])

    @staticmethod
    def polynomial_dissimilarity(adj1, adj2, k=5, alpha=1):
        """
        Author: Jessica T. Davis
        Submitted as part of the 2019 NetSI Collabathon.
        """

        def similarity_score(adj, k, alpha):
            """
            Calculate the similarity score used in the polynomial dissimilarity
            distance. This uses a polynomial transformation of the eigenvalues
            of the of the adjacency matrix in combination with the eigenvectors
            of the adjacency matrix. See p. 27 of Donnat and Holmes (2018).
            """

            eig_vals, Q = np.linalg.eig(adj)

            n = adj.shape[0]

            def polynomial(kp):
                return eig_vals**kp / (n - 1) ** (alpha * (kp - 1))

            W = np.diag(sum([polynomial(k) for k in range(1, k + 1)]))
            P_A = np.dot(np.dot(Q, W), Q.T)

            return P_A

        P_adj1 = similarity_score(adj1, k, alpha)
        P_adj2 = similarity_score(adj2, k, alpha)

        dist = np.linalg.norm(P_adj1 - P_adj2, ord="fro") / adj1.shape[0] ** 2

        return dist
