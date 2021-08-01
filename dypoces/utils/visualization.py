import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class nxDraw:
    @staticmethod
    def draw_spectral(A, path=None, **kwargs):
        G = nx.from_numpy_matrix(A)
        nx.draw_spectral(G, **kwargs)
        if path is not None:
            plt.savefig(path)

    @staticmethod
    def draw_kamada_kawai(A, path=None, **kwargs):
        G = nx.from_numpy_matrix(A)
        nx.draw_kamada_kawai(G, **kwargs)
        if path is not None:
            plt.savefig(path)


if __name__ == "__main__":
    n = 500
