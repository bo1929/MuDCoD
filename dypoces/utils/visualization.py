import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def catplot_results(df, x, y, hue, col, row, title):
    sns.set_theme(style="whitegrid")
    g = sns.catplot(
        data=df,
        x=x,
        y=y,
        hue=hue,
        col=col,
        row=row,
        palette="hls",
        capsize=0.2,
        height=6,
        aspect=0.75,
        kind="point",
        **kwargs
    )
    g.fig.suptitle(title)
    g.despine(left=True)
    return g


if __name__ == "__main__":
    pass
