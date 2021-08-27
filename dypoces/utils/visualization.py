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


def point_catplot(title=None, **kwargs):
    sns.set(font_scale=2)
    # Kind of hardcoded mplrc.
    g = sns.catplot(
        height=8,
        aspect=0.8,
        capsize=0.5,
        errwidth=0.7,
        size=10,
        palette="hls",
        kind="point",
        **kwargs
    )
    g.fig.suptitle(title, y=1.05)
    g.despine(left=True)
    return g


def heatmap_facetgrid(
    data=None, x=None, y=None, hue=None, row=None, col=None, title=None
):
    sns.set(font_scale=2)

    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop("data")
        d = data.pivot(
            index=args[1],
            columns=args[0],
            values=args[2],
        )
        sns.heatmap(d, **kwargs)

    g = sns.FacetGrid(data, col=col, row=row, height=8, aspect=0.8, size=10)
    g.map_dataframe(
        draw_heatmap, x, y, hue, cbar=True, square=True, cbar_kws={"shrink": 0.5}
    )
    g.fig.suptitle(title, y=0.9)
    facecolor = g.fig.get_facecolor()
    for ax in g.axes.flat:
        # set aspect of all axis
        ax.set_aspect("equal", "box")
        # set background color of axis instance
        ax.set_facecolor(facecolor)
    return g


if __name__ == "__main__":
    pass
