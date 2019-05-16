# -*- coding: utf-8 -*-
"""
Plot Domingos and GE decomposition.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D


class MidpointNormalize(colors.Normalize):
    """
    Normalize a color map, based on some point in the middle.

    We use this to optimize the color mapping for the 3D GE decomposition
    plots, as we want to be able to distinguish low GEs (<25) in particular.

    Source: https://matplotlib.org/tutorials/colors/colormapnorms.html
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plotBiasVarianceDomingos(L, B, V, Vu, Vb, x_label, avg_accuracy=None,
                             x_steps=None, x_scale='linear', title=None,
                             save_loc=None):
    """
    Plot Domingos bias-variance decomposition for 0-1 loss. See [1].

    Parameters
    ----------
    L, B, V, Vu, Vb : iterables of length k, indicating total
    loss, bias, total variance, unbiased and biased variance, respectively.

    x_label : description of x-axis

    avg_accuracy : numpy array of length k: average accuracy. Is plotted
    when provided.

    x_steps: list of k numbers, indicating the x-ticks for the
    entries of L, B, V, Vu, Vb(, avg_accuracy)

    x_scale : 'linear' (default) or 'log'

    title : str, added to plot when provided.

    save_loc : when provided, plot will be saved to this location

    Returns
    -------
    -

    References
    ----------
    [1] Domingos,P.:A unified bias-variance decomposition and its
    applications. In: Langley, P. (ed.) Proceedings of the Seventeenth
    International Conference on Machine Learning (ICML 2000),
    Stanford University, Stanford, CA, USA, June 29–July 2, 2000,
    pp. 231–238. Morgan Kaufmann (2000)
    """

    # Check data
    assert len(L) == len(B) == len(V) == len(Vu) == len(Vb), \
        'First 5 arguments (L, B, V, Vu, Vb) must have same length'

    if avg_accuracy is not None:
        assert len(avg_accuracy) == len(L), \
            'avg_accuracy must have same length as the first 5 arguments'

    if x_steps is None:
        x_steps = list(range(len(L)))

    plt.figure(figsize=(6, 3))

    if title is not None:
        plt.title(title)

    plt.xlabel(x_label)
    plt.ylabel("Loss (%)")
    plt.xscale(x_scale)
    plt.xlim(x_steps[0], x_steps[-1])
    plt.ylim(0, 1)

    plt.grid()

    plt.plot(x_steps, L, marker='+', linestyle='-', color="black", label="L")
    plt.plot(x_steps, B, marker='x', linestyle='--', color="black", label="B")
    plt.plot(x_steps, V, marker='*', linestyle='--', color="black", label="V")
    plt.plot(x_steps, Vu, marker='D', linestyle=':', color="black", label="Vu")
    plt.plot(x_steps, Vb, marker='s', linestyle='-.', color="black",
             label="Vb")

    if avg_accuracy is not None:
        plt.plot(x_steps, avg_accuracy, linestyle='-', color="blue",
                 label="Accuracy")

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    if save_loc is not None:
        plt.savefig(save_loc, dpi=300)


def plotGE3D(avgs, var_r, var_i, x_label, x_steps=None, title=None,
             save_loc=None):
    """
    Plot 3D graph of bias-variance, including plus and minus standard deviation

    Parameters
    ----------
    avgs, var_r, var_i: n x k numpy arrays of planes of GE, with n the number
    of test traces and k the number of different variables (e.g. complexity)
    tried

    x_label : description of x-axis

    x_steps: list of k numbers, indicating the x-ticks for the
    entries of L, B, V, Vu, Vb(, avg_accuracy)

    x_scale : 'linear' (default) or 'log'

    title : str, added to plot when provided.

    save_loc : when provided, plot will be saved to this location

    Returns
    -------
    -

    """

    # Check data
    assert avgs.shape == var_r.shape == var_i.shape, \
        'First 3 arguments (avgs, var_r, var_i) must have same shape'

    test_length = len(avgs)

    if x_steps is None:
        print('avgs.shape: {}'.format(avgs.shape))
        x_steps = np.arange(0, avgs.shape[1], 1)

    Y = np.arange(0, test_length, 1)
    X, Y = np.meshgrid(x_steps, Y)

    fig = plt.figure(figsize=(6, 3))
    ax = fig.gca(projection='3d')

    # Normalize the colors in the graph
    normalize = MidpointNormalize(vmin=0, vmax=150, midpoint=25)

    var_r_plane = ax.plot_surface(X, Y, var_r, color='grey',
                                  alpha=0.25, linewidth=0, antialiased=True)


    var_i_plane = ax.plot_surface(X, Y, var_i, color='grey',
                                  alpha=0.20, linewidth=0, antialiased=True)

    # Good colormaps: cm.viridis_r, cm.RdYlGn_r
    bias_plane = ax.plot_surface(
            X, Y, avgs, norm=normalize, cmap=cm.RdYlGn_r,
            alpha=0.8, linewidth=0, antialiased=True)

    ax.set_xlabel(x_label)
    ax.set_ylabel('Attack set size')
    ax.set_zlabel('Guessing entropy')

    ax.set_xlim(x_steps[0], x_steps[-1])
    # ax.set_zlim(0, 130)

    if title is not None:
        plt.title(title)

    ax.view_init(40, 40)

    plt.show()

    if save_loc is not None:
        plt.subplots_adjust(left=0.05, right=1.0, top=1.0, bottom=0.1)
        plt.savefig(save_loc, dpi=300)
