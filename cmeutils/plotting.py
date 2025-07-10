import matplotlib.pyplot as plt
import numpy as np


def get_histogram(
    data, normalize=False, as_probability=False, bins="auto", x_range=None
):
    """Bins a 1-D array of data into a histogram using
    the numpy.histogram method.

    Parameters
    ----------
    data : 1-D numpy.array, required
        Array of data used to generate the histogram
    normalize : bool, default=False
        If set to true, normalizes the histogram bin heights
        by the sum of data so that the distribution adds
        up to 1. This gives a probability density function (PDF)
        where the bin_heights represents the probability density.
        See `as_probability` to convert from density to per-bin
        probability.
    as_probability : bool, default=False
        If `True`, converts the normalized PDF into a probability mass
        function (PMF) by multiplying each bin heigh by its corresponding
        bin width.
    bins : float, int, or str, default="auto"
        Method used by numpy to determine bin borders.
        Check the numpy.histogram docs for more details.
    x_range : (float, float), default = None
        The lower and upper range of the histogram bins.
        If set to None, then the min and max values of data are used.

    Returns
    -------
    bin_cetners : 1-D numpy.array
        Array of the bin center values
    bin_heights : 1-D numpy.array
        Array of the bin height values

    """
    bin_heights, bin_borders = np.histogram(
        a=data, bins=bins, range=x_range, density=normalize
    )
    bin_widths = np.diff(bin_borders)
    bin_centers = bin_borders[:-1] + bin_widths / 2
    if as_probability:
        bin_heights *= bin_widths
    return bin_centers, bin_heights


def threedplot(
    x,
    y,
    z,
    xlabel="xlabel",
    ylabel="ylabel",
    zlabel="zlabel",
    plot_name="plot_name",
):
    """Plot a 3d heat map from 3 lists of numbers. This function is useful
    for plotting a dependent variable as a function of two independent
    variables.
    In the example below we use f(x,y)= -x^2 - y^2 +6 because it looks cool.

    Example
    -------

    We create two indepent variables and a dependent variable in the z axis and
    plot the result. Here z is the equation of an elliptic paraboloid.

    import random

    x = []
    for i in range(0,1000):
        n = random.uniform(-20,20)
        x.append(n)

    y = []
    for i in range(0,1000):
        n = random.uniform(-20,20)
        y.append(n)

    z = []
    for i in range(0,len(x)):
        z.append(-x[i]**2 - y[i]**2 +6)

    fig = threedplot(x,y,z)
    fig.show()

    Parameters
    ----------

    x,y,z : list of int/floats

    xlabel, ylabel, zlabel : str

    plot_name : str


    """
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = plt.axes(projection="3d")
    ax.set_xlabel(xlabel, fontdict=dict(weight="bold"), fontsize=12)
    ax.set_ylabel(ylabel, fontdict=dict(weight="bold"), fontsize=12)
    ax.set_zlabel(zlabel, fontdict=dict(weight="bold"), fontsize=12)
    p = ax.scatter(x, y, z, c=z, cmap="rainbow", linewidth=7)
    plt.colorbar(p, pad=0.1, aspect=2.3)

    return fig
