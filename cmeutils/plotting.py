import numpy as np


def get_histogram(data, normalize=False, bins="auto", x_range=None):
    """Bins a 1-D array of data into a histogram using
    the numpy.histogram method.

    Parameters
    ----------
    data : 1-D numpy.array, required
        Array of data used to generate the histogram
    normalize : boolean, default=False
        If set to true, normalizes the histogram bin heights
        by the sum of data so that the distribution adds
        up to 1
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
    bin_heights, bin_borders = np.histogram(data, bins=bins, range=x_range)
    if normalize is True:
        bin_heights = bin_heights/sum(bin_heights)
    bin_widths = np.diff(bin_borders)
    bin_centers = bin_borders[:-1] + bin_widths / 2
    return bin_centers, bin_heights
