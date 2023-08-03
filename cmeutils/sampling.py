import numpy as np
from pymbar import timeseries


def equil_sample(
    data, threshold_fraction=0.0, threshold_neff=1, conservative=True
):
    """Returns a statistically independent subset of an array of data.

    Parameters
    ----------
    data : numpy.typing.Arraylike
        1-D time dependent data to check for equilibration.
    threshold_fraction : float, optional, default=0.8
        Fraction of data expected to be equilibrated.
    threshold_neff : int, optional, default=100
        Minimum amount of effectively correlated samples to consider a_t
        'equilibrated'.
    conservative : bool, default=True
        if set to True, uniformly-spaced indices are chosen with interval
        ceil(g), where g is the statistical inefficiency.
        Otherwise, indices are chosen non-uniformly with interval of
        approximately g in order to end up with approximately T/g total indices

    Returns
    -------
    (numpy.ndarray, numpy.ndarray, int, int)

    """
    is_equil, prod_start, ineff, Neff = is_equilibrated(
        data, threshold_fraction, threshold_neff
    )

    if is_equil:
        uncorr_indices = timeseries.subsample_correlated_data(
            data[prod_start:], g=ineff, conservative=conservative
        )
        uncorr_sample = data[prod_start:][uncorr_indices]
        return (uncorr_sample, uncorr_indices, prod_start, Neff)

    else:
        raise ValueError(
            "Property does not have requisite threshold of production data "
            "expected. More production data is needed, or the threshold needs "
            "to be lowered. See is_equilibrated for more information."
        )


def is_equilibrated(data, threshold_fraction=0.50, threshold_neff=50, nskip=1):
    """Check if a dataset is equilibrated based on a fraction of equil data.

    Using `pymbar.timeseries` module, check if a timeseries dataset has enough
    equilibrated data based on two threshold values. The threshold_fraction
    value translates to the fraction of total data from the dataset 'a_t' that
    can be thought of as being in the 'production' region. The threshold_neff
    is the minimum amount of effectively uncorrelated samples to have in a_t to
    consider it equilibrated.

    The `pymbar.timeseries` module returns the starting index of the
    'production' region from 'a_t'. The fraction of 'production' data is
    then compared to the threshold value. If the fraction of 'production' data
    is >= threshold fraction this will return a list of
    [True, t0, g, Neff] and [False, None, None, None] otherwise.

    Parameters
    ----------
    data : numpy.typing.Arraylike
        1-D time dependent data to check for equilibration.
    threshold_fraction : float, optional, default=0.8
        Fraction of data expected to be equilibrated.
    threshold_neff : int, optional, default=100
        Minimum amount of effectively correlated samples to consider a_t
        'equilibrated'.
    nskip : int, optional, default=1
        Since the statistical inefficiency is computed for every time origin
        in a call to timeseries.detect_equilibration, for larger datasets
        (> few hundred), increasing nskip might speed this up, while
        discarding more data.

    Returns
    -------
    list : [True, t0, g, Neff]
        If the data set is considered properly equilibrated
    list : [False, None, None, None]
        If the data set is not considered properly equilibrated

    """
    if threshold_fraction < 0.0 or threshold_fraction > 1.0:
        raise ValueError(
            f"Passed 'threshold_fraction' value: {threshold_fraction}, "
            "expected value between 0.0-1.0."
        )
    threshold_neff = int(threshold_neff)
    if threshold_neff < 1:
        raise ValueError(
            f"Passed 'threshold_neff' value: {threshold_neff}, expected value "
            "1 or greater."
        )
    [t0, g, Neff] = timeseries.detect_equilibration(data, nskip=nskip)
    frac_equilibrated = 1.0 - (t0 / np.shape(data)[0])

    if (frac_equilibrated >= threshold_fraction) and (Neff >= threshold_neff):
        return [True, t0, g, Neff]
    else:
        return [False, None, None, None]
