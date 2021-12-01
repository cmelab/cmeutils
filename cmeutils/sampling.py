import numpy as np
from pymbar import timeseries

def equil_sample(
        data, threshold_fraction=0.0, threshold_neff=1, conservative=True
    ):
    is_equil, prod_start, ineff, Neff = is_equilibrated(
            data, threshold_fraction, threshold_neff
            )

    if is_equil is True:
        uncorr_indices = timeseries.subsampleCorrelatedData(
                data[prod_start:], g=ineff, conservative=conservative
            )
        uncorr_sample = data[prod_start:][uncorr_indices]
        return(uncorr_sample, uncorr_indices, prod_start, Neff)

    elif is_equil is False:
        raise ValueError(
            "Property does not have requisite threshold of production data "
            "expected. More production data is needed, or the threshold needs "
            "to be lowered. See is_equilibrated for more information."
        )

def is_equilibrated(data, threshold_fraction=0.80, threshold_neff=100, nskip=1):
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
    [t0, g, Neff] = timeseries.detectEquilibration(data, nskip=nskip)
    frac_equilibrated = 1.0 - (t0 / np.shape(data)[0])

    if (frac_equilibrated >= threshold_fraction) and (Neff >= threshold_neff):
        return [True, t0, g, Neff]
    else:
        return [False, None, None, None]

def trim_non_equilibrated(data, threshold_fraction=0.80, threshold_neff=100, nskip=1):
    """Prune timeseries array to just the production data.

    Refer to equilibration.is_equilibrated for addtional information.

    This method returns a list of length 3, where list[0] is the trimmed array,
    list[1] is the index of the original dataset where equilibration begins,
    list[2] is the calculated statistical inefficiency, which can be used
    when subsampling the data using `pymbar.timseries.subsampleCorrelatedData`,
    list[3] is the number of effective uncorrelated data points.

    Refer to https://pymbar.readthedocs.io/en/master/timeseries.html for
    additional information.

    Parameters
    ----------
    data : numpy.typing.Arraylike
        1-D time dependent data to check for equilibration.
    threshold_fraction : float, optional, default=0.75
        Fraction of data expected to be equilibrated.
    threshold_neff : int, optional, default=100
        Minimum amount of uncorrelated samples.
    nskip : int, optional, default=1
        Since the statistical inefficiency is computed for every time origin
        in a call to timeseries.detectEquilibration, for larger datasets
        (> few hundred), increasing nskip might speed this up, while
        discarding more data.
    """
    [truth, t0, g, Neff] = is_equilibrated(
        data,
        threshold_fraction=threshold_fraction,
        threshold_neff=threshold_neff,
        nskip=nskip,
    )
    if not truth:
        raise ValueError(
            f"Data with a threshold_fraction of {threshold_fraction} and "
            f"threshold_neff {threshold_neff} is not equilibrated!"
        )

    return [data[t0:], t0, g, Neff]
