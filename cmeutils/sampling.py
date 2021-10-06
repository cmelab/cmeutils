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
            "to be lowered. See project.src.analysis.equilibration.is_equilibrated"
            " for more information."
        )

def is_equilibrated(data, treshold_fraction, threshold_neff):
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
    frac_equilibrated = 1.0 - (t0 / np.shape(a_t)[0])

    if (frac_equilibrated >= threshold_fraction) and (Neff >= threshold_neff):
        return [True, t0, g, Neff]
    else:
        return [False, None, None, None]
