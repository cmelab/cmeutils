import scipy.signal as signal


def find_peaks(data, max_height=None):
    """Finds peaks in 2-D data.
    That peaks are determined by the specified value of the max height (which uses y-values/values along the y-axis). 
    Any points/values that excede the max height will be identified as peaks.
    
    Parameters
    ----------
    data : str
        Path to file(gsd or log), or a 1D array.
    max_height : int, default None
        The max value a point can be until it is recognized as a peak.
        
    Returns
    ----------
     tuple   
       """
    peaks = signal.find_peaks(data, height = max_height)
    return peaks