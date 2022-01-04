import scipy.signal as signal


def find_peaks(data, height=None):
    """Finds peaks in 1-D data.
    The peaks are any value greater than the variable entered for height. The data values are 1-D values (like x, y, or z)   
    that you wish to find the peaks of.
    Any points/values that excede the height will be identified as peaks.
    
    Parameters
    ----------
    data : numpy.ndarray, shape (N,1)
       Such as x, y, OR z.
    max_height : int, default None
        Required height of peaks. 
        
    Returns
    ----------
    tuple
        The indices of peaks
    """
    peaks = signal.find_peaks(data, height=height)
    return peaks  
