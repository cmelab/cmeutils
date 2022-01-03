import scipy.signal as signal


def find_peaks(data):
    """Finds peaks in 1-D data.
    The peaks are any value greater than the average of all the data values. The data values are 1-D values (like x, y, or z) that     you wish to find the peaks of. 
    Any points/values that excede the max height will be identified as peaks.
    
    Parameters
    ----------
    data : numpy.ndarray, shape (N,1)
       Such as x, y, OR z.
    Returns
    ----------
    tuple
        The indices of peaks
    """
    total_peaks = signal.find_peaks(data)
    avg_peaks = sum(data) / len(data)
    peaks = signal.find_peaks(data, height=avg_peaks)
    return peaks 
