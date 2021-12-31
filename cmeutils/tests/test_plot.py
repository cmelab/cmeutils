import pytest

import scipy.signal as signal
import numpy as np

from base_test import BaseTest

from cmeutils.plot_tools import find_peaks

class TestPlot(BaseTest):
    def test_plot(self):
        x = [1,2,3,4,5]
        data = np.array(x)
        peaks = signal.find_peaks(data)
        assert isinstance(data, np.ndarray)
        assert isinstance(peaks, tuple)