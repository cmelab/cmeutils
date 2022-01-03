import pytest

import scipy.signal as signal
import numpy as np

from base_test import BaseTest

from cmeutils.plot_tools import find_peaks

class TestPlot(BaseTest):
    def test_plot(self, rdf_txt):
        line= np.genfromtxt(rdf_txt, names=True, delimiter=",")
        y= line["rdf"]
        peaks = signal.find_peaks(y)
        assert isinstance(y, np.ndarray)
        assert isinstance(peaks, tuple)

