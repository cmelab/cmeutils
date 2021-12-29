import pytest

from base_test import BaseTest

from cmeutils.plot_tools import find_peaks

class TestPlot(BaseTest):
    def test_plot(self, data, max_height):
        peaks = signal.find_peaks(data, height = max_height)
        assert isinstance(peaks, tuple)