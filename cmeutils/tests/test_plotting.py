import numpy as np
import pytest

from cmeutils.plotting import get_histogram
from base_test import BaseTest


class TestPlotting(BaseTest):
    def test_histogram_bins(self):
        sample = np.random.randn(100)
        bin_c, bin_h = get_histogram(sample, bins=20)
        assert len(bin_c) == len(bin_h) == 20

    def test_histogram_normalize(self):
        sample = np.random.randn(100)*-1
        bin_c, bin_h = get_histogram(sample, normalize=True)
        for n in bin_h:
            assert n <= 1

