import numpy as np
import pytest

from cmeutils.sampling import equil_sample, is_equilibrated
from cmeutils.tests.base_test import BaseTest


class TestSampler(BaseTest):
    def test_is_equilibrated(self, correlated_data_tau100_n10000):
        data = correlated_data_tau100_n10000
        assert not is_equilibrated(
            data, threshold_fraction=0.80, threshold_neff=100
        )[0]
        assert not is_equilibrated(
            data, threshold_fraction=0.40, threshold_neff=100
        )[0]
        assert is_equilibrated(data, threshold_fraction=0.10, threshold_neff=1)[
            0
        ]

        assert not is_equilibrated(
            correlated_data_tau100_n10000,
            threshold_fraction=0.10,
            threshold_neff=5000,
        )[0]
        assert not is_equilibrated(
            correlated_data_tau100_n10000,
            threshold_fraction=0.10,
            threshold_neff=9999,
        )[0]
        assert is_equilibrated(
            correlated_data_tau100_n10000,
            threshold_fraction=0.10,
            threshold_neff=10,
        )[0]

    def test_incorrect_threshold_fraction(self, correlated_data_tau100_n10000):
        with pytest.raises(
            ValueError, match=r"Passed \'threshold_fraction\' value"
        ):
            is_equilibrated(
                correlated_data_tau100_n10000, threshold_fraction=2.0
            )

        with pytest.raises(
            ValueError, match=r"Passed \'threshold_fraction\' value"
        ):
            is_equilibrated(
                correlated_data_tau100_n10000, threshold_fraction=-2.0
            )

    def test_incorrect_threshold_neff(self, correlated_data_tau100_n10000):
        data = correlated_data_tau100_n10000
        with pytest.raises(
            ValueError, match=r"Passed \'threshold_neff\' value"
        ):
            is_equilibrated(data, threshold_fraction=0.75, threshold_neff=0)
        with pytest.raises(
            ValueError, match=r"Passed \'threshold_neff\' value"
        ):
            is_equilibrated(data, threshold_fraction=0.75, threshold_neff=-1)

    def test_return_trimmed_data(self, correlated_data_tau100_n10000):
        data = correlated_data_tau100_n10000
        [equil_data, uncorr_indices, prod_start, Neff] = equil_sample(
            data, threshold_fraction=0.2, threshold_neff=10
        )
        assert np.shape(equil_data)[0] < np.shape(data)[0]

    def test_trim_high_threshold(self, correlated_data_tau100_n10000):
        data = correlated_data_tau100_n10000
        with pytest.raises(
            ValueError, match=r"Property does not have requisite threshold"
        ):
            [equil_data, uncorr_indices, prod_start, Neff] = equil_sample(
                data, threshold_fraction=0.98
            )
        with pytest.raises(
            ValueError,
            match=r"More production data is needed",
        ):
            [equil_data, uncorr_indices, prod_start, Neff] = equil_sample(
                data, threshold_fraction=0.75, threshold_neff=10000
            )
