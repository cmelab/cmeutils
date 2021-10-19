import numpy as np
import pytest

from base_test import BaseTest
from cmeutils.geometry import get_plane_normal, angle_between_vectors


class TestGeometry(BaseTest):
    def test_get_plane_normal(self):
        points = np.array(
            [[ -6.2895217, -12.332656 , -13.624254 ],
             [ -6.6731234, -12.346686 , -13.5787945],
             [ -6.828023 , -11.994795 , -13.309965 ],
             [ -6.3998036, -11.814078 , -13.244863 ],
             [ -6.1292906, -12.02033  , -13.428385 ]]
        )
        ctr, norm = get_plane_normal(points)
        assert np.allclose(ctr, np.array([-6.4639525, -12.101709, -13.437253]))
        assert np.allclose(norm, np.array([0.1170466, -0.5699933, 0.8132698]))

        with pytest.raises(AssertionError):
            get_plane_normal(points[:2])

    def test_angle_between_vectors(self):
        assert np.isclose(
            90, angle_between_vectors(np.array([1,0,0]),np.array([0,1,0]))
        )
        assert np.isclose(
            0, angle_between_vectors(np.array([1,0,0]),np.array([-1,0,0]))
        )
