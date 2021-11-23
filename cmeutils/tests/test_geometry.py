import numpy as np
import pytest

from base_test import BaseTest
from cmeutils.geometry import get_plane_normal, angle_between_vectors


class TestGeometry(BaseTest):
    def test_get_plane_normal(self):
        points = np.array(
            [[ 1, 0, 0],
             [ 0, 1, 0],
             [-1, 0, 0],
             [ 0,-1, 0]]
        )
        ctr, norm = get_plane_normal(points)
        assert np.allclose(ctr, np.array([0, 0, 0]))
        assert np.allclose(norm, np.array([0, 0, 1]))

        with pytest.raises(AssertionError):
            get_plane_normal(points[:2])

    def test_angle_between_vectors(self):
        assert np.isclose(
            90, angle_between_vectors(np.array([1,0,0]),np.array([0,1,0]))
        )
        assert np.isclose(
            0, angle_between_vectors(np.array([1,0,0]),np.array([-1,0,0]))
        )
