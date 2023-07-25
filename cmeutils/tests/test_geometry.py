import math

import numpy as np
import pytest

from base_test import BaseTest
from cmeutils.geometry import get_plane_normal, angle_between_vectors, moit, \
    radial_grid_positions, \
    spherical_grid_positions


class TestGeometry(BaseTest):
    def test_moit(self):
        _moit = moit(points=[(-1, 0, 0), (1, 0, 0)], masses=[1, 1])
        assert np.array_equal(_moit, np.array([0, 2., 2.]))

    def test_get_plane_normal(self):
        points = np.array(
            [[1, 0, 0],
             [0, 1, 0],
             [-1, 0, 0],
             [0, -1, 0]]
        )
        ctr, norm = get_plane_normal(points)
        assert np.allclose(ctr, np.array([0, 0, 0]))
        assert np.allclose(norm, np.array([0, 0, 1]))

        with pytest.raises(AssertionError):
            get_plane_normal(points[:2])

    def test_angle_between_vectors_deg(self):
        assert np.isclose(
            90, angle_between_vectors(np.array([1, 0, 0]), np.array([0, 1, 0]))
        )
        assert np.isclose(
            0, angle_between_vectors(np.array([1, 0, 0]), np.array([-1, 0, 0]))
        )

    def test_angle_between_vectors_rad(self):
        assert np.isclose(
            math.pi / 2,
            angle_between_vectors(
                np.array([1, 0, 0]), np.array([0, 1, 0]), degrees=False
            )
        )
        assert np.isclose(
            0,
            angle_between_vectors(
                np.array([1, 0, 0]), np.array([-1, 0, 0]), degrees=False
            )
        )

    def test_radial_grid_positions(self):
        grid = radial_grid_positions(
            init_radius=1, final_radius=2, init_position=np.zeros(2),
            n_circles=2, circle_slice=2, circle_coverage=np.pi
        )
        assert np.array_equal(
            grid,
            np.array([
                [1., 0.],
                [-1., 0.],
                [2., 0],
                [-2, 0]
            ])
        )

    def test_radial_grid_positions_quarter_circle(self):
        grid = radial_grid_positions(
            init_radius=1, final_radius=2, init_position=np.zeros(2),
            n_circles=2, circle_slice=4, circle_coverage=np.pi / 2
        )
        assert np.array_equal(
            grid,
            np.array([
                [1., 0.],
                [0.866, 0.5],
                [0.5, 0.866],
                [0., 1.],
                [2., 0.],
                [1.732, 1.],
                [1., 1.732],
                [0., 2.]
            ])
        )

    def test_radial_grid_positions_along_x(self):
        grid = radial_grid_positions(
            init_radius=1, final_radius=3, init_position=np.zeros(2),
            n_circles=3, circle_slice=1, circle_coverage=np.pi * 2)
        assert np.array_equal(
            grid,
            np.array([
                [1., 0.],
                [2., 0.],
                [3., 0.]
            ])
        )

    def test_radial_grid_positions_init_position(self):
        grid = radial_grid_positions(
            init_radius=1, final_radius=2, init_position=np.array([1, 1]),
            n_circles=2, circle_slice=2, circle_coverage=np.pi)
        assert np.array_equal(
            grid,
            np.array([
                [2., 1.],
                [0., 1.],
                [3., 1.],
                [-1., 1.]
            ])
        )

    def test_spherical_grid_positions(self):
        grid = spherical_grid_positions(
            init_radius=1, final_radius=2, init_position=np.zeros(3),
            n_circles=2, circle_slice=2, circle_coverage=np.pi, z_coverage=np.pi
        )
        assert np.array_equal(
            grid,
            np.array([
                [0., 0., 1.],
                [0.866, 0., 0.5],
                [0.866, 0., -0.5],
                [0., 0., -1.],
                [-0.866, 0., 0.5],
                [-0.866, 0., -0.5],
                [0., 0., 2.],
                [1.732, 0., 1.0],
                [1.732, 0., -1.],
                [0., 0., -2.],
                [-1.732, 0., 1.],
                [-1.732, 0., -1.]
            ])
        )

    def test_spherical_grid_positions_half_circle(self):
        grid = spherical_grid_positions(
            init_radius=1, final_radius=1, init_position=np.zeros(3),
            n_circles=2, circle_slice=2, circle_coverage=np.pi,
            z_coverage=np.pi / 2
        )
        assert np.array_equal(
            grid,
            np.array([
                [0., 0., 1.],
                [0.5, 0, 0.866],
                [0.866, 0., 0.5],
                [1, 0., 0.],
                [-0.5, 0, 0.866],
                [-0.866, 0., 0.5],
                [-1., 0., 0.]
            ])
        )
