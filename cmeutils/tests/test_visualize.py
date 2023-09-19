import copy

import numpy as np
import pytest

from cmeutils.tests.base_test import BaseTest
from cmeutils.visualize import FresnelGSD


class TestFresnelGSD(BaseTest):
    def test_view(self, p3ht_fresnel):
        p3ht_fresnel.view()

    def test_path_trace(self, p3ht_fresnel):
        p3ht_fresnel.path_trace(samples=10, light_samples=1)

    def test_scale_diameter(self, p3ht_fresnel):
        p3ht_fresnel.diameter_scale = 0.6
        assert p3ht_fresnel.diameter_scale == 0.6
        assert np.array_equal(
            p3ht_fresnel.radius, np.ones_like(p3ht_fresnel.radius) * 0.60
        )

    def test_update_snapshot(self, gsdfile):
        test_fresnel = FresnelGSD(gsd_file=gsdfile)
        snap = test_fresnel.snapshot
        test_fresnel.frame = 2
        snap2 = test_fresnel.snapshot
        assert test_fresnel.frame == 2
        assert snap is not snap2

    def test_color_dict(self, p3ht_fresnel):
        p3ht_fresnel.color_dict = {
            "c3": np.array([0.5, 0.5, 0.5]),
            "cc": np.array([0.5, 0.5, 0.5]),
            "cd": np.array([0.5, 0.5, 0.5]),
            "h4": np.array([0.5, 0.5, 0.5]),
            "ha": np.array([0.5, 0.5, 0.5]),
            "hc": np.array([0.5, 0.5, 0.5]),
            "ss": np.array([0.5, 0.5, 0.5]),
        }
        p3ht_fresnel.set_type_color(particle_type="c3", color=(0.1, 0.1, 0.1))
        assert np.array_equal(
            p3ht_fresnel.color_dict["c3"], np.array([0.1, 0.1, 0.1])
        )
        p3ht_fresnel.view()

    def test_camera_position(self, p3ht_fresnel):
        init_pos = p3ht_fresnel.camera_position
        p3ht_fresnel.view_axis = (1, 1, 0)
        assert not all(np.not_equal(init_pos, p3ht_fresnel.camera_position))
        box = p3ht_fresnel.snapshot.configuration.box[:3]
        assert np.array_equal(
            p3ht_fresnel.camera_position, box * np.array([1, 1, 0])
        )
        camera = p3ht_fresnel.camera()
        assert np.array_equal(camera.position, box * np.array([1, 1, 0]))

    def test_set_color_no_type(self, p3ht_fresnel):
        with pytest.raises(ValueError):
            p3ht_fresnel.set_type_color("ca", (0.1, 0.1, 0.1))

    def test_set_bad_frame(self, p3ht_fresnel):
        with pytest.raises(ValueError):
            p3ht_fresnel.frame = 10

    def test_set_bad_view(self, p3ht_fresnel):
        with pytest.raises(ValueError):
            p3ht_fresnel.view_axis = 10

    def test_bad_color_dict(self, p3ht_fresnel):
        with pytest.raises(ValueError):
            p3ht_fresnel.color_dict = np.array([0.1, 0.1, 0.1])

    def test_set_bad_unwrap_pos(self, p3ht_fresnel):
        with pytest.raises(ValueError):
            p3ht_fresnel.unwrap_positions = "true"

    def set_set_colors(self, p3ht_fresnel):
        colors = np.zeros_like(p3ht_fresnel.positions)
        colors[:] = np.array([0.5, 0.5, 0.5])
        p3ht_fresnel.colors = colors
        assert all(np.array_equal(colors, p3ht_fresnel.colors))

    def test_unwrap_positions(self, p3ht_fresnel):
        assert p3ht_fresnel.unwrap_positions is False
        p3ht_fresnel.unwrap_positions = True
        assert p3ht_fresnel.unwrap_positions is True

    def test_view_axis(self, p3ht_fresnel):
        p3ht_fresnel.view_axis = (1, 1, 1)
        assert np.array_equal(np.array([1, 1, 1]), p3ht_fresnel.view_axis)
        p3ht_fresnel.view_axis = [0, 1, 1]
        assert np.array_equal(np.array([0, 1, 1]), p3ht_fresnel.view_axis)

    def test_solid(self, p3ht_fresnel):
        p3ht_fresnel.solid = 0.5
        assert p3ht_fresnel.solid == 0.5
        material = p3ht_fresnel.material()
        assert material.solid == 0.5

    def test_roughness(self, p3ht_fresnel):
        p3ht_fresnel.roughness = 0.5
        assert p3ht_fresnel.roughness == 0.5
        material = p3ht_fresnel.material()
        assert material.roughness == 0.5

    def test_specular(self, p3ht_fresnel):
        p3ht_fresnel.specular = 0.5
        assert p3ht_fresnel.specular == 0.5
        material = p3ht_fresnel.material()
        assert material.specular == 0.5

    def test_specular_trans(self, p3ht_fresnel):
        p3ht_fresnel.specular_trans = 0.5
        assert p3ht_fresnel.specular_trans == 0.5
        material = p3ht_fresnel.material()
        assert material.spec_trans == 0.5

    def test_metal(self, p3ht_fresnel):
        p3ht_fresnel.metal = 0.5
        assert p3ht_fresnel.metal == 0.5
        material = p3ht_fresnel.material()
        assert material.metal == 0.5

    def test_geometry(self, p3ht_fresnel):
        geometry = p3ht_fresnel.geometry()
        for i, j in zip(p3ht_fresnel.positions, geometry.position):
            assert np.array_equal(i, j)
        for i, j in zip(p3ht_fresnel.radius, geometry.radius):
            assert i == j

    def test_height(self, p3ht_fresnel):
        p3ht_fresnel.height = 5
        assert p3ht_fresnel.height == 5
        camera = p3ht_fresnel.camera()
        assert camera.height == 5

    def test_up(self, p3ht_fresnel):
        p3ht_fresnel.up = np.array([1, 0, 0])
        camera = p3ht_fresnel.camera()
        assert np.array_equal(camera.up, np.array([1, 0, 0]))

    def test_box_length(self, p3ht_fresnel):
        assert np.allclose(
            p3ht_fresnel.box_length, p3ht_fresnel.snapshot.configuration.box
        )

    def test_box_radius(self, p3ht_fresnel):
        p3ht_fresnel.box_radius = 0.1
        assert p3ht_fresnel.box_radius == 0.1

    def test_default_height(self, p3ht_fresnel):
        assert (
            p3ht_fresnel.height
            == np.linalg.norm(
                p3ht_fresnel.box_length[:3] * p3ht_fresnel.view_axis
            )
            * p3ht_fresnel._height_factor
        )

    def test_reset_height(self, p3ht_fresnel):
        default_height = copy.deepcopy(p3ht_fresnel.height)
        p3ht_fresnel.height = 5
        p3ht_fresnel.reset_height()
        assert p3ht_fresnel.height == default_height

    def test_height_after_update_frame(self, gsdfile):
        test_fresnel = FresnelGSD(gsd_file=gsdfile)
        test_fresnel.height = 5
        test_fresnel.frame = 2
        assert test_fresnel.height != 5

    def test_show_box(self, p3ht_fresnel):
        p3ht_fresnel.view()
        assert len(p3ht_fresnel.scene.geometry) == 2
        p3ht_fresnel.show_box = False
        p3ht_fresnel.view()
        assert len(p3ht_fresnel.scene.geometry) == 1
