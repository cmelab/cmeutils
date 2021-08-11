import numpy as np
import pytest

from base_test import BaseTest
from cmeutils.gsd_utils import get_type_position, snap_molecule_cluster
from cmeutils.gsd_utils import get_all_types, _validate_inputs


class TestGSD(BaseTest):
    def test_get_type_position(self, gsdfile):
        pos_array = get_type_position(gsd_file=gsdfile, typename="A")
        assert type(pos_array) is type(np.array([]))
        assert pos_array.shape == (2, 3)

    def test_get_multiple_types(self, gsdfile):
        pos_array = get_type_position(gsd_file=gsdfile, typename=["A", "B"])
        assert type(pos_array) is type(np.array([]))
        assert pos_array.shape == (5,3)

    def test_get_position_and_images(self, gsdfile_images):
        pos, imgs = get_type_position(gsd_file=gsdfile_images, typename="A")
        assert type(pos) is type(imgs) is type(np.array([]))
        assert pos.shape == imgs.shape

    def test_validate_inputs(self, gsdfile, snap):
        # Catch errors when both gsd_file and snap are passed
        with pytest.raises(ValueError):
            _validate_inputs(gsdfile, snap, 1)
        with pytest.raises(AssertionError):
            _validate_inputs(gsdfile, None, 1.0)
        with pytest.raises(AssertionError):
            _validate_inputs(None, gsdfile, 1)
        with pytest.raises(OSError):
            _validate_inputs("bad_gsd_file", None, 0)

    def test_get_all_types(self, gsdfile):
        types = get_all_types(gsdfile)
        assert types == ["A", "B"]

    def test_snap_molecule_cluster(self, gsdfile_bond):
        cluster = snap_molecule_cluster(gsd_file=gsdfile_bond)
        assert np.array_equal(cluster, [0, 1, 0, 1, 2])
