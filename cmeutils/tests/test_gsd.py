import numpy as np
import pytest

from base_test import BaseTest


class TestGSD(BaseTest):
    def test_get_type_position(self, gsdfile):
        from cmeutils.gsd_utils import get_type_position
        pos_array = get_type_position(gsd_file=gsdfile, typename="A")
        assert type(pos_array) is type(np.array([]))
        assert pos_array.shape == (2, 3)

    def test_get_multiple_types(self, gsdfile):
        from cmeutils.gsd_utils import get_type_position
        pos_array = get_type_position(gsd_file=gsdfile, typename=["A", "B"])
        assert type(pos_array) is type(np.array([]))
        assert pos_array.shape == (5,3)

    def test_validate_inputs(self, gsdfile, snap):
        from cmeutils.gsd_utils import _validate_inputs

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
        from cmeutils.gsd_utils import get_all_types

        types = get_all_types(gsdfile)
        assert types == ["A", "B"]

    def test_snap_molecule_cluster(self, gsdfile_bond):
        from cmeutils.gsd_utils import snap_molecule_cluster

        cluster = snap_molecule_cluster(gsd_file=gsdfile_bond)
        assert np.array_equal(cluster, [0, 1, 0, 1, 2])
