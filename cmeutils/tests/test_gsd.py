import numpy as np
import pytest

from cmeutils import gsd_utils
from base_test import BaseTest


class TestGSD(BaseTest):

    def test_get_type_position(self, test_gsd):
        from cmeutils.gsd_utils import get_type_position

        pos_array = get_type_position(gsd_file = test_gsd, type_name = 'A')
        assert type(pos_array) is type(np.array([]))
        assert pos_array.shape == (2,3)

    def test_validate_inputs(self, test_gsd, test_snap):
        # Catch error with both gsd_file and snap are passed
        with pytest.raises(ValueError):
            snap = gsd_utils._validate_inputs(test_gsd, test_snap, 1)
        with pytest.raises(AssertionError):
            snap = gsd_utils._validate_inputs(test_gsd, None, 1.0)
        with pytest.raises(AssertionError):
            snap = gsd_utils._validate_inputs(None, test_gsd, 1)
        with pytest.raises(OSError):
            gsd_utils._validate_inputs("bad_gsd_file", None, 0)

    def test_get_all_types(self, test_gsd):
        types = gsd_utils.get_all_types(test_gsd)
        assert types == ['A', 'B']

    def test_snap_molecule_cluster(self, test_gsd_bonded):
        cluster = gsd_utils.snap_molecule_cluster(gsd_file=test_gsd_bonded)

