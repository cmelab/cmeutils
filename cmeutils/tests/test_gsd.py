import numpy as np
import pytest

from cmeutils import gsd_utils
from base_test import BaseTest


class TestGSD(BaseTest):

    def test_get_type_position(self, test_gsd):
        pos_array = gsd_utils.get_type_position(gsd_file = test_gsd,
                                type_name = 'A')
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


