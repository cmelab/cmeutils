import numpy as np

from cmeutils import gsd_utils
from base_test import BaseTest


class TestGSD(BaseTest):
    def test_get_type_position(self, test_gsd):
        pos_array = gsd_utils.get_type_position(gsd_file = test_gsd,
                                type_name = 'A')
        assert type(pos_array) is type(np.array([]))
        assert pos_array.shape == (2,3)

