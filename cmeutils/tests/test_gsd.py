import numpy as np

from cmeutils import gsd_utils
from base_test import BaseTest


class TestGSD(BaseTest):
    def test_frame_get_type_position(self, test_gsd):
        pos_array = gsd_utils.frame_get_type_position(test_gsd, 'A')
        assert type(pos_array) is type(np.array([]))
        assert pos_array.shape == (2,3)

