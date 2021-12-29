import pytest

from base_test import BaseTest
from cmeutils.dynamics import msd_from_gsd

class TestDynamics(BaseTest):
    def test_gsd_msd(self, gsdfile_images):
        msd_window = msd_from_gsd(gsdfile_images)
        msd_direct = msd_from_gsd(gsdfile_images, msd_mode="direct")

    def test_gsd_no_images(self, gsdfile):
        with pytest.warns(UserWarning):
            msd = msd_from_gsd(gsdfile)
