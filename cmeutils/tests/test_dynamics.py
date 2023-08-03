import pytest

from cmeutils.dynamics import msd_from_gsd
from cmeutils.tests.base_test import BaseTest


class TestDynamics(BaseTest):
    def test_gsd_msd(self, gsdfile_images):
        msd_window = msd_from_gsd(gsdfile_images)
        msd_direct = msd_from_gsd(gsdfile_images, msd_mode="direct")
        assert msd_window is not None
        assert msd_direct is not None

    def test_gsd_no_images(self, gsdfile):
        with pytest.warns(UserWarning):
            msd = msd_from_gsd(gsdfile)
        assert msd is not None
