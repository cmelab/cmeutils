import pytest

from cmeutils.tests.base_test import BaseTest
from cmeutils.dynamics import msd_from_gsd

class TestDynamics(BaseTest):
    def test_gsd_msd(self, gsdfile):
        msd_window = msd_from_gsd(gsdfile)
        msd_direct = msd_from_gsd(gsdfile, msd_mode="direct")
