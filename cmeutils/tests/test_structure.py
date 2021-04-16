import pytest

from cmeutils.tests.base_test import BaseTest
from cmeutils import structure

class TestStructure(BaseTest):
    def test_gsd_rdf(self, test_gsd_bonded):
        rdf_exclude, norm = structure.gsd_rdf(test_gsd_bonded, "A", "B")
        rdf_dont_exclude, norm2 = structure.gsd_rdf(test_gsd_bonded, "A", "B", exclude_bonded=False)
        assert norm2 == 1
        assert rdf_dont_exclude != rdf_exclude

