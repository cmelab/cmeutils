import pytest

from cmeutils.tests.base_test import BaseTest


class TestStructure(BaseTest):
    def test_gsd_rdf(self, gsdfile_bond):
        from cmeutils.structure import gsd_rdf

        rdf_ex, norm = gsd_rdf(gsdfile_bond, "A", "B")
        rdf_noex, norm2 = gsd_rdf(gsdfile_bond, "A", "B", exclude_bonded=False)
        assert norm2 == 1
        assert rdf_noex != rdf_ex

    def test_gsd_rdf_samename(self, gsdfile_bond):
        from cmeutils.structure import gsd_rdf

        rdf_ex, norm = gsd_rdf(gsdfile_bond, "A", "B")
        rdf_noex, norm2 = gsd_rdf(gsdfile_bond, "A", "B", exclude_bonded=False)
        assert norm2 == 1
        assert rdf_noex != rdf_ex
