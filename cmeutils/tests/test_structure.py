import pytest

import numpy as np
import freud

from cmeutils.tests.base_test import BaseTest
from cmeutils.structure import gsd_rdf, get_quaternions, order_parameter, all_atom_rdf


class TestStructure(BaseTest):
    def test_gsd_rdf(self, gsdfile_bond):
        rdf_ex, norm = gsd_rdf(gsdfile_bond, "A", "B")
        rdf_noex, norm2 = gsd_rdf(gsdfile_bond, "A", "B", exclude_bonded=False)
        assert norm2 == 1
        assert not np.array_equal(rdf_noex, rdf_ex)

    def test_gsd_rdf_samename(self, gsdfile_bond):
        rdf_ex, norm = gsd_rdf(gsdfile_bond, "A", "A")
        rdf_noex, norm2 = gsd_rdf(gsdfile_bond, "A", "A", exclude_bonded=False)
        assert norm2 == 1
        assert not np.array_equal(rdf_noex, rdf_ex)

    def test_get_quaternions(self):
        with pytest.raises(ValueError):
            get_quaternions(0)

        with pytest.raises(ValueError):
            get_quaternions(5.3)

        qs = get_quaternions()
        assert len(qs) == 20
        assert len(qs[0]) == 4

    def test_order_parameter(self, p3ht_gsd, p3ht_cg_gsd, mapping):
        r_max = 2
        a_max = 30

        order, cl_idx = order_parameter(
            p3ht_gsd, p3ht_cg_gsd, mapping, r_max, a_max
        )

        assert np.isclose(order[0], 0.33125)
        assert len(cl_idx[0]) == 160

    def test_all_atom_rdf(self, gsdfile):
        rdf = all_atom_rdf(gsdfile)
        assert isinstance(rdf, freud.density.RDF)
