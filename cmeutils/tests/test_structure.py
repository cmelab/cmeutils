import pytest

import gsd

import numpy as np
import freud

from cmeutils.tests.base_test import BaseTest

from cmeutils.structure import (
        angle_distribution,
        bond_distribution,
        gsd_rdf,
        get_quaternions, 
        order_parameter,
        all_atom_rdf,
        get_centers
    )

class TestStructure(BaseTest):
    def test_angle_distribution(self, p3ht_gsd):
        angles = angle_distribution(p3ht_gsd, "cc", "ss", "cc", start=0, stop=1)
        for ang in angles:
            assert 80 < ang < 100

    def test_angle_distribution_order(self, p3ht_gsd):
        angles = angle_distribution(p3ht_gsd, "ss", "cc", "cd", start=0, stop=1)
        angles2 = angle_distribution(p3ht_gsd, "cd", "cc", "ss", start=0, stop=1)
        assert angles.shape[0] > 0
        assert angles.shape == angles2.shape
        assert np.array_equal(angles, angles2)

    def test_angle_not_found(self, p3ht_gsd):
        with pytest.raises(ValueError):
            angles = angle_distribution(p3ht_gsd, "cc", "xx", "cc", start=0, stop=1)

    def test_bond_distribution(self, p3ht_gsd):
        bonds = bond_distribution(p3ht_gsd, "cc", "ss", start=0, stop=1)
        for bond in bonds:
            assert 0.45 < bond < 0.52

    def test_bond_distribution_order(self, p3ht_gsd):
        bonds = bond_distribution(p3ht_gsd, "cc", "ss", start=0, stop=1)
        bonds2 = bond_distribution(p3ht_gsd, "ss", "cc", start=0, stop=1)
        assert bonds.shape == bonds2.shape
        assert np.array_equal(bonds, bonds2)

    def test_bond_not_found(self, p3ht_gsd):
        with pytest.raises(ValueError):
            bonds = bond_distribution(p3ht_gsd, "xx", "ss", start=0, stop=1)

    def test_bond_histogram(self, p3ht_gsd):
        bonds_hist = bond_distribution(
                p3ht_gsd, "cc", "ss", start=0, stop=1, histogram=True
        )
        bonds_no_hist = bond_distribution(
                p3ht_gsd, "cc", "ss", start=0, stop=1, histogram=False
        )
        assert bonds_hist.ndim == 2
        assert bonds_no_hist.ndim == 1

    def test_angle_histogram(self, p3ht_gsd):
        angles_hist = angle_distribution(
                p3ht_gsd, "cc", "ss", start=0, stop=1, histogram=True
        )
        angles_no_hist = angle_distribution(
                p3ht_gsd, "cc", "ss", start=0, stop=1, histogram=False
        )
        assert angles_hist.ndim == 2
        assert angles_no_hist.ndim == 1

    def test_gsd_rdf(self, gsdfile_bond):
        rdf_ex, norm = gsd_rdf(gsdfile_bond, "A", "B")
        rdf_noex, norm2 = gsd_rdf(gsdfile_bond, "A", "B", exclude_bonded=False)
        assert np.isclose(norm2, 2/3, 1e-4)
        assert not np.array_equal(rdf_noex, rdf_ex)

    def test_gsd_rdf_samename(self, gsdfile_bond):
        rdf_ex, norm = gsd_rdf(gsdfile_bond, "A", "A")
        rdf_noex, norm2 = gsd_rdf(gsdfile_bond, "A", "A", exclude_bonded=False)
        assert norm2 == 1
        assert not np.array_equal(rdf_noex, rdf_ex)

    def test_gsd_rdf_pair_order(self, gsdfile_bond):
        rdf, norm = gsd_rdf(gsdfile_bond, "A", "B")
        rdf_y = rdf.rdf*norm
        rdf2, norm2 = gsd_rdf(gsdfile_bond, "B", "A")
        rdf_y2 = rdf2.rdf*norm2

        for i,j in zip(rdf_y, rdf_y2):
            assert np.allclose(i,j,atol=1e-4)

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
        
    def test_get_centers(self, gsdfile):
        new_gsdfile = "centers.gsd"
        centers = get_centers(gsdfile, new_gsdfile) 
        assert isinstance(centers, type(None))
