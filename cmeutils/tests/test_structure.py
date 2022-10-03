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
    def test_angle_distribution_deg(self, p3ht_gsd):
        angles = angle_distribution(p3ht_gsd, "cc", "ss", "cc", start=0, stop=1, degrees=True)
        for ang in angles:
            assert 80 < ang < 100

    def test_angle_distribution_range(self, p3ht_gsd):
        angles = angle_distribution(
                p3ht_gsd,
                "cc",
                "ss",
                "cc",
                start=0,
                stop=1,
                histogram=True,
                degrees=True,
                theta_min=10,
                theta_max=180
        )
        assert np.allclose(angles[0,0],10, atol=0.5)
        assert np.allclose(angles[-1,0], 180, atol=0.5)

    def test_angle_distribution_rad(self, p3ht_gsd):
        angles = angle_distribution(p3ht_gsd, "cc", "ss", "cc", start=0, stop=1, degrees=False)
        for ang in angles:
            assert 1.40 < ang < 1.75

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
            print(bond)
            assert 0.45 < bond < 0.52

    def test_bond_distribution_range(self, p3ht_gsd):
        bonds = bond_distribution(
                p3ht_gsd,
                "cc",
                "ss",
                start=0,
                stop=1,
                l_min=0,
                l_max=1,
                histogram=True
        )
        assert np.allclose(bonds[0,0],0, atol=0.5)
        assert np.allclose(bonds[-1,0], 1, atol=0.5)

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

    def test_bond_dist_normalize(self, p3ht_gsd):
        bonds_hist = bond_distribution(
                p3ht_gsd,
                "cc",
                "ss",
                start=0,
                stop=1,
                histogram=True,
                normalize=True
        )
        assert np.allclose(np.sum(bonds_hist[:,1]), 1, 1e-3)

    def test_bond_range_outside(self, p3ht_gsd):
        with pytest.warns(UserWarning):
            bonds_hist = bond_distribution(
                p3ht_gsd,
                "cc",
                "ss",
                start=0,
                stop=1,
                histogram=True,
                l_min=0.52,
                l_max=0.60
            )

    def test_angle_histogram(self, p3ht_gsd):
        angles_hist = angle_distribution(
                p3ht_gsd, "cc", "ss", "cc", start=0, stop=1, histogram=True
        )
        angles_no_hist = angle_distribution(
                p3ht_gsd, "cc", "ss", "cc", start=0, stop=1, histogram=False
        )
        assert angles_hist.ndim == 2
        assert angles_no_hist.ndim == 1

    def test_angle_dist_normalize(self, p3ht_gsd):
        angles_hist = angle_distribution(
                p3ht_gsd,
                "cc",
                "ss",
                "cc",
                start=0,
                stop=1,
                histogram=True,
                normalize=True,
        )
        assert np.allclose(np.sum(angles_hist[:,1]), 1, 1e-3)
    
    def test_angle_range_outside(self, p3ht_gsd):
        with pytest.warns(UserWarning):
            angles_hist = angle_distribution(
                    p3ht_gsd,
                    "cc",
                    "ss",
                    "cc",
                    start=0,
                    stop=1,
                    histogram=True,
                    theta_min = 120,
                    theta_max=180
            )

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
