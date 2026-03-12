import math

import freud
import numpy as np
import pytest
from scipy.integrate import trapezoid

from cmeutils.geometry import get_quaternions
from cmeutils.structure import (
    angle_distribution,
    bond_distribution,
    concentration_profile,
    diffraction_pattern,
    dihedral_distribution,
    gsd_rdf,
    order_parameter,
    structure_factor,
)
from cmeutils.tests.base_test import BaseTest


class TestStructure(BaseTest):
    def test_bad_combo(self, butane_gsd):
        with pytest.raises(ValueError):
            dihedral_distribution(
                butane_gsd,
                "c3",
                "c3",
                "c3",
                "c3",
                histogram=True,
                normalize=False,
                as_probability=True,
            )
        with pytest.raises(ValueError):
            angle_distribution(
                butane_gsd,
                "c3",
                "c3",
                "c3",
                histogram=True,
                normalize=False,
                as_probability=True,
            )
        with pytest.raises(ValueError):
            bond_distribution(
                butane_gsd,
                "c3",
                "c3",
                histogram=True,
                normalize=False,
                as_probability=True,
            )

    def test_dihedral_distribution_deg(self, butane_gsd):
        dihedrals = dihedral_distribution(
            butane_gsd, "c3", "c3", "c3", "c3", start=0, stop=1, degrees=True
        )
        for phi in dihedrals:
            assert -180 <= phi <= 180

    def test_dihedral_distribution_rad(self, butane_gsd):
        dihedrals = dihedral_distribution(
            butane_gsd, "c3", "c3", "c3", "c3", degrees=False
        )
        for phi in dihedrals:
            assert -math.pi <= phi <= math.pi

    def test_dihedral_distribution_not_found(self, butane_gsd):
        with pytest.raises(ValueError):
            dihedral_distribution(butane_gsd, "c3", "c3", "c3", "c")

    def test_dihedral_distribution_histogram(self, butane_gsd):
        dihedrals = dihedral_distribution(
            butane_gsd, "c3", "c3", "c3", "c3", histogram=True
        )
        for phi in dihedrals[:, 0]:
            assert -math.pi <= phi <= math.pi

    def test_dihedral_distribution_pdf(self, butane_gsd):
        dihedrals = dihedral_distribution(
            butane_gsd,
            "c3",
            "c3",
            "c3",
            "c3",
            histogram=True,
            normalize=True,
            as_probability=False,
        )
        bin_width = dihedrals[:, 0][1] - dihedrals[:, 0][0]
        assert np.allclose(np.sum(dihedrals[:, 1] * bin_width), 1, 1e-3)

    def test_dihedral_distribution_pmf(self, butane_gsd):
        dihedrals = dihedral_distribution(
            butane_gsd,
            "c3",
            "c3",
            "c3",
            "c3",
            histogram=True,
            normalize=True,
            as_probability=True,
        )
        assert np.allclose(np.sum(dihedrals[:, 1]), 1, 1e-3)

    def test_angle_distribution_deg(self, p3ht_gsd):
        angles = angle_distribution(p3ht_gsd, "cc", "ss", "cc", degrees=True)
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
            theta_max=180,
        )
        assert np.allclose(angles[0, 0], 10, atol=0.5)
        assert np.allclose(angles[-1, 0], 180, atol=0.5)

    def test_angle_distribution_rad(self, p3ht_gsd):
        angles = angle_distribution(
            p3ht_gsd, "cc", "ss", "cc", start=0, stop=1, degrees=False
        )
        for ang in angles:
            assert 1.40 < ang < 1.75

    def test_angle_distribution_order(self, p3ht_gsd):
        angles = angle_distribution(p3ht_gsd, "ss", "cc", "cd", start=0, stop=1)
        angles2 = angle_distribution(
            p3ht_gsd, "cd", "cc", "ss", start=0, stop=1
        )
        assert angles.shape[0] > 0
        assert angles.shape == angles2.shape
        assert np.array_equal(angles, angles2)

    def test_angle_not_found(self, p3ht_gsd):
        with pytest.raises(ValueError):
            angle_distribution(p3ht_gsd, "cc", "xx", "cc", start=0, stop=1)

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
            histogram=True,
        )
        assert np.allclose(bonds[0, 0], 0, atol=0.5)
        assert np.allclose(bonds[-1, 0], 1, atol=0.5)

    def test_bond_distribution_order(self, p3ht_gsd):
        bonds = bond_distribution(p3ht_gsd, "cc", "ss", start=0, stop=1)
        bonds2 = bond_distribution(p3ht_gsd, "ss", "cc", start=0, stop=1)
        assert bonds.shape == bonds2.shape
        assert np.array_equal(bonds, bonds2)

    def test_bond_not_found(self, p3ht_gsd):
        with pytest.raises(ValueError):
            bond_distribution(p3ht_gsd, "xx", "ss", start=0, stop=1)

    def test_bond_histogram(self, p3ht_gsd):
        bonds_hist = bond_distribution(
            p3ht_gsd, "cc", "ss", start=0, stop=1, histogram=True
        )
        bonds_no_hist = bond_distribution(
            p3ht_gsd, "cc", "ss", start=0, stop=1, histogram=False
        )
        assert bonds_hist.ndim == 2
        assert bonds_no_hist.ndim == 1

    def test_bond_dist_pdf(self, p3ht_gsd):
        bonds_hist = bond_distribution(
            p3ht_gsd,
            "cc",
            "ss",
            start=0,
            stop=1,
            histogram=True,
            normalize=True,
            as_probability=False,
        )
        bin_width = bonds_hist[:, 0][1] - bonds_hist[:, 0][0]
        assert np.allclose(np.sum(bonds_hist[:, 1] * bin_width), 1, 1e-3)

    def test_bond_dist_pmf(self, p3ht_gsd):
        bonds_hist = bond_distribution(
            p3ht_gsd,
            "cc",
            "ss",
            start=0,
            stop=1,
            histogram=True,
            normalize=True,
            as_probability=True,
        )
        assert np.allclose(np.sum(bonds_hist[:, 1]), 1, 1e-3)

    def test_bond_range_outside(self, p3ht_gsd):
        with pytest.warns(UserWarning):
            bond_distribution(
                p3ht_gsd,
                "cc",
                "ss",
                start=0,
                stop=1,
                histogram=True,
                l_min=0.52,
                l_max=0.60,
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

    def test_angle_dist_pdf(self, p3ht_gsd):
        angles_hist = angle_distribution(
            p3ht_gsd,
            "cc",
            "ss",
            "cc",
            start=0,
            stop=1,
            histogram=True,
            normalize=True,
            as_probability=False,
        )
        bin_width = angles_hist[:, 0][1] - angles_hist[:, 0][0]
        assert np.allclose(np.sum(angles_hist[:, 1] * bin_width), 1, 1e-3)

    def test_angle_dist_pmf(self, p3ht_gsd):
        angles_hist = angle_distribution(
            p3ht_gsd,
            "cc",
            "ss",
            "cc",
            start=0,
            stop=1,
            histogram=True,
            normalize=True,
            as_probability=True,
        )
        assert np.allclose(np.sum(angles_hist[:, 1]), 1, 1e-3)

    def test_angle_range_outside(self, p3ht_gsd):
        with pytest.warns(UserWarning):
            angle_distribution(
                p3ht_gsd,
                "cc",
                "ss",
                "cc",
                start=0,
                stop=1,
                histogram=True,
                theta_min=120,
                theta_max=180,
            )

    def test_diffraction_pattern(self, gsdfile_bond):
        views = get_quaternions(n_views=5)
        dp = diffraction_pattern(gsdfile_bond, views=views)
        assert isinstance(dp, freud.diffraction.DiffractionPattern)

    def test_structure_factor_direct(self, gsdfile_bond):
        sf = structure_factor(gsdfile_bond, k_min=0.2, k_max=5)
        assert isinstance(sf, freud.diffraction.StaticStructureFactorDirect)

    def test_structure_factor_debye(self, gsdfile_bond):
        sf = structure_factor(gsdfile_bond, k_min=0.2, k_max=5, method="debye")
        assert isinstance(sf, freud.diffraction.StaticStructureFactorDebye)

    def test_structure_factor_bad_method(self, gsdfile_bond):
        with pytest.raises(ValueError):
            structure_factor(gsdfile_bond, k_min=0.2, k_max=5, method="a")

    def test_rdf_bad_args(self, AB_chain_gsd):
        with pytest.raises(ValueError):
            gsd_rdf(
                gsdfile=AB_chain_gsd,
                start=0,
                stop=10,
                exclude_bond_depth=2,
                exclude_all_bonded=True,
            )

        with pytest.raises(ValueError):
            gsd_rdf(
                gsdfile=AB_chain_gsd,
                A_name="A",
                start=0,
                stop=10,
            )

    def test_gsd_rdf(self, AB_chain_gsd):
        rdf, scale_factor = gsd_rdf(
            gsdfile=AB_chain_gsd,
            start=0,
            stop=10,
            exclude_bond_depth=0,
            exclude_all_bonded=False,
        )
        assert isinstance(rdf, freud.density.RDF)
        rdf.rdf
        assert scale_factor == 1

    def test_gsd_rdf_exclude_all_bonded(self, AB_chain_gsd):
        rdf, scale_factor = gsd_rdf(
            gsdfile=AB_chain_gsd,
            start=0,
            stop=10,
            exclude_all_bonded=True,
        )
        assert scale_factor == 1
        assert np.array_equal(rdf.rdf, np.zeros_like(rdf.rdf))

    def test_gsd_rdf_exclusions(self, AB_chain_gsd):
        """Exclude bonded neighbor."""
        rdf, scale_factor = gsd_rdf(
            gsdfile=AB_chain_gsd,
            start=0,
            stop=10,
            exclude_bond_depth=1,
            exclude_all_bonded=False,
        )
        assert scale_factor != 1
        # In this GSD file, the bond length is ~1, so the first peak shows up around r=1
        zero_indices = np.where(rdf.bin_centers < 1.5)[0]
        zero_array = np.zeros_like(zero_indices)
        rdf_values = rdf.rdf[zero_indices]
        assert np.allclose(rdf_values, zero_array)

        rdf, scale_factor = gsd_rdf(
            gsdfile=AB_chain_gsd,
            start=0,
            stop=10,
            exclude_bond_depth=2,
            exclude_all_bonded=False,
        )
        assert scale_factor != 1
        # The second peak shows up around r=2, should be gone
        zero_indices = np.where(rdf.bin_centers < 2.5)[0]
        zero_array = np.zeros_like(zero_indices)
        rdf_values = rdf.rdf[zero_indices]
        assert np.allclose(rdf_values, zero_array)

    def test_gsd_rdf_r_max(self, LJ_gsd):
        """Test 2 RDFs with different r_cuts. The values of the shared r_cut region should be very close"""
        rdf, scale_factor = gsd_rdf(
            gsdfile=LJ_gsd,
            start=0,
            stop=10,
            r_max=3,
            exclude_bond_depth=0,
            exclude_all_bonded=False,
        )
        rdf2, scale_factor2 = gsd_rdf(
            gsdfile=LJ_gsd,
            start=0,
            stop=10,
            r_max=4,
            exclude_bond_depth=0,
            exclude_all_bonded=False,
        )

        check_indices = np.where(rdf.bin_centers < 3)[0]
        check_indices2 = np.where(rdf2.bin_centers < 3)[0]
        rdf_integral = trapezoid(
            rdf.rdf[check_indices], rdf.bin_centers[check_indices]
        )
        rdf2_integral = trapezoid(
            rdf2.rdf[check_indices2], rdf2.bin_centers[check_indices2]
        )
        assert np.isclose(rdf_integral, rdf2_integral, rtol=0.05)

    def test_order_parameter(self, p3ht_gsd, p3ht_cg_gsd, mapping):
        r_max = 2
        a_max = 30

        order, cl_idx = order_parameter(
            p3ht_gsd, p3ht_cg_gsd, mapping, r_max, a_max
        )

        assert np.isclose(order[0], 0.33125)
        assert len(cl_idx[0]) == 160

    def test_conc_profiel(self, slab_snapshot):
        A_indices = np.arange(20)
        B_indices = np.arange(20, 40)
        d_profile, A_count, B_count, total_count = concentration_profile(
            slab_snapshot, A_indices, B_indices, n_bins=5, box_axis=0
        )
        assert (
            len(d_profile) == len(A_count) == len(B_count) == len(total_count)
        )
        assert (A_count / total_count)[0] == 1
        assert (A_count / total_count)[-1] == 0
        assert (B_count / total_count)[-1] == 1
        assert (B_count / total_count)[0] == 0

    def test_strides(self, gsdfile_bond):
        bonds = bond_distribution(
            gsdfile_bond, "A", "B", histogram=False, stride=2
        )
        assert len(bonds) == 10
        gsd_rdf(gsdfile_bond, "A", "B", stride=5)
        structure_factor(gsdfile_bond, k_min=0.2, k_max=5, stride=2)
