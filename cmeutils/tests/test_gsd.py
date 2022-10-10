import pytest

import gsd.hoomd
import mbuild as mb
from mbuild.formats.hoomd_forcefield import to_hoomdsnapshot
import numpy as np
import packaging.version

from base_test import BaseTest
from cmeutils.gsd_utils import (
    get_type_position, get_molecule_cluster, get_all_types, _validate_inputs,
    snap_delete_types, xml_to_gsd, create_rigid_snapshot, update_rigid_snapshot
)

try:
    import hoomd
    if "version" in dir(hoomd):
        hoomd_version = packaging.version.parse(hoomd.version.version)
    else:
        hoomd_version = packaging.version.parse(hoomd.__version__)
    has_hoomd = True
except ImportError:
    has_hoomd = False


class TestGSD(BaseTest):
    def test_ellipsoid_gsd(self, butane_gsd):
        new_gsd = ellipsoid(gsd(butane_gsd, "ellipsoid.gsd", 0.5, 1.0)

    def test_create_rigid_snapshot(self):
        benzene = mb.load("c1ccccc1", smiles=True)
        benzene.name = "Benzene"
        box = mb.fill_box(benzene, 5, box=[1,1,1])
        box.label_rigid_bodies(discrete_bodies="Benzene")

        rigid_init_snap = create_rigid_snapshot(box)
        assert rigid_init_snap.particles.N == 5
        assert rigid_init_snap.particles.types == ["R"]

    def test_update_rigid_snapshot(self):
        benzene = mb.load("c1ccccc1", smiles=True)
        benzene.name = "Benzene"
        box = mb.fill_box(benzene, 5, box=[1,1,1])
        box.label_rigid_bodies(discrete_bodies="Benzene")

        rigid_init_snap = create_rigid_snapshot(box)
        _snapshot, refs = to_hoomdsnapshot(box, hoomd_snapshot=rigid_init_snap)
        snap, rigid_obj = update_rigid_snapshot(_snapshot, box)
        assert snap.particles.N == 65
        assert np.array_equal(snap.particles.typeid[0:5], np.array([0] * 5))

    def test_get_type_position(self, gsdfile):
        pos_array = get_type_position(gsd_file=gsdfile, typename="A")
        assert type(pos_array) is type(np.array([]))
        assert pos_array.shape == (2, 3)

    def test_get_multiple_types(self, gsdfile):
        pos_array = get_type_position(gsd_file=gsdfile, typename=["A", "B"])
        assert type(pos_array) is type(np.array([]))
        assert pos_array.shape == (5,3)

    def test_get_position_and_images(self, gsdfile_images):
        pos, imgs = get_type_position(
                gsd_file=gsdfile_images,
                typename="A",
                images=True
                )
        assert type(pos) is type(imgs) is type(np.array([]))
        assert pos.shape == imgs.shape

    def test_validate_inputs(self, gsdfile, snap):
        # Catch errors when both gsd_file and snap are passed
        with pytest.raises(ValueError):
            _validate_inputs(gsdfile, snap, 1)
        with pytest.raises(AssertionError):
            _validate_inputs(gsdfile, None, 1.0)
        with pytest.raises(AssertionError):
            _validate_inputs(None, gsdfile, 1)
        with pytest.raises(OSError):
            _validate_inputs("bad_gsd_file", None, 0)

    def test_get_all_types(self, gsdfile):
        types = get_all_types(gsdfile)
        assert types == ["A", "B"]

    def test_get_molecule_cluster(self, gsdfile_bond):
        cluster = get_molecule_cluster(gsd_file=gsdfile_bond)
        assert np.array_equal(cluster, [1, 0, 1, 0, 0])

    def test_snap_delete_types(self, snap):
        new_snap = snap_delete_types(snap, "A")
        assert "A" not in new_snap.particles.types

    def test_snap_delete_types_bonded(self, snap_bond):
        new_snap = snap_delete_types(snap_bond, "A")
        assert "A" not in new_snap.particles.types

    @pytest.mark.skipif(
        not has_hoomd or hoomd_version.major != 2,
        reason="HOOMD is not installed or is wrong version"
    )
    def test_xml_to_gsd(self, tmp_path, p3ht_gsd, p3ht_xml):
        new_gsd = tmp_path / "new.gsd"
        xml_to_gsd(p3ht_xml, new_gsd)
        with gsd.hoomd.open(p3ht_gsd) as old, gsd.hoomd.open(new_gsd) as new:
            old_snap = old[-1]
            new_snap = new[-1]
        assert np.all(
            old_snap.particles.position == new_snap.particles.position
        )
        assert np.all(old_snap.particles.image == new_snap.particles.image)

