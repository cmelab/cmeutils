import freud
import gsd.hoomd
import mbuild as mb
import numpy as np
import packaging.version
import pytest
from base_test import BaseTest
from gmso.external import from_mbuild, to_gsd_snapshot
from mbuild.formats.hoomd_forcefield import to_hoomdsnapshot

from cmeutils.gsd_utils import (
    _validate_inputs,
    create_rigid_snapshot,
    ellipsoid_gsd,
    frame_to_freud_system,
    get_all_types,
    get_molecule_cluster,
    get_type_position,
    identify_snapshot_connections,
    snap_delete_types,
    update_rigid_snapshot,
    xml_to_gsd,
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
    def test_frame_to_freud_system(self, butane_gsd):
        with gsd.hoomd.open(butane_gsd) as traj:
            frame = traj[0]
        freud_sys = frame_to_freud_system(frame)
        assert isinstance(freud_sys, freud.locality.NeighborQuery)

    def test_ellipsoid_gsd(self, butane_gsd):
        ellipsoid_gsd(butane_gsd, "ellipsoid.gsd", 0.5, 1.0)
        with gsd.hoomd.open(name="ellipsoid.gsd", mode="r") as f:
            snap = f[-1]
        assert snap.particles.type_shapes[0]["type"] == "Ellipsoid"

    def test_create_rigid_snapshot(self):
        benzene = mb.load("c1ccccc1", smiles=True)
        benzene.name = "Benzene"
        box = mb.fill_box(benzene, 5, box=[1, 1, 1])
        box.label_rigid_bodies(discrete_bodies="Benzene")

        rigid_init_snap = create_rigid_snapshot(box)
        assert rigid_init_snap.particles.N == 5
        assert rigid_init_snap.particles.types == ["R"]

    def test_update_rigid_snapshot(self):
        benzene = mb.load("c1ccccc1", smiles=True)
        benzene.name = "Benzene"
        box = mb.fill_box(benzene, 5, box=[1, 1, 1])
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
        assert pos_array.shape == (5, 3)

    def test_get_position_and_images(self, gsdfile_images):
        pos, imgs = get_type_position(
            gsd_file=gsdfile_images, typename="A", images=True
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

    @pytest.mark.skip(reason="HOOMD2 required for testing")
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

    def test_identify_snapshot_connections_benzene(self):
        benzene = mb.load("c1ccccc1", smiles=True)
        topology = from_mbuild(benzene)
        no_connection_snapshot, _ = to_gsd_snapshot(topology)
        assert no_connection_snapshot.bonds.N == 12
        assert no_connection_snapshot.angles.N == 0
        assert no_connection_snapshot.dihedrals.N == 0
        updated_snapshot = identify_snapshot_connections(no_connection_snapshot)

        topology.identify_connections()
        topology_snapshot, _ = to_gsd_snapshot(topology)
        assert updated_snapshot.angles.N == topology_snapshot.angles.N
        assert np.array_equal(
            sorted(
                updated_snapshot.angles.group,
                key=lambda angle: (
                    angle[1],
                    angle[0],
                    angle[2],
                ),
            ),
            sorted(
                topology_snapshot.angles.group,
                key=lambda angle: (
                    angle[1],
                    angle[0],
                    angle[2],
                ),
            ),
        )
        assert sorted(updated_snapshot.angles.types) == sorted(
            topology_snapshot.angles.types
        )
        assert len(updated_snapshot.angles.typeid) == len(
            topology_snapshot.angles.typeid
        )
        assert updated_snapshot.dihedrals.N == topology_snapshot.dihedrals.N
        assert np.array_equal(
            sorted(
                updated_snapshot.dihedrals.group,
                key=lambda angle: (
                    angle[1],
                    angle[0],
                    angle[2],
                ),
            ),
            sorted(
                topology_snapshot.dihedrals.group,
                key=lambda angle: (
                    angle[1],
                    angle[0],
                    angle[2],
                ),
            ),
        )
        assert sorted(updated_snapshot.dihedrals.types) == sorted(
            topology_snapshot.dihedrals.types
        )
        assert len(updated_snapshot.dihedrals.typeid) == len(
            topology_snapshot.dihedrals.typeid
        )

    def test_identify_connection_thiophene(self):
        thiophene = mb.load("c1cscc1", smiles=True)
        topology = from_mbuild(thiophene)
        no_connection_snapshot, _ = to_gsd_snapshot(topology)
        updated_snapshot = identify_snapshot_connections(no_connection_snapshot)
        assert updated_snapshot.angles.N == 13
        assert sorted(updated_snapshot.angles.types) == sorted(
            ["C-S-C", "H-C-S", "C-C-H", "C-C-S", "C-C-C"]
        )

        assert updated_snapshot.dihedrals.N == 16
        assert sorted(updated_snapshot.dihedrals.types) == sorted(
            [
                "C-C-C-H",
                "C-C-C-C",
                "H-C-C-H",
                "H-C-S-C",
                "H-C-C-S",
                "C-C-S-C",
                "C-C-C-S",
            ]
        )

    def test_identify_connection_no_dihedrals(self):
        methane = mb.load("C", smiles=True)
        topology = from_mbuild(methane)
        no_connection_snapshot, _ = to_gsd_snapshot(topology)
        assert no_connection_snapshot.bonds.N != 0
        assert no_connection_snapshot.angles.N == 0
        assert no_connection_snapshot.dihedrals.N == 0
        updated_snapshot = identify_snapshot_connections(no_connection_snapshot)
        assert updated_snapshot.angles.N == 6
        assert updated_snapshot.angles.types == ["H-C-H"]
        assert updated_snapshot.angles.typeid == [0, 0, 0, 0, 0, 0]
        assert updated_snapshot.dihedrals.N == 0
        assert updated_snapshot.dihedrals.types is None
        assert updated_snapshot.dihedrals.typeid is None

    def test_identify_connection_no_connections(self):
        snapshot = gsd.hoomd.Frame()
        snapshot.particles.N = 2
        snapshot.particles.types = ["A", "B"]
        snapshot.particles.typeid = [0, 1]
        with pytest.warns(UserWarning):
            updated_snapshot = identify_snapshot_connections(snapshot)
            assert updated_snapshot.bonds.N == 0
            assert updated_snapshot.angles.N == 0
            assert updated_snapshot.dihedrals.N == 0

    def test_identify_connections_pekk_cg(self, pekk_cg_gsd):
        with gsd.hoomd.open(pekk_cg_gsd) as traj:
            snap = traj[0]
            assert snap.angles.types == [] 
            snap_with_connections = identify_snapshot_connections(snap)
            assert "K-E-K" in snap_with_connections.angles.types
            assert "E-K-K" in snap_with_connections.angles.types
            assert "K-E-K-K" in snap_with_connections.dihedrals.types
            assert "E-K-K-E" in snap_with_connections.dihedrals.types
