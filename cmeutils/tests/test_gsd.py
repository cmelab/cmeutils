import pytest

import gsd.hoomd
import numpy as np

from base_test import BaseTest
from cmeutils.gsd_utils import (
    get_type_position, get_molecule_cluster, get_all_types, _validate_inputs,
    snap_delete_types, xml_to_gsd
)


class TestGSD(BaseTest):
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

    @pytest.mark.skip(reason="Requires that hoomd2 is installed.")
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

