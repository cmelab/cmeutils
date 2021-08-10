import pytest
import tempfile

import gsd
import gsd.hoomd
import numpy as np


class BaseTest:
    @pytest.fixture
    def gsdfile(self, tmp_path):
        filename = tmp_path / "test.gsd"
        create_gsd(filename)
        return filename

    @pytest.fixture
    def gsdfile_bond(self, tmp_path):
        filename = tmp_path / "test_bond.gsd"
        create_gsd(filename, add_bonds=True)
        return filename

    @pytest.fixture
    def gsdfile_images(self, tmp_path):
        filename = tmp_path / "test_images.gsd"
        create_gsd(filename, images=True)
        return filename

    @pytest.fixture
    def snap(self, gsdfile):
        with gsd.hoomd.open(name=gsdfile, mode="rb") as f:
            snap = f[-1]
        return snap


def create_frame(i, add_bonds, images, seed=42):
    np.random.seed(seed)
    s = gsd.hoomd.Snapshot()
    s.configuration.step = i
    s.particles.N = 5
    s.particles.types = ["A", "B"]
    s.particles.typeid = [0, 0, 1, 1, 1]
    s.particles.position = np.random.random(size=(5, 3))
    s.configuration.box = [3, 3, 3, 0, 0, 0]
    if add_bonds:
        s.bonds.N = 2
        s.bonds.types = ["AB"]
        s.bonds.typeid = [0, 0]
        s.bonds.group = [[0, 2], [1, 3]]
    if images:
        s.particles.image = np.full(shape=(5,3), fill_value=i)
    else:
        s.particles.image = np.zeros(shape=(5,3))
    s.validate()
    return s


def create_gsd(filename, add_bonds=False, images=False):
    with gsd.hoomd.open(name=filename, mode="wb") as f:
        f.extend((create_frame(i, add_bonds=add_bonds, images=images) for i in range(10)))
