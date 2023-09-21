from os import path

import gsd.hoomd
import numpy as np
import pytest
from pymbar.testsystems import correlated_timeseries_example

from cmeutils.visualize import FresnelGSD

asset_dir = path.join(path.dirname(__file__), "assets")


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

    @pytest.fixture
    def snap_bond(self, gsdfile_bond):
        with gsd.hoomd.open(name=gsdfile_bond, mode="rb") as f:
            snap = f[-1]
        return snap

    @pytest.fixture
    def butane_gsd(self):
        return path.join(asset_dir, "butanes.gsd")

    @pytest.fixture
    def p3ht_gsd(self):
        return path.join(asset_dir, "p3ht.gsd")

    @pytest.fixture
    def p3ht_cg_gsd(self):
        return path.join(asset_dir, "p3ht-cg.gsd")

    @pytest.fixture
    def mapping(self):
        return np.loadtxt(path.join(asset_dir, "mapping.txt"), dtype=int)

    @pytest.fixture
    def p3ht_xml(self):
        return path.join(asset_dir, "p3ht.xml")

    @pytest.fixture(scope="session")
    def correlated_data_tau100_n10000(self):
        return correlated_timeseries_example(N=10000, tau=100, seed=432)

    @pytest.fixture
    def p3ht_fresnel(self):
        return FresnelGSD(gsd_file=path.join(asset_dir, "p3ht.gsd"))


def create_frame(i, add_bonds, images, seed=42):
    np.random.seed(seed)
    s = gsd.hoomd.Frame()
    s.configuration.step = i
    s.particles.N = 5
    s.particles.types = ["A", "B"]
    s.particles.typeid = [0, 0, 1, 1, 1]
    s.particles.position = np.random.random(size=(5, 3))
    s.configuration.box = [3, 3, 3, 0, 0, 0]
    if add_bonds:
        s.bonds.N = 3
        s.bonds.types = ["AB", "BB"]
        s.bonds.typeid = [0, 0, 1]
        s.bonds.group = [[0, 2], [1, 3], [3, 4]]
    if images:
        s.particles.image = np.full(shape=(5, 3), fill_value=i)
    else:
        s.particles.image = np.zeros(shape=(5, 3))
    s.validate()
    return s


def create_gsd(filename, add_bonds=False, images=False):
    with gsd.hoomd.open(name=filename, mode="wb") as f:
        f.extend(
            [
                create_frame(i, add_bonds=add_bonds, images=images)
                for i in range(10)
            ]
        )
