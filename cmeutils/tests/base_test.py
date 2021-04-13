import pytest
import tempfile

import gsd
import gsd.hoomd
import numpy as np


class BaseTest:
    @pytest.fixture
    def test_gsd(self, tmp_path):
        filename = tmp_path / "test.gsd"
        create_gsd(filename)
        return filename



def create_frame(i, seed=42):
    np.random.seed(seed)
    s = gsd.hoomd.Snapshot()
    s.configuration.step = i
    s.particles.N = 4
    s.particles.types = ['A', 'B']
    s.particles.typeid = [0,0,1,1]
    s.particles.position = np.random.random(size=(4,3))
    s.configuration.box = [3, 3, 3, 0, 0, 0]
    return s

def create_gsd(filename):
    with gsd.hoomd.open(name=filename, mode='wb') as f:
        f.extend((create_frame(i) for i in range(10)))
