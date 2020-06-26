from os import path

import gsd
import gsd.hoomd
import numpy as np

from cme_lab_utils import gsd_utils

def create_frame(i):
    s = gsd.hoomd.Snapshot()
    s.configuration.step = i
    s.particles.N = 4
    s.particles.types = ['A', 'B']
    s.particles.typeid = [0,0,1,1]
    s.particles.position = np.random.random(size=(4,3))
    s.configuration.box = [3, 3, 3, 0, 0, 0]
    return s

def create_gsd():
    f = gsd.hoomd.open(name='test.gsd', mode='wb')
    f.extend((create_frame(i) for i in range(10)))

def test_frame_get_type_position():
    if not path.exists('test.gsd'):
        create_gsd()
    pos_array = gsd_utils.frame_get_type_position('test.gsd', 'A')
    assert type(pos_array) is type(np.array([]))
    assert pos_array.shape == (2,3)

