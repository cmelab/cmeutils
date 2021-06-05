import gsd
import gsd.hoomd
import hoomd
import numpy as np
import re

def write_snapshot(beads):
    """
    beads : iterable, required
        An iterable of any of the structure classes in cmeutils polymer.py
        If you want to coarse-grain based on monomers, pass in a list of
        System's monomers.
    """
    all_types = []
    all_pairs = []
    pair_groups = []
    all_angles = []
    angle_groups = []
    all_pos = []
    box = None

    for idx, bead in enumerate(beads):
        if box is None:
            box = bead.system.box
        all_types.append(bead.name)
        all_pos.append(bead.unwrapped_center)
        try:
            if bead.parent == beads[idx+1].parent:
                pair = sorted([bead.name, beads[idx+1].name])
                pair_type = "-".join((pair[0], pair[1]))
                all_pairs.append(pair_type)
                pair_groups.append([idx, idx+1])

                if bead.parent == beads[idx+2].parent:
                    b1, b2, b3 = bead.name, beads[idx+1].name, beads[idx+2].name
                    b1, b3 = sorted([b1, b3], key=_natural_sort)
                    angle_type = "-".join((b1, b2, b3))
                    all_angles.append(angle_type)
                    angle_groups.append([idx, idx+1, idx+2])
        except IndexError:
            pass

    types = list(set(all_types)) 
    pairs = list(set(all_pairs)) 
    angles = list(set(all_angles))
    type_ids = [np.where(np.array(types)==i)[0][0] for i in all_types]
    pair_ids = [np.where(np.array(pairs)==i)[0][0] for i in all_pairs]
    angle_ids = [np.where(np.array(angles)==i)[0][0] for i in all_angles]

    #Wrap the particle positions
    _box = hoomd.data.boxdim(
            Lx=box[0],
            Ly=box[1],
            Lz=box[2]
            )
    w_positions = np.stack([_box.wrap(xyz)[0] for xyz in all_pos])
    w_images = np.stack([_box.wrap(xyz)[1] for xyz in all_pos])

    s = gsd.hoomd.Snapshot()
    #Particles
    s.particles.N = len(all_types)
    s.particles.types = types 
    s.particles.typeid = np.array(type_ids) 
    s.particles.position = w_positions
    s.particles.image = w_images
    #Bonds
    s.bonds.N = len(all_pairs)
    s.bonds.M = 2
    s.bonds.types = pairs
    s.bonds.typeid = np.array(pair_ids)
    s.bonds.group = np.vstack(pair_groups)
    #Angles
    s.angles.N = len(all_angles)
    s.angles.M = 3
    s.angles.types = angles
    s.angles.typeid = np.array(angle_ids)
    s.angles.group = np.vstack(angle_groups)
    s.configuration.box = box 
    return s

def _atoi(text):
    return int(text) if text.isdigit() else text

def _natural_sort(text):
    """Break apart a string containing letters and digits."""
    return [_atoi(a) for a in re.split(r"(\d+)", text)]

