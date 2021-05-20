import gsd
import gsd.hoomd
import numpy as np

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
        all_pos.append(bead.center)
        try:
            if bead.parent == beads[idx+1].parent:
                pair = sorted([bead.name, beads[idx+1].name])
                all_pairs.append(f"{pair[0]}-{pair[1]}")
                pair_groups.append([idx, idx+1])

                if bead.parent == beads[idx+2].parent:
                    angle = [bead.name, beads[idx+1].name, beads[idx+2].name]
                    all_angles.append(
                            f"{angle[0]}-{angle[1]}-{angle[2]}"
                            )
                    angle_groups.append([idx, idx+1, idx+2])
        except IndexError:
            pass

    types = list(set(all_types)) 
    pairs = list(set(all_pairs)) 
    angles = list(set(all_angles))
    type_ids = [np.where(np.array(types)==i)[0][0] for i in all_types]
    pair_ids = [np.where(np.array(pairs)==i)[0][0] for i in all_pairs]
    angle_ids = [np.where(np.array(angles)==i)[0][0] for i in all_angles]

    s = gsd.hoomd.Snapshot()
    #Particles
    s.particles.N = len(all_types)
    s.particles.types = types 
    s.particles.typeid = np.array(type_ids) 
    s.particles.position = np.array(all_pos)
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

