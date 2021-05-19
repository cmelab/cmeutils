import gsd
import gsd.hoomd
import numpy as np

def write_snapshot(system, beads):
    """
    """
    all_types = []
    all_pos = []
    if beads == "monomers":
        n_beads = system.n_monomers
        for mol in system.molecules:
            for mon in mol.monomers:
                all_types.append(mon.name)
                all_pos.append(mon.center)
    if beads == "segments":
        pass
    if beads == "components":
        pass

    types = list(set(all_types)) 
    s = gsd.hoomd.Snapshot()
    s.particles.N = n_beads
    s.particles.types = types 
    s.particles.typeids = [
            np.where(np.array(types)==i)[0][0] for i in all_types
        ]
    s.particles.position = all_pos
    s.configuration.box = system.box
    return s

