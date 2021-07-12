import freud
import gsd
import gsd.hoomd
import numpy as np

from cmeutils import gsd_utils

def msd_from_gsd(
        gsdfile,
        atom_types,
        start=0,
        stop=None,
        msd_mode="window"
        ):
    """
    """
    if stop is None:
        stop = -1

    with gsd.hoomd.open(gsdfile, "rb") as trajectory:
        init_box = trajectory[start].configuration.box
        final_box = trajectory[stop].configuration.box
        assert all(
                [i == j for i,j in zip(init_box, final_box)]
                ), f"The box is not consistent over the range {start}:{stop}"

        positions = []
        for frame in trajectory[start:stop]:
            if atom_types == "all":
                atom_pos = frame.particles.position[:]
            else:
                atom_pos = gsd_utils.get_type_position(atom_types, snap=frame)
            positions.append(atom_pos)
        msd = freud.msd.MSD(box=init_box, mode=msd_mode)
        msd.compute(np.array(positions), reset=False)
    return msd
 
