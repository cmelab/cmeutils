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
        assert(
                trajectory[start].configuration.box == 
                trajectory[stop].configuration.box
                ), f"The box is not consistent over the range{start}:{stop}"

        frame_positions = []
        for frame in trajectory[start:stop]:
            if atom_type == "all":
                atom_pos = frame.particles.position[:]
            else:
                atom_pos = gsd_utils.get_type_position(atom_types, snap=frame)
            frame_positions.append(atom_pos)

        msd = freud.msd.MSD(box=trajectory.configuration.box, mode=msd_mode)
        msd.compute(positions)
    return msd.msd
 
