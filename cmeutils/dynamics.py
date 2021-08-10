import freud
import gsd
import gsd.hoomd
import numpy as np
from warnings import warn

from cmeutils import gsd_utils

def msd_from_gsd(
        gsdfile,
        atom_types="all",
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
        images = []
        for frame in trajectory[start:stop]:
            if atom_types == "all":
                atom_pos = frame.particles.position[:]
                atom_img = frame.particles.image[:]
            else:
                atom_pos, atom_img  = gsd_utils.get_type_position(
                            atom_types,
                            snap=frame,
                            images=True
                )
            positions.append(atom_pos)
            images.append(atom_img)
        if np.count_nonzero(np.array(images)) == 0:
            warn(
                f"All of the images over the range {start}-{stop} "
                "are [0,0,0]. You may want to ensure this gsd file "
                "had the particle images written to it."
                )
        msd = freud.msd.MSD(box=init_box, mode=msd_mode)
        msd.compute(np.array(positions), np.array(images), reset=False)
    return msd
 