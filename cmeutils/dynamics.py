from warnings import warn

import freud
import gsd
import gsd.hoomd
import numpy as np

from cmeutils import gsd_utils


def tensile_test(gsd_file, period, tensile_axis, ref_energy=1, ref_distance=1):
    # Get initial box info and initial stress average
    tensor_index_map = {0: 0, 1: 3, 2: 5}
    with gsd.hoomd.open(gsd_file) as traj:
        n_frames = len(traj)
        # init_snap = traj[0]
        # init_length = init_snap.configuration.box[tensile_axis]
        # Store relevant stress tensor value for each frame
        frame_stress_data = np.zeros(n_frames)
        for idx, snap in enumerate(traj):
            frame_stress_data[idx] = snap.log[
                "md/compute/ThermodynamicQuantities/pressure_tensor"
            ][tensor_index_map[tensile_axis]]

    # window_means = np.zeros(n_frames // period)
    # window_stds = np.zeros(n_frames // period)


def msd_from_gsd(
    gsdfile, atom_types="all", start=0, stop=-1, msd_mode="window"
):
    """Calculate the mean-square displacement (MSD) of the particles in a
    trajectory using Freud.

    Parameters
    ----------
    gsdfile : str
        Filename of the GSD trajectory
    atom_types : str, or list of str
        Name(s) of particles to use in calcualtion of the MSD
    start : int
        The first frame from the gsd file to use
        (default 0)
    stop : int
        The last frame from the gsd file to use
        (default -1)
    msd_mode : str
        Choose from "window" or "direct". See Freud for the differences
        https://freud.readthedocs.io/en/latest/modules/msd.html#freud.msd.MSD
    """
    with gsd.hoomd.open(gsdfile, "r") as trajectory:
        init_box = trajectory[start].configuration.box
        final_box = trajectory[stop].configuration.box
        assert all(
            [i == j for i, j in zip(init_box, final_box)]
        ), f"The box is not consistent over the range {start}:{stop}"

        positions = []
        images = []
        for frame in trajectory[start:stop]:
            if atom_types == "all":
                atom_pos = frame.particles.position[:]
                atom_img = frame.particles.image[:]
            else:
                atom_pos, atom_img = gsd_utils.get_type_position(
                    atom_types, snap=frame, images=True
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
