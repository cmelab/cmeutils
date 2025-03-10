from warnings import warn

import freud
import gsd
import gsd.hoomd
import numpy as np
import scipy
import unyt as u

from cmeutils import gsd_utils


def tensile_test(
    gsd_file,
    tensile_axis,
    ref_energy=None,
    ref_distance=None,
    bootstrap_sampling=False,
):
    if ref_energy or ref_distance:
        if not all([ref_energy, ref_distance]):
            raise RuntimeError(
                "Both ref_energy and ref_distnace must be defined."
            )
        if not (
            isinstance(ref_energy, u.array.unyt_quantity)
            and isinstance(ref_distance, u.array.unyt_quantity)
        ):
            raise ValueError(
                "ref_energy and ref_distance should be given as "
                "unyt.array.unyt_quantity."
            )
        # Units of Pa
        conv_factor = ref_energy.to("J/mol") / (
            ref_distance.to("m") ** 3 * u.Avogadros_number_mks
        )
        # Units of MPa
        conv_factor *= 1e-6
    else:
        conv_factor = 1

    # Get initial box info and initial stress average
    tensor_index_map = {0: 0, 1: 3, 2: 5}
    with gsd.hoomd.open(gsd_file) as traj:
        n_frames = len(traj)
        init_snap = traj[0]
        init_length = init_snap.configuration.box[tensile_axis]
        # Store relevant stress tensor value for each frame
        frame_stress_data = np.zeros(n_frames)
        frame_box_data = np.zeros(n_frames)
        for idx, snap in enumerate(traj):
            frame_stress_data[idx] = snap.log[
                "md/compute/ThermodynamicQuantities/pressure_tensor"
            ][tensor_index_map[tensile_axis]]
            frame_box_data[idx] = snap.configuration.box[tensile_axis]

    # Perform stress sampling
    box_lengths = np.unique(frame_box_data)
    strain = np.zeros_like(box_lengths, dtype=float)
    window_means = np.zeros_like(box_lengths, dtype=float)
    window_stds = np.zeros_like(box_lengths, dtype=float)
    window_sems = np.zeros_like(box_lengths, dtype=float)
    for idx, box_length in enumerate(box_lengths):
        strain[idx] = (box_length - init_length) / init_length
        indices = np.where(frame_box_data == box_length)[0]
        stress = frame_stress_data[indices] * conv_factor
        if bootstrap_sampling:
            n_data_points = len(stress)
            n_samples = 5
            window_size = n_data_points // 5
            bootstrap_means = []
            for i in range(n_samples):
                start = np.random.randint(
                    low=0, high=(n_data_points - window_size)
                )
                window_sample = stress[
                    start : start + window_size  # noqa: E203
                ]
                bootstrap_means.append(np.mean(window_sample))
            avg_stress = np.mean(bootstrap_means)
            std_stress = np.std(bootstrap_means)
            sem_stress = scipy.stats.sem(bootstrap_means)
        else:  # Use use the last half of the stress values
            cut = -len(stress) // 2
            avg_stress = np.mean(stress[cut:])
            std_stress = np.std(stress[cut:])
            sem_stress = scipy.stats.sem(stress[cut:])

        window_means[idx] = avg_stress
        window_stds[idx] = std_stress
        window_sems[idx] = sem_stress

    return strain, -window_means, window_stds, window_sems


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
