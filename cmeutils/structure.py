import warnings

import freud
import gsd
import gsd.hoomd
import numpy as np
from rowan import vector_vector_rotation

from cmeutils.geometry import (
    angle_between_vectors,
    dihedral_angle,
    get_plane_normal,
)
from cmeutils.gsd_utils import frame_to_freud_system, get_molecule_cluster
from cmeutils.plotting import get_histogram


def angle_distribution(
    gsd_file,
    A_name,
    B_name,
    C_name,
    start=0,
    stop=-1,
    degrees=False,
    histogram=False,
    theta_min=0.0,
    theta_max=None,
    normalize=False,
    bins="auto",
):
    """Returns the bond angle distribution for a given triplet of particles

    Parameters
    ----------
    gsdfile : str
        Filename of the GSD trajectory.
    A_name, B_name, C_name : str
        Name(s) of particles that form the angle triplet
        (found in gsd.hoomd.Frame.particles.types)
        They must be given in the same order as they form the angle
    start : int
        Starting frame index for accumulating bond lengths.
        Negative numbers index from the end. (default 0)
    stop : int
        Final frame index for accumulating bond lengths. (default -1)
    degrees : bool, default=False
        If True, the angle values are returned in degrees.
        if False, the angle values are returned in radians.
    histogram : bool, default=False
        If set to True, places the resulting angles into a histogram
        and retrums the histogram's bin centers and heights as
        opposed to the actual calcualted angles.
    theta_min : float, default = 0.0
        Sets the minimum theta value to be included in the distribution
    theta_max : float, default = None
        Sets the maximum theta value to be included in the distribution
        If left as None, then theta_max will be either pi radians or
        180 degrees depending on the value set for the degrees parameter
    normalize : bool, default=False
        If set to True, normalizes the angle distribution by the
        sum of the bin heights, so that the distribution adds up to 1.
    bins : float, int, or str,  default="auto"
        The number of bins to use when finding the distribution
        of bond angles. Using "auto" will set the number of
        bins based on the ideal bin size for the data.
        See the numpy.histogram docs for more details.

    Returns
    -------
    1-D numpy.array  or 2-D numpy.array
        If histogram is False, Array of actual bond angles in degrees
        If histogram is True, returns a 2D array of bin centers and bin heights.

    """
    if not degrees and theta_max is None:
        theta_max = np.pi
    elif degrees and theta_max is None:
        theta_max = 180

    trajectory = gsd.hoomd.open(gsd_file, mode="r")
    name = "-".join([A_name, B_name, C_name])
    name_rev = "-".join([C_name, B_name, A_name])

    angles = []
    for snap in trajectory[start:stop]:
        if name not in snap.angles.types and name_rev not in snap.angles.types:
            raise ValueError(
                f"Angles {name} or {name_rev} not found in "
                " snap.angles.types. "
                "A_name, B_name, C_name must match the order "
                "as they appear in snap.angles.types."
            )
        for idx, angle_id in enumerate(snap.angles.typeid):
            angle_name = snap.angles.types[angle_id]
            if angle_name == name or angle_name == name_rev:
                pos1 = snap.particles.position[snap.angles.group[idx][0]]
                img1 = snap.particles.image[snap.angles.group[idx][0]]
                pos2 = snap.particles.position[snap.angles.group[idx][1]]
                img2 = snap.particles.image[snap.angles.group[idx][1]]
                pos3 = snap.particles.position[snap.angles.group[idx][2]]
                img3 = snap.particles.image[snap.angles.group[idx][2]]
                pos1_unwrap = pos1 + (img1 * snap.configuration.box[:3])
                pos2_unwrap = pos2 + (img2 * snap.configuration.box[:3])
                pos3_unwrap = pos3 + (img3 * snap.configuration.box[:3])
                u = pos1_unwrap - pos2_unwrap
                v = pos3_unwrap - pos2_unwrap
                angles.append(
                    np.round(angle_between_vectors(u, v, False, degrees), 3)
                )
    trajectory.close()

    if histogram:
        if min(angles) < theta_min or max(angles) > theta_max:
            warnings.warn(
                "There are bond angles that fall outside of "
                "your set theta_min and theta_max range. "
                "You may want to adjust this range to "
                "include all bond angles."
            )
        bin_centers, bin_heights = get_histogram(
            data=np.array(angles),
            normalize=normalize,
            bins=bins,
            x_range=(theta_min, theta_max),
        )
        return np.stack((bin_centers, bin_heights)).T
    else:
        return np.array(angles)


def bond_distribution(
    gsd_file,
    A_name,
    B_name,
    start=0,
    stop=-1,
    histogram=False,
    l_min=0.0,
    l_max=4.0,
    normalize=True,
    bins=100,
):
    """Returns the bond length distribution for a given bond pair

    Parameters
    ----------
    gsdfile : str
        Filename of the GSD trajectory.
    A_name, B_name : str
        Name(s) of particles that form the bond pair
        (found in gsd.hoomd.Frame.particles.types)
    start : int
        Starting frame index for accumulating bond lengths.
        Negative numbers index from the end. (default 0)
    stop : int
        Final frame index for accumulating bond lengths. (default -1)
    histogram : bool, default=False
        If set to True, places the resulting bonds into a histogram
        and retrums the histogram's bin centers and heights as
        opposed to the actual calcualted bonds.
    l_min : float, default = 0.0
        Sets the minimum bond length to be included in the distribution
    l_max : float, default = 5.0
        Sets the maximum bond length value to be included in the distribution
    normalize : bool, default=False
        If set to True, normalizes the angle distribution by the
        sum of the bin heights, so that the distribution adds up to 1.
    bins : float, int, or str,  default="auto"
        The number of bins to use when finding the distribution
        of bond angles. Using "auto" will set the number of
        bins based on the ideal bin size for the data.
        See the numpy.histogram docs for more details.

    Returns
    -------
    1-D numpy.array  or 2-D numpy.array
        If histogram is False, Array of actual bond angles in degrees
        If histogram is True, returns a 2D array of bin centers and bin heights.

    """
    trajectory = gsd.hoomd.open(gsd_file, mode="r")
    name = "-".join([A_name, B_name])
    name_rev = "-".join([B_name, A_name])

    bonds = []
    for snap in trajectory[start:stop]:
        if name not in snap.bonds.types and name_rev not in snap.bonds.types:
            raise ValueError(
                f"Bond types {name} or {name_rev} not found "
                "snap.bonds.types."
            )
        for idx, bond in enumerate(snap.bonds.typeid):
            bond_name = snap.bonds.types[bond]
            if bond_name in [name, name_rev]:
                pos1 = snap.particles.position[snap.bonds.group[idx][0]]
                img1 = snap.particles.image[snap.bonds.group[idx][0]]
                pos2 = snap.particles.position[snap.bonds.group[idx][1]]
                img2 = snap.particles.image[snap.bonds.group[idx][1]]
                pos1_unwrap = pos1 + (img1 * snap.configuration.box[:3])
                pos2_unwrap = pos2 + (img2 * snap.configuration.box[:3])
                bonds.append(
                    np.round(np.linalg.norm(pos2_unwrap - pos1_unwrap), 3)
                )
    trajectory.close()

    if histogram:
        if min(bonds) < l_min or max(bonds) > l_max:
            warnings.warn(
                "There are bond lengths that fall outside of "
                "your set l_min and l_max range. You may want to adjust "
                "this range to include all bond lengths."
            )
        bin_centers, bin_heights = get_histogram(
            data=np.array(bonds),
            normalize=normalize,
            bins=bins,
            x_range=(l_min, l_max),
        )
        return np.stack((bin_centers, bin_heights)).T
    else:
        return np.array(bonds)


def dihedral_distribution(
    gsd_file,
    A_name,
    B_name,
    C_name,
    D_name,
    start=0,
    stop=-1,
    degrees=False,
    histogram=False,
    normalize=False,
    bins="auto",
):
    """Returns the bond angle distribution for a given triplet of particles

    Parameters
    ----------
    gsdfile : str
        Filename of the GSD trajectory.
    A_name, B_name, C_name, D_name: str
        Name(s) of particles that form the dihedral quadruplett
        (found in gsd.hoomd.Frame.particles.types)
        They must be given in the same order as they form the dihedral
    start : int
        Starting frame index for accumulating bond lengths.
        Negative numbers index from the end. (default 0)
    stop : int
        Final frame index for accumulating bond lengths. (default -1)
    degrees : bool, default=False
        If True, the angle values are returned in degrees.
        if False, the angle values are returned in radians.
    histogram : bool, default=False
        If set to True, places the resulting angles into a histogram
        and retrums the histogram's bin centers and heights as
        opposed to the actual calcualted angles.
    normalize : bool, default=False
        If set to True, normalizes the dihedral distribution by the
        sum of the bin heights, so that the distribution adds up to 1.
    bins : float, int, or str,  default="auto"
        The number of bins to use when finding the distribution
        of bond angles. Using "auto" will set the number of
        bins based on the ideal bin size for the data.
        See the numpy.histogram docs for more details.

    Returns
    -------
    1-D numpy.array  or 2-D numpy.array
        If histogram is False, Array of actual dihedral angles
        If histogram is True, returns a 2D array of bin centers and bin heights.

    """
    trajectory = gsd.hoomd.open(gsd_file, mode="r")
    name = "-".join([A_name, B_name, C_name, D_name])
    name_rev = "-".join([D_name, C_name, B_name, A_name])

    dihedrals = []
    for snap in trajectory[start:stop]:
        if (
            name not in snap.dihedrals.types
            and name_rev not in snap.dihedrals.types
        ):
            raise ValueError(
                f"Dihedrals {name} or {name_rev} not found in "
                " snap.dihedrals.types. "
                "A_name, B_name, C_name, D_name must match the order "
                "as they appear in snap.dihedrals.types."
            )
        for idx, _id in enumerate(snap.dihedrals.typeid):
            dih_name = snap.dihedrals.types[_id]
            if dih_name == name or dih_name == name_rev:
                pos1 = snap.particles.position[snap.dihedrals.group[idx][0]]
                img1 = snap.particles.image[snap.dihedrals.group[idx][0]]
                pos2 = snap.particles.position[snap.dihedrals.group[idx][1]]
                img2 = snap.particles.image[snap.dihedrals.group[idx][1]]
                pos3 = snap.particles.position[snap.dihedrals.group[idx][2]]
                img3 = snap.particles.image[snap.dihedrals.group[idx][2]]
                pos4 = snap.particles.position[snap.dihedrals.group[idx][3]]
                img4 = snap.particles.image[snap.dihedrals.group[idx][3]]
                pos1_unwrap = pos1 + (img1 * snap.configuration.box[:3])
                pos2_unwrap = pos2 + (img2 * snap.configuration.box[:3])
                pos3_unwrap = pos3 + (img3 * snap.configuration.box[:3])
                pos4_unwrap = pos4 + (img4 * snap.configuration.box[:3])
                phi = dihedral_angle(
                    pos1_unwrap, pos2_unwrap, pos3_unwrap, pos4_unwrap
                )
                dihedrals.append(phi)
    trajectory.close()

    if histogram:
        bin_centers, bin_heights = get_histogram(
            data=np.array(dihedrals),
            normalize=normalize,
            bins=bins,
            x_range=(-np.pi, np.pi),
        )
        return np.stack((bin_centers, bin_heights)).T
    else:
        return np.array(dihedrals)


def get_quaternions(n_views=20):
    """Get the quaternions for the specified number of views.

    The first (n_view - 3) views will be the views even distributed on a sphere,
    while the last three views will be the face-on, edge-on, and corner-on
    views, respectively.

    These quaternions are useful as input to `view_orientation` kwarg in
    `freud.diffraction.Diffractometer.compute`.

    Parameters
    ----------
    n_views : int, default 20
        The number of views to compute.

    Returns
    -------
    list of numpy.ndarray
        Quaternions as (4,) arrays.
    """
    if n_views <= 3 or not isinstance(n_views, int):
        raise ValueError("Please set n_views to an integer greater than 3.")
    # Calculate points for even distribution on a sphere
    ga = np.pi * (3 - 5**0.5)
    theta = ga * np.arange(n_views - 3)
    z = np.linspace(1 - 1 / (n_views - 3), 1 / (n_views - 3), n_views - 3)
    radius = np.sqrt(1 - z * z)
    points = np.zeros((n_views, 3))
    points[:-3, 0] = radius * np.cos(theta)
    points[:-3, 1] = radius * np.sin(theta)
    points[:-3, 2] = z

    # face on
    points[-3] = np.array([0, 0, 1])
    # edge on
    points[-2] = np.array([0, 1, 1])
    # corner on
    points[-1] = np.array([1, 1, 1])

    unit_z = np.array([0, 0, 1])
    return [vector_vector_rotation(i, unit_z) for i in points]


def gsd_rdf(
    gsdfile,
    A_name,
    B_name,
    start=0,
    stop=-1,
    r_max=None,
    r_min=0,
    bins=100,
    exclude_bonded=True,
):
    """Compute intermolecular RDF from a GSD file.

    This function calculates the radial distribution function given a GSD file
    and the names of the particle types. By default it will calculate the RDF
    for the entire trajectory.

    It is assumed that the bonding, number of particles, and simulation box do
    not change during the simulation.

    Parameters
    ----------
    gsdfile : str
        Filename of the GSD trajectory.
    A_name, B_name : str
        Name(s) of particles between which to calculate the RDF (found in
        gsd.hoomd.Frame.particles.types)
    start : int
        Starting frame index for accumulating the RDF. Negative numbers index
        from the end. (default 0)
    stop : int
        Final frame index for accumulating the RDF. If None, the last frame
        will be used. (default -1)
    r_max : float
        Maximum radius of RDF. If None, half of the maximum box size is used.
        (default -1)
    r_min : float
        Minimum radius of RDF. (default 0)
    bins : int
        Number of bins to use when calculating the RDF. (default 100)
    exclude_bonded : bool
        Whether to remove particles in same molecule from the neighbor list.
        (default True)

    Returns
    -------
    (freud.density.RDF, float)
    """
    with gsd.hoomd.open(gsdfile, mode="r") as trajectory:
        snap = trajectory[0]

        if r_max is None:
            # Use a value just less than half the maximum box length.
            r_max = np.nextafter(
                np.max(snap.configuration.box[:3]) * 0.49, 0, dtype=np.float32
            )

        rdf = freud.density.RDF(bins=bins, r_max=r_max, r_min=r_min)

        type_A = snap.particles.typeid == snap.particles.types.index(A_name)
        type_B = snap.particles.typeid == snap.particles.types.index(B_name)

        if exclude_bonded:
            molecules = get_molecule_cluster(snap=snap)
            molecules_A = molecules[type_A]
            molecules_B = molecules[type_B]

        for snap in trajectory[start:stop]:
            A_pos = snap.particles.position[type_A]
            if A_name == B_name:
                B_pos = A_pos
                exclude_ii = True
                ab_ratio = 1
            else:
                B_pos = snap.particles.position[type_B]
                exclude_ii = False
                ab_ratio = len(A_pos) / len(B_pos)

            box = snap.configuration.box
            system = (box, A_pos)
            aq = freud.locality.AABBQuery.from_system(system)
            nlist = aq.query(
                B_pos, {"r_max": r_max, "exclude_ii": exclude_ii}
            ).toNeighborList()

            if exclude_bonded:
                pre_filter = len(nlist)
                nlist.filter(
                    molecules_A[nlist.point_indices]
                    != molecules_B[nlist.query_point_indices]
                )
                post_filter = len(nlist)

            rdf.compute(aq, neighbors=nlist, reset=False)

        normalization = post_filter / pre_filter if exclude_bonded else 1
        normalization *= ab_ratio

        return rdf, normalization


def structure_factor(
    gsdfile,
    k_min,
    k_max,
    start=0,
    stop=-1,
    bins=100,
    method="direct",
    ref_length=None,
):
    """Uses freud's diffraction module to find 1D structure factors.

    Parameters
    ----------
    gsdfile : str, required
        File path to the GSD trajectory.
    k_max : float, required
        Maximum value to include in the calculation.
    k_min : float, required
        Minimum value included in the calculation
    start : int, default 0
        Starting frame index for accumulating the Sq. Negative numbers index
        from the end.
    stop : int, optional default None
        Final frame index for accumulating the Sq. If None, the last frame
        will be used.
    bins : int, optional default 100
        Number of bins to use when calculating the Sq.
        Used as the `bins` parameter in `StaticStructureFactorDirect` or
        the `num_k_values` parameter in `StaticStructureFactorDebye`.
    method : str optional default "direct"
        Choose the method used by freud.
        Options are "direct" or "debye"
        See: https://freud.readthedocs.io/en/latest/modules/diffraction.html#freud.diffraction.StaticStructureFactorDirect # noqa: E501
    ref_length : float, optional, default None
        Set a reference length to convert from reduced units to real units.
        If None, uses 1 by default.

    Returns
    -------
    freud.diffraction.StaticStructureFactorDirect or
    freud.diffraction.StaticStructureFactorDebye
    """

    if not ref_length:
        ref_length = 1

    if method.lower() == "direct":
        sf = freud.diffraction.StaticStructureFactorDirect(
            bins=bins, k_max=k_max / ref_length, k_min=k_min / ref_length
        )
    elif method.lower() == "debye":
        sf = freud.diffraction.StaticStructureFactorDebye(
            num_k_values=bins,
            k_max=k_max / ref_length,
            k_min=k_min / ref_length,
        )
    else:
        raise ValueError(
            f"Optional methods are `debye` or `direct`, you chose {method}"
        )
    with gsd.hoomd.open(gsdfile, mode="r") as trajectory:
        for frame in trajectory[start:stop]:
            system = frame_to_freud_system(frame=frame, ref_length=ref_length)
            sf.compute(system=system, reset=False)
    return sf


def diffraction_pattern(
    gsdfile,
    views,
    start=0,
    stop=-1,
    ref_length=None,
    grid_size=1024,
    output_size=None,
):
    """Uses freud's diffraction module to find 2D diffraction patterns.

    Notes
    -----
    The diffraction pattern is averaged over both views and frames
    set by the length of views and the range start - stop.

    Parameters
    ----------
    gsdfile : str, required
        File path to the GSD trajectory.
    views : list, required
        List of orientations (quarternions) to average over.
        See cmeutils.structure.get_quarternions
    start : int, default 0
        Starting frame index for accumulating the Sq. Negative numbers index
        from the end.
    stop : int, optional default None
        Final frame index for accumulating the Sq. If None, the last frame
        will be used.
    ref_length : float, optional, default None
        Set a reference length to convert from reduced units to real units.
        If None, uses 1 by default.
    grid_size : unsigned int, optional, default 1024
        Resolution of the diffraction grid.
    output_size : unsigned int, optional, default None
        Resolution of the output diffraction image.

    Returns
    -------
    freud.diffraction.DiffractionPattern
    """

    if not ref_length:
        ref_length = 1
    dp = freud.diffraction.DiffractionPattern(
        grid_size=grid_size, output_size=output_size
    )
    with gsd.hoomd.open(gsdfile) as trajectory:
        for frame in trajectory[start:stop]:
            system = frame_to_freud_system(frame=frame, ref_length=ref_length)
            for view in views:
                dp.compute(system=system, view_orientation=view, reset=False)
    return dp


def get_centers(gsdfile, new_gsdfile):
    """Create a gsd file containing the molecule centers from an existing gsd
     file.


    This function calculates the centers of a trajectory given a GSD file
    and stores them into a new GSD file just for centers. By default it will
    calculate the centers of an entire trajectory.

    Parameters
    ----------
    gsdfile : str
        Filename of the GSD trajectory.
    new_gsdfile : str
        Filename of new GSD for centers.
    """
    with (
        gsd.hoomd.open(new_gsdfile, "w") as new_traj,
        gsd.hoomd.open(gsdfile, "r") as traj,
    ):
        snap = traj[0]
        cluster_idx = get_molecule_cluster(snap=snap)
        for snap in traj:
            new_snap = gsd.hoomd.Frame()
            new_snap.configuration.box = snap.configuration.box
            f_box = freud.box.Box.from_box(snap.configuration.box)
            # Use the freud box to unwrap the particle positions
            unwrapped_positions = f_box.unwrap(
                snap.particles.position, snap.particles.image
            )
            uw_centers = []
            for i in range(max(cluster_idx) + 1):
                cluster_uw_pos = unwrapped_positions[np.where(cluster_idx == i)]
                uw_centers.append(np.mean(cluster_uw_pos, axis=0))
            uw_centers = np.stack(uw_centers)
            new_snap.particles.position = f_box.wrap(uw_centers)
            new_snap.particles.N = len(uw_centers)
            new_snap.particles.types = ["A"]
            new_snap.particles.image = f_box.get_images(uw_centers)
            new_snap.particles.typeid = np.zeros(len(uw_centers))
            new_snap.validate()
            new_traj.append(new_snap)


def order_parameter(aa_gsd, cg_gsd, mapping, r_max, a_max, large=6, start=-10):
    """Calculate the order parameter of a system.

    The order parameter is used to describe the proportion of structures in
    "large" clusters. The clustering takes into account the distance between
    centers and angles between planes of structures and only considers neighbors
    within the angle and distance cutoffs. This clustering metric has been used
    in our previous work to quantify ordering in perylene
    (DOI:10.1021/acsomega.6b00371) and thiophenes in p3ht
    (DOI:10.3390/polym10121305).

    In an attempt to be more transferrable, this functions relies on GRiTS
    (https://github.com/cmelab/grits) to handle the finding and mapping of
    structures within the atomistic system.

    Example::

        from grits import CG_System
        from grits.utils import amber_dict

        gsdfile = "trajectory.gsd"
        cg_gsdfile = "cg-trajectory.gsd"
        system = CG_System(
            gsdfile,
            beads={"_B" : "c1cscc1"},
            conversion_dict=amber_dict
        )
        mapping = system.mapping["_B...c1cscc1"]
        system.save(cg_gsdfile)
        order, _ = order_parameter(gsdfile, cg_gsdfile, mapping, r_max, a_max)

    Parameters
    ----------
    aa_gsd : str
        Path to the atomistic gsd file
    cg_gsd : str
        Path to the coarse-grain gsd file
    mapping : list of numpy.ndarray
        List of arrays containing indices of atomistic particles which map to
        each coarse-grain bead
    r_max : float
        Cut-off distance for the order parameter analysis
    a_max : float
        Cut-off angle in degrees for the order parameter analysis
    large : int, default 6
        The number of "beads" needed for a cluster to be considered "large"
    start : int, default -10
        The starting frame of the gsdfiles.

    Returns
    -------
    order : list of float
        Order parameter for each frame analyzed from the trajectory.
    cl_idx : list of numpy.ndarray
        The cluster index of each coarse-grain bead. See
        freud.cluster.Cluster.cluster_idx
    """
    order = []
    cl_idx = []
    with gsd.hoomd.open(aa_gsd) as aa_f, gsd.hoomd.open(cg_gsd) as cg_f:
        for aa_snap, cg_snap in zip(aa_f[start:], cg_f[start:]):
            f_box = freud.box.Box.from_box(aa_snap.configuration.box)
            unwrap_xyz = f_box.unwrap(
                aa_snap.particles.position, aa_snap.particles.image
            )
            aq = freud.locality.AABBQuery.from_system(cg_snap)
            n_list = aq.query(
                cg_snap.particles.position,
                query_args={"exclude_ii": True, "r_max": r_max},
            ).toNeighborList()

            vec_point = [
                get_plane_normal(unwrap_xyz[mapping[i]])[1]
                for i in n_list.point_indices
            ]
            vec_querypoint = [
                get_plane_normal(unwrap_xyz[mapping[i]])[1]
                for i in n_list.query_point_indices
            ]
            n_list.filter(
                [
                    angle_between_vectors(i, j) < a_max
                    for i, j in zip(vec_point, vec_querypoint)
                ]
            )
            cl = freud.cluster.Cluster()
            cl.compute(cg_snap, neighbors=n_list)
            n_large = sum([len(i) for i in cl.cluster_keys if len(i) >= large])
            n_total = cg_snap.particles.N
            order.append(n_large / n_total)
            cl_idx.append(cl.cluster_idx)
    return order, cl_idx


def concentration_profile(snap, A_indices, B_indices, n_bins=70, box_axis=0):
    """Calculate the concentration profile for two species
    along a spatial dimension.

    Parameters
    ----------
    snap : gsd.hoomd.Frame
        A snapshot object containing particle information, including positions.
    A_indices : numpy array
        Indices of particles belonging to species A.
    B_indices : numpy array
        Indices of particles belonging to species B.
    n_bins : int, optional, default 70
        Number of bins for the concentration profile.
    box_axis : int, optional, default 0
        Index of the box edge that the concentration profile is calculated.
        Options are 0, 1, or 2 which correspond to [x, y, z].

    Returns
    -------
    d_profile : numpy array
        Positions corresponding to the center of each bin
        in the concentration profile.
    A_count : numpy array
        Particle count for species A in each bin.
    B_count : numpy array
        Particle count for species B in each bin.
    total_count : numpy array
        Total particle count in each bin.

    Notes
    -----
    Use this to create a concentration profile plot of "left" species
    and "right" species in the simulation's volume.

    Example::
        # Plot the concentration profile for a snapshot with 200 particles
        # "left" species are particles 0-99 and "right" species are 100-199

        from cmeutils.structure import concentration_profile
        import matplotlib.pyplot as plt

        x_range, left, right, total = concentration_profile(
            snap=snapshot,
            A_indices=range(0, 100),
            B_indices=range(100, 200),
            n_bins=50,
            box_axis=0
        )

        plt.plot(x_range, left/total)
        plt.plot(x_range, right/total)

    """

    L = snap.configuration.box[box_axis]
    dl = L / n_bins
    d_profile = np.linspace(-L / 2 + dl, L / 2, n_bins)

    A_pos = snap.particles.position[A_indices, box_axis]
    B_pos = snap.particles.position[B_indices, box_axis]
    A_count, _ = np.histogram(A_pos, bins=d_profile, density=False)
    B_count, _ = np.histogram(B_pos, bins=d_profile, density=False)

    total_count = A_count + B_count

    return d_profile[:-1], A_count, B_count, total_count


def all_atom_rdf(
    gsdfile,
    start=0,
    stop=-1,
    r_max=None,
    r_min=0,
    bins=100,
):
    """Compute intermolecular RDF from a GSD file.

    This function calculates the radial distribution function given a GSD file
    for all atoms. By default it will calculate the RDF
    for the entire trajectory.
    It is assumed that the bonding, number of particles, and simulation box do
    not change during the simulation.

    Parameters
    ----------
    gsdfile : str
        Filename of the GSD trajectory.
    start : int, default 0
        Starting frame index for accumulating the RDF. Negative numbers index
        from the end.
    stop : int, default -1
        Final frame index for accumulating the RDF. If None, the last frame
        will be used.
    r_max : float, default None
        Maximum radius of RDF. If None, half of the maximum box size is used.
    r_min : float, default 0
        Minimum radius of RDF.
    bins : int, default 100
        Number of bins to use when calculating the RDF.

    Returns
    -------
    freud.density.RDF
    """
    with gsd.hoomd.open(gsdfile, mode="r") as trajectory:
        snap = trajectory[start]
        if r_max is None:
            # Use a value just less than half the maximum box length.
            r_max = np.nextafter(
                np.max(snap.configuration.box[:3]) * 0.5, 0, dtype=np.float32
            )
        rdf = freud.density.RDF(bins=bins, r_max=r_max, r_min=r_min)
        for snap in trajectory[start:stop]:
            rdf.compute(snap, reset=False)
    return rdf
