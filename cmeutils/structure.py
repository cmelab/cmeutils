import itertools
import warnings

import freud
import gsd
import gsd.hoomd
import networkx as nx
import numpy as np

from cmeutils.geometry import (
    angle_between_vectors,
    dihedral_angle,
    get_plane_normal,
)
from cmeutils.gsd_utils import frame_to_freud_system, snapshot_to_graph
from cmeutils.plotting import get_histogram


def angle_distribution(
    gsd_file,
    A_name,
    B_name,
    C_name,
    start=0,
    stop=-1,
    stride=1,
    degrees=False,
    histogram=False,
    theta_min=0.0,
    theta_max=None,
    normalize=False,
    as_probability=False,
    bins="auto",
):
    """Returns the bond angle distribution for a given triplet of particles

    Parameters
    ----------
    gsdfile : str
        Filename of the GSD trajectory.
    A_name, B_name, C_name : str, required
        Name(s) of particles that form the angle triplet
        (found in gsd.hoomd.Frame.particles.types)
        They must be given in the same order as they form the angle
    start : int, default = 0
        Starting frame index for accumulating bond lengths.
        Negative numbers index from the end.
    stop : int, default = -1
        Final frame index for accumulating bond lengths.
    stride : int, default = 1
        The stride size when iterating through start:stop
    degrees : bool, default=False
        If True, the angle values are returned in degrees.
        if False, the angle values are returned in radians.
    histogram : bool, default = False
        If set to True, places the resulting angles into a histogram
        and retrims the histogram's bin centers and heights as
        opposed to the actual calcualted angles.
        If set to False, an array of the actual angles measured is returned.
    theta_min : float, default = 0.0
        Sets the minimum theta value to be included in the distribution
    theta_max : float, default = None
        Sets the maximum theta value to be included in the distribution
        If left as None, then theta_max will be either pi radians or
        180 degrees depending on the value set for the degrees parameter
    normalize : bool, default = False
        If set to True, normalizes the angle distribution by the
        sum of the bin heights, so that the distribution adds up to 1.
        If set to `True`, you are left with the probability density
        function (PDF). See `as_probability` to convert the probability
        density function to a probability.
    as_probability : bool, default = False
        If set to `True`, then the PDF is multiplied by bin widths
        to give a unitless probability.
    bins : float, int, or str,  default = "auto"
        The number of bins to use when finding the distribution
        of bond angles. Using "auto" will set the number of
        bins based on the ideal bin size for the data.
        See the numpy.histogram docs for more details.

    Returns
    -------
    1-D numpy.array  or 2-D numpy.array
        If histogram is False, Array of actual bond angles.
        If histogram is True, returns a 2D array of bin centers and bin heights.

    Notes
    -----
    Results based on parameter combinations:

    | histogram | normalize | as_probability | Result          |
    |-----------|-----------|----------------|-----------------|
    | True      | False     | False          | Raw bin counts  |
    | True      | True      | False          | PDF             |
    | True      | True      | True           | PMF             |
    | False     | False     | False          | Array of angles |
    | True      | False     | True           | Invalid         |

    """
    if as_probability and not normalize:
        raise ValueError(
            "`normalize` must be `True` to use `as_probability=True`"
        )
    if not degrees and theta_max is None:
        theta_max = np.pi
    elif degrees and theta_max is None:
        theta_max = 180

    trajectory = gsd.hoomd.open(gsd_file, mode="r")
    name = "-".join([A_name, B_name, C_name])
    name_rev = "-".join([C_name, B_name, A_name])

    angles = []
    for snap in trajectory[start:stop:stride]:
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
            as_probability=as_probability,
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
    stride=1,
    histogram=False,
    l_min=0.0,
    l_max=4.0,
    normalize=False,
    as_probability=False,
    bins=100,
):
    """Returns the bond length distribution for a given bond pair

    Parameters
    ----------
    gsdfile : str
        Filename of the GSD trajectory.
    A_name, B_name : str, required
        Name(s) of particles that form the bond pair
        (found in gsd.hoomd.Frame.particles.types)
    start : int, default = 0
        Starting frame index for accumulating bond lengths.
        Negative numbers index from the end.
    stop : int, default = -1
        Final frame index for accumulating bond lengths.
    stride : int, default = 1
        The stride size when iterating through start:stop
    histogram : bool, default = False
        If set to True, places the resulting bonds into a histogram
        and retrums the histogram's bin centers and heights as
        opposed to the actual calcualted bonds.
        If set to False, an array of the actual lengths measured is returned.
    l_min : float, default = 0.0
        Sets the minimum bond length to be included in the distribution
    l_max : float, default = 5.0
        Sets the maximum bond length value to be included in the distribution
    normalize : bool, default = False
        If set to True, normalizes the bond distribution by the
        sum of the bin heights, so that the distribution adds up to 1.
        If set to `True`, you are left with the probability density
        function (PDF). See `as_probability` to convert the probability
        density function to a probability.
    as_probability : bool, default = False
        If set to `True`, then the PDF is multiplied by bin widths
        to give a unitless probability.
    bins : float, int, or str,  default = "auto"
        The number of bins to use when finding the distribution
        of bond angles. Using "auto" will set the number of
        bins based on the ideal bin size for the data.
        See the numpy.histogram docs for more details.

    Returns
    -------
    1-D numpy.array  or 2-D numpy.array
        If histogram is False, Array of actual bond lengths.
        If histogram is True, returns a 2D array of bin centers and bin heights.

    Notes
    -----
    Results based on parameter combinations:

    | histogram | normalize | as_probability | Result                |
    |-----------|-----------|----------------|-----------------------|
    | True      | False     | False          | Raw bin counts        |
    | True      | True      | False          | PDF                   |
    | True      | True      | True           | PMF                   |
    | False     | False     | False          | Array of bond lengths |
    | True      | False     | True           | Invalid               |

    """
    if as_probability and not normalize:
        raise ValueError(
            "`normalize` must be `True` to use `as_probability=True`"
        )

    trajectory = gsd.hoomd.open(gsd_file, mode="r")
    name = "-".join([A_name, B_name])
    name_rev = "-".join([B_name, A_name])

    bonds = []
    for snap in trajectory[start:stop:stride]:
        if name not in snap.bonds.types and name_rev not in snap.bonds.types:
            raise ValueError(
                f"Bond types {name} or {name_rev} not found snap.bonds.types."
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
            as_probability=as_probability,
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
    stride=1,
    degrees=False,
    histogram=False,
    normalize=False,
    as_probability=False,
    bins="auto",
):
    """Returns the bond dihedral distribution for a given quadruplett of particles

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
        Final frame index for accumulating bond dihedrals. (default -1)
    stride : int
        The stride size when iterating through start:stop
    degrees : bool, default=False
        If True, the angle values are returned in degrees.
        if False, the angle values are returned in radians.
    histogram : bool, default=False
        If set to True, places the resulting angles into a histogram
        and retrums the histogram's bin centers and heights as
        opposed to the actual calcualted angles.
        If set to False, an array of the actual angles measured is returned.
    normalize : bool, default=False
        If set to True, normalizes the dihedral distribution by the
        sum of the bin heights, so that the distribution adds up to 1.
        If set to `True`, you are left with the probability density
        function (PDF). See `as_probability` to convert the probability
        density function to a probability.
    as_probability : bool, default=False
        If set to `True`, then the PDF is multiplied by bin widths
        to give a unitless probability.
    bins : float, int, or str,  default="auto"
        The number of bins to use when finding the distribution
        of dihedral angles. Using "auto" will set the number of
        bins based on the ideal bin size for the data.
        See the numpy.histogram docs for more details.

    Returns
    -------
    1-D numpy.array  or 2-D numpy.array
        If histogram is False, Array of actual dihedral angles
        If histogram is True, returns a 2D array of bin centers and bin heights.

    Notes
    -----
    Results based on parameter combinations:

    | histogram | normalize | as_probability | Result                   |
    |-----------|-----------|----------------|--------------------------|
    | True      | False     | False          | Raw bin counts           |
    | True      | True      | False          | PDF                      |
    | True      | True      | True           | PMF                      |
    | False     | False     | False          | Array of dihedral angles |
    | True      | False     | True           | Invalid                  |
    """
    if as_probability and not normalize:
        raise ValueError(
            "`normalize` must be `True` to use `as_probability=True`"
        )

    trajectory = gsd.hoomd.open(gsd_file, mode="r")
    name = "-".join([A_name, B_name, C_name, D_name])
    name_rev = "-".join([D_name, C_name, B_name, A_name])

    dihedrals = []
    for snap in trajectory[start:stop:stride]:
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
            as_probability=as_probability,
            bins=bins,
            x_range=(-np.pi, np.pi),
        )
        return np.stack((bin_centers, bin_heights)).T
    else:
        return np.array(dihedrals)


def gsd_rdf(
    gsdfile,
    A_name=None,
    B_name=None,
    start=0,
    stop=-1,
    stride=1,
    r_max=None,
    r_min=0,
    bins=100,
    exclude_bond_depth=None,
    exclude_all_bonded=False,
):
    """Uses freud's RDF module to calculate an RDF averaged over a GSD file.

    Notes
    -----
    This method lets you set a bond-depth exclusion to prevent neighbors of the
    same molecule to be used in the RDF calculations. Bond depth is counted in
    terms of steps away on a connected bond graph.

    For example, ``exclude_bond_depth=1`` excludes pairs directly bonded together,
    ``exclude_bond_depth=2`` excludes both pairs (i, j) and (i, k) in (i-j-k), and so on.
    This can be set to any positive integer value, and isn't limited to particles belonging
    to the same bond, angle or dihedral. To exclude all intramolecular pairs use
    the ``exclude_all_bonded`` parameter instead.

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
    stride : int, default = 1
        The stride size when iterating through start:stop
    r_max : float
        Maximum radius of RDF. If None, half of the minimum box size is used.
    r_min : float
        Minimum radius of RDF. (default 0)
    bins : int
        Number of bins to use when calculating the RDF. (default 100)
    exclude_bond_depth : int, optional (default None)
        Excludes all pairs within a depth (distance on a bond graph)
        from the RDF calculation.
    exclude_all_bonded : bool, optional (default False)
        Excludes all pairs belonging to the same molecule from the
        RDF calculation.

    Returns
    -------
    tuple(rdf, rdf_correction) : (freud.density.RDF, float)
        Access r values with ``rdf.bin_centers`` and g(r) with ``rdf.rdf``
        rdf_correction is always 1 unless ``exclude_bond_depth`` or ``exclude_all_bonded`` are used.
        This corrects the g(r) normalization to account for excluded pairs.
        To include this in the results, g(r) = rdf.rdf * rdf_correction
    """
    if any([A_name, B_name]) and not all([A_name, B_name]):
        raise ValueError(
            "If A_name or B_name is given, the other must be defined as well. "
            "To calculate an RDF between the same bead type, set A_name and B_name equal to the same value. "
            "To calculate an RDF between all possible pairs, leave both as ``None``. "
        )

    if all([exclude_bond_depth, exclude_all_bonded]):
        raise ValueError(
            "Only use one of excluded_bond_depth and exclude_all_bonded."
        )

    with gsd.hoomd.open(gsdfile, mode="r") as trajectory:
        # Grab the first snapshot for some book-keeping.
        snap = trajectory[start]

        # Use a value just less than half the minimum box length.
        # TODO: Iterate through all boxes, use the smallest one? Edge cases might arise with NPT sims
        if r_max is None:
            r_max = np.nextafter(
                np.min(snap.configuration.box[:3]) * 0.49, 0, dtype=np.float32
            )

        rdf = freud.density.RDF(bins=bins, r_max=r_max, r_min=r_min)

        # Filter particles by type A and type B
        if A_name is not None and B_name is not None:
            type_A = snap.particles.typeid == snap.particles.types.index(A_name)
            type_B = snap.particles.typeid == snap.particles.types.index(B_name)
            # These 2 *_indices variables store global particle indices (Before filtering by type)
            # If excluding by bond depth, these need to be passed into filter_nlist
            type_A_indices = np.where(type_A)[0]
            type_B_indices = np.where(type_B)[0]
            exclude_ii = A_name == B_name
        else:  # Use all particles for this RDF
            type_A = type_B = np.ones(
                snap.particles.N, dtype=bool
            )  # Array of True at all indices
            type_A_indices = type_B_indices = np.arange(snap.particles.N)
            exclude_ii = True

        # Build up pair exclusions if exclude_bond_depth or exclude_all_bonded
        # Reuse these for each frame's RDF
        # If the bonding topology isn't changing, we only need to get bond graph and excluded pairs once
        rdf_correction = 1
        if exclude_bond_depth or exclude_all_bonded:
            max_idx = snap.particles.N
            bond_graph = snapshot_to_graph(snap)
            # This gives a seuquence of tuples [(1, 4), (5, 8)...(i, j)]
            excluded_pairs = get_excluded_pairs(
                bond_graph, exclude_bond_depth, exclude_all_bonded
            )
            # Map information of sequence of tuples to an array of unique ints.
            # This is used for faster filtering in filter_nlist() (vectorized instead of for loop)
            excluded_pairs_encoded = np.array(
                [i * max_idx + j for i, j in excluded_pairs]
            )

            n_excluded = len(excluded_pairs)
            if A_name == B_name or not any(
                [A_name, B_name]
            ):  # Using same type or all particles
                n_total_pairs = (
                    len(type_A_indices) * (len(type_A_indices) - 1) / 2
                )
            else:  # RDF is not between same types, or using all particles
                n_total_pairs = len(type_A_indices) * len(type_B_indices)
            # Overwrite default value only if using exclude_bond_depth
            if n_total_pairs == n_excluded:
                warnings.warn(
                    "Exclusions resulted in no pairs being used to calculate this RDF."
                )
            else:
                rdf_correction = n_total_pairs / (n_total_pairs - n_excluded)

        for snap in trajectory[start:stop:stride]:
            A_xyz = snap.particles.position[type_A_indices]
            B_xyz = snap.particles.position[type_B_indices]

            # Build up the complete neighborlist
            box = snap.configuration.box
            system = (box, A_xyz)
            aq = freud.locality.AABBQuery.from_system(system)
            query_args = {"r_max": r_max, "exclude_ii": exclude_ii}
            nlist = aq.query(B_xyz, query_args).toNeighborList()

            # Filter excluded pairs from all pairs in nlist
            if exclude_bond_depth or exclude_all_bonded:
                nlist = filter_nlist(
                    nlist=nlist,
                    excluded_pairs_encoded=excluded_pairs_encoded,
                    query_indices=type_A_indices,
                    point_indices=type_B_indices,
                    max_idx=snap.particles.N,
                )

            rdf.compute(aq, neighbors=nlist, reset=False)
    return rdf, rdf_correction


def structure_factor(
    gsdfile,
    k_min,
    k_max,
    start=0,
    stop=-1,
    stride=1,
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
    start : int, default = 0
        Starting frame index for accumulating the Sq. Negative numbers index
        from the end.
    stop : int, optional default = -1
        Final frame index for accumulating the Sq.
    stride : int, default = 1
        The stride size when iterating through start:stop
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
        for frame in trajectory[start:stop:stride]:
            system = frame_to_freud_system(frame=frame, ref_length=ref_length)
            sf.compute(system=system, reset=False)
    return sf


def diffraction_pattern(
    gsdfile,
    views,
    start=0,
    stop=-1,
    stride=1,
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
    start : int, default = 0
        Starting frame index for accumulating the Sq. Negative numbers index
        from the end.
    stop : int, optional default = -1
        Final frame index for accumulating the Sq.
    stride : int, default = 1
        The stride size when iterating through start:stop
    ref_length : float, optional, default = None
        Set a reference length to convert from reduced units to real units.
        If None, uses 1 by default.
    grid_size : unsigned int, optional, default = 1024
        Resolution of the diffraction grid.
    output_size : unsigned int, optional, default = None
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
        for frame in trajectory[start:stop:stride]:
            system = frame_to_freud_system(frame=frame, ref_length=ref_length)
            for view in views:
                dp.compute(system=system, view_orientation=view, reset=False)
    return dp


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


def get_excluded_pairs(
    bond_graph, excluded_bond_depth=None, exclude_all_bonded=False
):
    """Returns a set of (i, j) pairs to exclude based on step distance of a bond graph."""
    excluded_pairs = set()
    if excluded_bond_depth:
        for i in bond_graph.nodes:
            lengths = nx.single_source_shortest_path_length(
                bond_graph, i, cutoff=excluded_bond_depth
            )
            for j, dist in lengths.items():
                # use j > i for 2 reasons: skip adding the same pair twice and its used in filter_nlist
                if j > i and dist <= excluded_bond_depth:
                    excluded_pairs.add((i, j))
    elif exclude_all_bonded:
        for component in nx.connected_components(bond_graph):
            component = sorted(component)
            for i, j in itertools.combinations(component, 2):
                excluded_pairs.add((i, j))
    return excluded_pairs


def filter_nlist(
    nlist, excluded_pairs_encoded, query_indices, point_indices, max_idx
):
    """Filter a freud NeighborList by removing excluded pairs.

    Handles the index space mismatch between the NeighborList (which uses
    local indices into type filtered position arrays) and excluded_pairs
    (which uses global particle indices from the bond graph).

    Parameters
    ----------
    nlist : freud.locality.NeighborList
        Neighbor list to filter, as returned by AABBQuery.toNeighborList().
        Query point and point indices are local (0 indexed within their
        type subsets).
    excluded_pairs_encoded : np.ndarray of int, shape (N_excluded,)
        Encoded global particle index pairs to exclude, precomputed as
        ``i * max_idx + j`` where ``i < j`` and ``max_idx = N_particles``.
        Encodes the output of get_excluded_pairs().
    query_indices : np.ndarray of int, shape (N_query,)
        Global particle indices of the query points (type A particles).
        Maps local query index to global particle index.
    point_indices : np.ndarray of int, shape (N_points,)
        Global particle indices of the reference points (type B particles).
        Maps local point index to global particle index.

    Returns
    -------
    freud.locality.NeighborList
        Filtered neighbor list with excluded pairs removed. Local indices
        are preserved, so it can be passed directly to freud.density.RDF.compute().
    """
    # Local indices of the neighborlist, since it was created after filtering particle type
    i_local = nlist.query_point_indices
    j_local = nlist.point_indices

    # excluded_pair indices are global, they came from the the bond graph of all particles
    # Convert i_local and j_local to global indices so we can look up against excluded_pairs
    i_global = query_indices[i_local]
    j_global = point_indices[j_local]

    # Sort all pairs so that the first index is always < the second.
    lo = np.minimum(i_global, j_global)
    hi = np.maximum(i_global, j_global)

    # Encode each pair as a single integer for vectorized lookup.
    # (lo, hi) = lo * max_idx + hi which is unique as long as hi < max_idx
    neighbor_pairs_encoded = lo * max_idx + hi

    # vectorized set lookup
    keep = ~np.isin(neighbor_pairs_encoded, excluded_pairs_encoded)
    return freud.locality.NeighborList.from_arrays(
        num_query_points=nlist.num_query_points,
        num_points=nlist.num_points,
        query_point_indices=i_local[keep],
        point_indices=j_local[keep],
        vectors=nlist.vectors[keep],
    )
