import freud
import gsd
import gsd.hoomd
import numpy as np
from rowan import vector_vector_rotation

from cmeutils import gsd_utils
from cmeutils.geometry import get_plane_normal, angle_between_vectors


def get_quaternions(n_views = 20):
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
    if n_views <=3 or not isinstance(n_views, int):
        raise ValueError("Please set n_views to an integer greater than 3.")
    # Calculate points for even distribution on a sphere
    ga = np.pi * (3 - 5**0.5)
    theta = ga * np.arange(n_views-3)
    z = np.linspace(1 - 1/(n_views-3), 1/(n_views-3), n_views-3)
    radius = np.sqrt(1 - z * z)
    points = np.zeros((n_views, 3))
    points[:-3,0] = radius * np.cos(theta)
    points[:-3,1] = radius * np.sin(theta)
    points[:-3,2] = z

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
        gsd.hoomd.Snapshot.particles.types)
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
    with gsd.hoomd.open(gsdfile, mode="rb") as trajectory:
        snap = trajectory[0]

        if r_max is None:
            # Use a value just less than half the maximum box length.
            r_max = np.nextafter(
                np.max(snap.configuration.box[:3]) * 0.5, 0, dtype=np.float32
            )

        rdf = freud.density.RDF(bins=bins, r_max=r_max, r_min=r_min)

        type_A = snap.particles.typeid == snap.particles.types.index(A_name)
        type_B = snap.particles.typeid == snap.particles.types.index(B_name)

        if exclude_bonded:
            molecules = gsd_utils.get_molecule_cluster(snap=snap)
            molecules_A = molecules[type_A]
            molecules_B = molecules[type_B]

        for snap in trajectory[start:stop]:
            A_pos = snap.particles.position[type_A]
            if A_name == B_name:
                B_pos = A_pos
                exclude_ii = True
            else:
                B_pos = snap.particles.position[type_B]
                exclude_ii = False

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
        return rdf, normalization


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
        Cut-off angle for the order parameter analysis
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
                query_args={"exclude_ii":True, "r_max": r_max}
            ).toNeighborList()

            vec_point = [
                get_plane_normal(unwrap_xyz[mapping[i]])[1]
                for i in n_list.point_indices
            ]
            vec_querypoint = [
                get_plane_normal(unwrap_xyz[mapping[i]])[1]
                for i in n_list.query_point_indices
            ]
            n_list.filter([
                angle_between_vectors(i,j) < a_max
                for i,j in zip(vec_point, vec_querypoint)
            ])
            cl = freud.cluster.Cluster()
            cl.compute(cg_snap, neighbors=n_list)
            n_large = sum([len(i) for i in cl.cluster_keys if len(i) >= large])
            n_total = cg_snap.particles.N
            order.append(n_large / n_total)
            cl_idx.append(cl.cluster_idx)
    return order, cl_idx


def all_atom_rdf(gsdfile,
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
    with gsd.hoomd.open(gsdfile, mode="rb") as trajectory:
        snap = trajectory[start]
        if r_max is None:
            #Use a value just less than half the maximum box length.
            r_max = np.nextafter(
                np.max(snap.configuration.box[:3]) * 0.5, 0, dtype=np.float32
            )
        rdf = freud.density.RDF(bins=bins, r_max=r_max, r_min=r_min)
        for snap in trajectory[start:stop]:
            rdf.compute(snap, reset=False)
    return rdf
