import freud
import gsd
import gsd.hoomd
import numpy as np
from rowan import vector_vector_rotation

from cmeutils import gsd_utils


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
            molecules = gsd_utils.snap_molecule_cluster(snap=snap)
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
                B_pos, {"r_max":r_max, "exclude_ii":exclude_ii}
            ).toNeighborList()

            if exclude_bonded:
                pre_filter = len(nlist)
                nlist.filter(
                    molecules_A[nlist.point.indices]
                    != molecules_B[nlist.query_point_indices]
                )
                post_filter = len(nlist)

            rdf.compute(aq, neighbors=nlist, reset=False)
    normalization = post_filter / pre_filter if exclude_bonded else 1
    return rdf, normalization

def get_centers(gsdfile, new_gsdfile):
    """Create a gsd file containing the molecule centers from an existing gsd file.
    

    This function calculates the centers of a trajectory given a GSD file
    and stores them into a new GSD file just for centers. By default it will calculate the centers of an entire trajectory.

    Parameters
    ----------
    gsdfile : str
        Filename of the GSD trajectory.
    new_gsdfile : str
        Filename of new GSD for centers.
    """
    with gsd.hoomd.open(new_gsdfile, 'wb') as new_traj, gsd.hoomd.open(gsdfile, 'rb') as traj:
        snap = traj[0]
        cluster_idx = gsd_utils.snap_molecule_cluster(snap=snap)
        for snap in traj:
            new_snap = gsd.hoomd.Snapshot()
            new_snap.configuration.box = snap.configuration.box
            clp = freud.cluster.ClusterProperties()
            clp.compute(snap, cluster_idx);
            new_snap.particles.position = clp.centers 
            new_snap.particles.N = len(clp.centers)
            new_snap.particles.types = ["A"]
            new_snap.particles.typeid = np.zeros(len(clp.centers)) 
            new_snap.validate()
            new_traj.append(new_snap)    
