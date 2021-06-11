from cmeutils import gsd_utils
import freud
import gsd
import gsd.hoomd
import numpy as np


def rotmat_to_q(m):
    """Convert a 3x3 rotation matrix to a quaternion."""
    qw = np.sqrt(1 + m[0,0] + m[1,1] + m[2,2]) / 2
    qx = (m[2,1] - m[1,2])/(4 * qw)
    qy = (m[0,2] - m[2,0])/(4 * qw)
    qz = (m[1,0] - m[0,1])/(4 * qw)
    return np.array([qx, qy, qz, qw])

def rotation_matrix_from_to(a, b):
    """Compute a rotation matrix R such that norm(b)*dot(R,a)/norm(a) = b.

    Parameters
    ----------
    a : numpy.ndarray
        A 3-vector
    b : numpy.ndarray
        Another 3-vector

    Returns:
    R numpy.ndarray:
        The 3x3 rotation matrix that will would rotate a parallel to b.
    """
    a1 = a/np.linalg.norm(a)
    b1 = b/np.linalg.norm(b)
    theta = np.arccos(np.dot(a1,b1))
    if theta<1e-6 or np.isnan(theta):
        return np.identity(3)
    if np.pi-theta<1e-6: #TODO(Eric): verify correct
        d = np.array([1.,0,0])
        x = np.cross(a1,d)
    else:
        x = np.cross(a1,b1)
        x /= np.linalg.norm(x)
    A = np.array([ [0,-x[2],x[1]], [x[2],0,-x[0]], [-x[1],x[0],0] ])
    R = np.identity(3) + np.sin(theta)*A + (1.-np.cos(theta))*np.dot(A,A)
    return R

def get_rotmats(n_views = 20):
    """Get the rotation matrices for the specified number of views."""
    ga = np.pi * (3 - 5**0.5)
    theta = ga * np.arange(n_views-3)
    z = np.linspace(1 - 1/(n_views-3), 1/(n_views-3), n_views-3)
    radius = np.sqrt(1 - z * z)
    points = np.zeros((n_views, 3))
    points[:-3,0] = radius * np.cos(theta)
    points[:-3,1] = radius * np.sin(theta)
    points[:-3,2] = z
    # face on
    points[-3] = np.array([0,0,1])
    # edge on
    points[-2] = np.array([0,1,1])
    # corner on
    points[-1] = np.array([1,1,1])
    return [rotation_matrix_from_to(i,np.array([0,0,1])) for i in points]

def get_quaternions(n_views = 20):
    """Get the quaternions for the specified number of views."""
    rotmats = get_rotmats(n_views = n_views)
    return [rotmat_to_q(i) for i in rotmats]


def gsd_rdf(
    gsdfile,
    A_name,
    B_name,
    start=0,
    stop=None,
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
        will be used. (default None)
    r_max : float
        Maximum radius of RDF. If None, half of the maximum box size is used.
        (default None)
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
    if not stop:
        stop = -1

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
