import freud
import gsd
import gsd.hoomd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def frame_get_type_position(gsdfile, typename, frame=-1):
    """
    This function returns the  positions of a particular particle
    type from a frame of a gsd trajectory file.

    Parameters
    ----------
    gsdfile : str,
              filename of the gsd trajectory
    typename : str,
               name of particles of which to get the positions
               (found in gsd.hoomd.Snapshot.particles.types)
    frame : int,
            frame number to get positions from. Supports
            negative indexing. (default -1)

    Returns
    -------
    numpy.ndarray
    """
    with gsd.hoomd.open(name=gsdfile, mode='rb') as f:
        snap = f[frame]
    typepos = snap.particles.position[
            snap.particles.typeid == snap.particles.types.index(typename)
            ]
    return typepos


def snap_bond_graph(snap):
    """
    Given a snapshot from a trajectory return an array
    corresponding to the molecule index of each particle

    Parameters
    ----------
    snap : gsd.hoomd.Snapshot

    Returns
    -------
    numpy array (N_particles,)

    """
    bond_graph = csr_matrix(
            (np.ones(snap.bonds.N),
            (snap.bonds.group[:, 0], snap.bonds.group[:, 1])),
            shape=(snap.particles.N, snap.particles.N),
    )
    n_components, labels = connected_components(
            csgraph=bond_graph,
            directed=False
            )
    return labels


def gsd_rdf(
    gsdfile,
    A_name,
    B_name,
    start=0,
    stop=None,
    rmax=None,
    rmin=0,
    bins=50,
    exclude_bonded=True,
):
    """
    This function calculates the radial distribution function given
    a gsd file and the names of the particles. By default it will calculate
    the rdf for the entire the trajectory.

    Parameters
    ----------
    gsdfile : str, filename of the gsd trajectory
    A_name, B_name : str, name(s) of particles between which to calculate the rdf
                     (found in gsd.hoomd.Snapshot.particles.types)
    start : int, which frame to start accumulating the rdf (default 0)
            (negative numbers index from the end)
    stop : int, which frame to stop accumulating the rdf (default None)
           If none is given, the function will default to the last frame.
    rmax : float, maximum radius to consider. (default None)
           If none is given, it'll be the minimum box length / 4
    bins : int, number of bins to use when calculating the distribution.
    exclude_bonded : bool, whether to remove particles in same molecule from the
                     neighborlist (default True)

    NOTE: It is assumed that the bonding and the number of particles does not change
    during the simulation

    Returns
    -------
    freud.density.RDF
    """
    with gsd.hoomd.open(gsdfile) as t:
        snap = t[0]

        if rmax is None:
            rmax = max(snap.configuration.box[:3]) * 0.45

        rdf = freud.density.RDF(bins=bins, r_max=rmax, r_min=rmin)

        type_A = snap.particles.typeid == snap.particles.types.index(A_name)
        type_B = snap.particles.typeid == snap.particles.types.index(B_name)

        if exclude_bonded:
            molecules = snap_bond_graph(snap)
            molecules_A = molecules[type_A]
            molecules_B = molecules[type_B]

        for snap in t[start:stop]:

            A_pos = snap.particles.position[type_A]
            if A_name != B_name:
                B_pos = snap.particles.position[type_B]
            else:
                B_pos = A_pos

            box = snap.configuration.box
            system = (box, A_pos)
            aq = freud.locality.AABBQuery.from_system(system)
            nlist = aq.query(B_pos, {"r_max": rmax}).toNeighborList()

            if exclude_bonded:
                nlist.filter(
                    molecules_A[nlist.point_indices]
                    != molecules_B[nlist.query_point_indices]
                )

            rdf.compute(aq, neighbors=nlist, reset=False)
        return rdf
