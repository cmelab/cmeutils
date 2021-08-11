import freud
import gsd
import gsd.hoomd
import numpy as np


def get_type_position(
        typename,
        gsd_file=None,
        snap=None,
        gsd_frame=-1,
        images=False):
    """
    This function returns the  positions of a particular particle
    type from a frame of a gsd trajectory file or from a snapshot.
    Pass in either a gsd file or a snapshot, but not both.

    Parameters
    ----------
    typename : str or list of str
        Name of particles of which to get the positions
        (found in gsd.hoomd.Snapshot.particles.types)
        If you want the positions of multiple types, pass
        in a list. Ex.) ['ca', 'c3']
    gsd_file : str, default None
        Filename of the gsd trajectory
    snap : gsd.hoomd.Snapshot, default None
        Trajectory snapshot
    gsd_frame : int, default -1
        Frame number to get positions from. Supports negative indexing.
    images : bool, default False
        If True; an array of the particle images is returned in addition
        to the particle positions.

    Returns
    -------
    numpy.ndarray(s)
        Retruns a single array of positions or
        arrays of positions and images
    """
    snap = _validate_inputs(gsd_file, snap, gsd_frame)
    if isinstance(typename, str):
        typename = [typename]

    type_pos = []
    type_images = []
    for _type in typename:
        type_pos.extend(
                snap.particles.position[
                snap.particles.typeid == snap.particles.types.index(_type)
            ]
        )
        if images:
            type_images.extend(
                snap.particles.image[
                    snap.particles.typeid == snap.particles.types.index(_type)
                ]
            )
    if images:
        return np.array(type_pos), np.array(type_images)
    else:
        return np.array(type_pos)


def get_all_types(gsd_file=None, snap=None, gsd_frame=-1):
    """
    Returns all particle types in a hoomd trajectory

    Parameters
    ----------
    gsd_file : str, default None
        Filename of the gsd trajectory
    snap : gsd.hoomd.Snapshot, default None
        Trajectory snapshot
    gsd_frame : int, default -1
        Frame number to get positions from. Supports negative indexing.

    Returns
    -------
    numpy.ndarray
    """
    snap = _validate_inputs(gsd_file, snap, gsd_frame)
    return snap.particles.types


def snap_molecule_cluster(gsd_file=None, snap=None, gsd_frame=-1):
    """Find molecule index for each particle.

    Compute clusters of bonded molecules and return an array of the molecule
    index of each particle.
    Pass in either a gsd file or a snapshot, but not both

    Parameters
    ----------
    gsd_file : str, default None
        Filename of the gsd trajectory
    snap : gsd.hoomd.Snapshot, default None
        Trajectory snapshot.
    gsd_frame : int, default -1
        Frame number of gsd_file to use to compute clusters.

    Returns
    -------
    numpy array (N_particles,)
    """
    snap = _validate_inputs(gsd_file, snap, gsd_frame)
    system = freud.AABBQuery.from_system(snap)
    n_query_points = n_points = snap.particles.N
    query_point_indices = snap.bonds.group[:, 0]
    point_indices = snap.bonds.group[:, 1]
    distances = system.box.compute_distances(
        system.points[query_point_indices], system.points[point_indices]
    )
    nlist = freud.NeighborList.from_arrays(
        n_query_points, n_points, query_point_indices, point_indices, distances
    )
    cluster = freud.cluster.Cluster()
    cluster.compute(system=system, neighbors=nlist)
    return cluster.cluster_idx


def _validate_inputs(gsd_file, snap, gsd_frame):
    if all([gsd_file, snap]):
        raise ValueError("Only pass in one of snapshot, gsd_file")
    if gsd_file:
        assert isinstance(gsd_frame, int)
        try:
            with gsd.hoomd.open(name=gsd_file, mode="rb") as f:
                snap = f[gsd_frame]
        except Exception as e:
            print("Unable to open the gsd_file")
            raise e
    elif snap:
        assert isinstance(snap, gsd.hoomd.Snapshot)
    return snap
