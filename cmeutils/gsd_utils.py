import gsd
import gsd.hoomd
import freud


def get_type_position(type_name, gsd_file=None, snap=None, gsd_frame=-1):
    """
    This function returns the  positions of a particular particle
    type from a frame of a gsd trajectory file or from a snapshot.
    Pass in either a gsd file or a snapshot, but not both.

    Parameters
    ----------
    type_name : str,
               name of particles of which to get the positions
               (found in gsd.hoomd.Snapshot.particles.types)
    gsd_file : str,
              filename of the gsd trajectory (default = None)
    snap : gsd.hoomd.Snapshot
            Trajectory snapshot (default = None)
    gsd_frame : int,
            frame number to get positions from. Supports
            negative indexing. (default = -1)

    Returns
    -------
    numpy.ndarray
    """
    snap = _validate_inputs(gsd_file, snap, gsd_frame)
    type_pos = snap.particles.position[
            snap.particles.typeid == snap.particles.types.index(type_name)
            ]
    return type_pos

def get_all_types(gsd_file=None, snap=None, gsd_frame=-1):
    """
    Returns all particle types in a hoomd trajectory
    
    Parameters
    ----------
    gsd_file : str,
              filename of the gsd trajectory (default = None)
    snap : gsd.hoomd.Snapshot
            Trajectory snapshot (default = None)
    gsd_frame : int,
            frame number to get positions from. Supports
            negative indexing. (default = -1)

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
    gsd_file : str,
               filename of the gsd trajectory (default = None)
    snap : gsd.hoomd.Snapshot
        Trajectory snapshot. (default = None)
    gsd_frame : int,
               frame number of gsd_file to use in computing clusters. (default = -1)

    Returns
    -------
    numpy array (N_particles,)

    """
    snap = _validate_inputs(gsd_file, snap, gsd_frame)
    system = freud.AABBQuery.from_system(snap)
    num_query_points = num_points = snap.bonds.N
    query_point_indices = snap.bonds.group[:, 0]
    point_indices = snap.bonds.group[:, 1]
    distances = system.box.compute_distances(
        system.points[query_point_indices], system.points[point_indices]
    )
    nlist = freud.NeighborList.from_arrays(
        num_query_points, num_points, query_point_indices, point_indices, distances
    )
    cluster = freud.cluster.Cluster()
    cluster.compute(system=system, neighbors=nlist)
    return cluster

def _validate_inputs(gsd_file, snap, gsd_frame):
    if all([gsd_file, snap]):
        raise ValueError("Only pass in one of snapshot, gsd_file")
    if gsd_file:
        assert isinstance(gsd_frame, int)
        try:
            with gsd.hoomd.open(name=gsd_file, mode='rb') as f:
                snap = f[gsd_frame]
        except Exception as e:
            print("Unable to open the gsd_file")
            print(e)
    elif snap:
        assert isinstance(snap, gsd.hoomd.Snapshot)
    return snap
