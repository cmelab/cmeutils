import gsd
import gsd.hoomd
import gsd.pygsd


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
