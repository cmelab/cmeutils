from tempfile import NamedTemporaryFile

import freud
import gsd.hoomd
import hoomd
import numpy as np

from cmeutils.geometry import moit


def get_type_position(
    typename,
    gsd_file=None,
    snap=None,
    gsd_frame=-1,
    images=False
):
    """Get the positions of a particle type.

    This function returns the positions of a particular particle type from a
    frame of a gsd trajectory file or from a snapshot.
    Pass in either a gsd file or a snapshot, but not both.

    Parameters
    ----------
    typename : str or list of str
        Name of particles of which to get the positions (found in
        gsd.hoomd.Snapshot.particles.types)
        If you want the positions of multiple types, pass in a list
        e.g., ['ca', 'c3']
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
        Returns an array of positions or arrays of positions and images
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
    """Return all particle types in a hoomd trajectory.

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


def get_molecule_cluster(gsd_file=None, snap=None, gsd_frame=-1):
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
    numpy.ndarray (N_particles,)
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
        raise ValueError("Only pass in a snapshot or a gsd_file")
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


def snap_delete_types(snap, delete_types):
    """Create a new snapshot with certain particle types deleted.

    Reads in a snapshot and writes the information (excluding delete_types) to
    a new snapshot. Does not change the original snapshot.

    Information written:
        - particles (N, types, position, typeid, image)
        - configuration (box)
        - bonds (N, group)

    Parameters
    ----------
    snap : gsd.hoomd.Snapshot
        The snapshot to read in

    Returns
    -------
    gsd.hoomd.Snapshot
        The new snapshot with particles deleted.
    """
    new_snap = gsd.hoomd.Snapshot()
    delete_ids = [snap.particles.types.index(i) for i in delete_types]
    selection = np.where(~np.isin(snap.particles.typeid, delete_ids))[0]
    new_snap.particles.N = len(selection)
    new_snap.particles.types = [
        i for i in snap.particles.types if i not in delete_types
    ]
    typeid_map = {
        i:new_snap.particles.types.index(e)
        for i, e in enumerate(snap.particles.types)
        if e in new_snap.particles.types
    }
    new_snap.particles.position = snap.particles.position[selection]
    new_snap.particles.image = snap.particles.image[selection]
    new_snap.particles.typeid = np.vectorize(typeid_map.get)(
        snap.particles.typeid[selection]
    )
    new_snap.configuration.box = snap.configuration.box
    if snap.bonds.N > 0:
        bonds = np.isin(snap.bonds.group, selection).all(axis=1)
        if bonds.any():
            inds = {e:i for i, e in enumerate(selection)}
            new_snap.bonds.group = np.vectorize(inds.get)(
                snap.bonds.group[bonds]
            )
            new_snap.bonds.N = len(new_snap.bonds.group)
    new_snap.validate()
    return new_snap


def create_rigid_snapshot(mb_compound):
    """Preps a hoomd snapshot to store rigid body information

    This method relies on using built-in mBuild methods to
    create the rigid body information.
    
    Parameters
    ----------
    mb_compound : mbuild.Compound, required
        mBuild compound containing the rigid body information
        of the complete system

    Returns
    -------
    hoomd.Snapshot
        A snapshot that contains the needed place-holder particles
        to set up a rigid body simulation in HoomdBlue Version 3.
        Pass this snapshot into the `init_snap` parameter
        of mBuild's `create_hoomd_forcefield` function.

    """
    import hoomd

    rigid_ids = [p.rigid_id for p in mb_compound.particles()]
    rigid_bodies = set(rigid_ids)
    # Total number of rigid particles
    N_mols = len(rigid_bodies)
    init_snap = hoomd.Snapshot()
    # Create place holder spots in the snapshot for rigid centers
    init_snap.particles.types = ["R"]
    init_snap.particles.N = N_mols
    return init_snap


def update_rigid_snapshot(snapshot, mb_compound):
    """Update a snapshot prepared for rigid bodies with system informaiton

    Parameters
    ----------
    snapshot : gsd.hoomd.Snapshot
        The snapshot returned from create_hoomd_forcefield
        or create_hoomd_simulation in mBuild
    mb_compound : mbuild.Compound, required
        mBuild compound containing the rigid body information
        of the complete system

    """
    rigid_ids = [p.rigid_id for p in mb_compound.particles()]
    rigid_bodies = set(rigid_ids)
    # Total number of rigid body particles
    N_mols = len(rigid_bodies)
    N_p =  [rigid_ids.count(i) for i in rigid_bodies]
	# Right now, we're assuming each molecule has the same num of particles
    assert len(set(N_p)) == 1
    N_p = N_p[0] # Number of particles per molecule
    mol_inds = [
        np.arange(N_mols + i * N_p, N_mols + i * N_p + N_p)
        for i in range(N_mols)
    ]

    for i, inds in enumerate(mol_inds):
	    total_mass = np.sum(snapshot.particles.mass[inds])
	    com = (
		    np.sum(
			    snapshot.particles.position[inds]
			    * snapshot.particles.mass[inds, np.newaxis],
                axis=0,
		    )
		    / total_mass
	    )
	    snapshot.particles.position[i] = com
	    snapshot.particles.body[i] = i
	    snapshot.particles.body[inds] = i * np.ones_like(inds)
	    snapshot.particles.mass[i] = np.sum(snapshot.particles.mass[inds])
	    snapshot.particles.moment_inertia[i] = moit(
		    snapshot.particles.position[inds],
		    snapshot.particles.mass[inds],
		    center=com,
	    )

    rigid = hoomd.md.constrain.Rigid()
    inds = mol_inds[0]
    r_pos = snapshot.particles.position[0]
    c_pos = snapshot.particles.position[inds]
    c_pos -= r_pos
    c_pos = [tuple(i) for i in c_pos]
    c_types = [
            snapshot.particles.types[i] for i in snapshot.particles.typeid[inds]
    ]
    c_orient = [tuple(i) for i in snapshot.particles.orientation[inds]]
    c_charge = [i for i in snapshot.particles.charge[inds]]
    c_diam = [i for i in snapshot.particles.diameter[inds]]

    rigid.body["R"] = {
            "constituent_types": c_types,
            "positions": c_pos,
            "charges": c_charge,
            "orientations": c_orient,
            "diameters": c_diam,
    } 
    return snapshot, rigid


def ellipsoid_gsd(gsd_file, new_file, lpar, lperp):
    """Add needed information to GSD file to visualize ellipsoids.

    Saves a new GSD file with lpar and lperp values populated
    for each particle. Ovito can be used to visualize the new GSD file. 
	
    Parameters
    ----------
    gsd_file : str 
        Path to the original GSD file containing trajectory information
    new_file : str
        Path and filename of the new GSD file
    lpar : float
        Value of lpar of the ellipsoids
    lperp : float
        Value of lperp of the ellipsoids
	
	"""
    with gsd.hoomd.open(new_file, "wb") as new_t:
        with gsd.hoomd.open(gsd_file) as old_t:
            for snap in old_t:
                snap.particles.type_shapes = [
                    {
                        "type": "Ellipsoid",
                        "a": lpar,
                        "b": lperp,
                        "c": lperp
                    },
                    {
                        "type": "Sphere",
                        "diameter": 0.01
                    },
                                  {
                        "type": "Sphere",
                        "diameter": 0.01
                    }
                ]
                snap.validate()
                new_t.append(snap)


def xml_to_gsd(xmlfile, gsdfile):
    """Writes hoomdxml data to gsd file.

    Assumes xml only contains one frame. Also sorts bonds so they will work with
    freud Neighborlist. Overwrites gsd file if it exists.

    Parameters
    ----------
    xmlfile : str
        Path to xml file
    gsdfile : str
        Path to gsd file
    """
    try:
        import hoomd
        import hoomd.deprecated
    except ImportError:
        raise ImportError(
                "You must have hoomd version 2 installed to use xml_to_gsd()"
        )

    hoomd.util.quiet_status()
    hoomd.context.initialize("")
    hoomd.deprecated.init.read_xml(xmlfile, restart=xmlfile)
    with NamedTemporaryFile() as f:
        hoomd.dump.gsd(
            filename=f.name,
            period=None,
            group=hoomd.group.all(),
            dynamic=["momentum"],
            overwrite=True
        )
        hoomd.util.unquiet_status()
        with gsd.hoomd.open(f.name) as t, gsd.hoomd.open(gsdfile, "wb") as newt:
            snap = t[0]
            bonds = snap.bonds.group
            bonds = bonds[np.lexsort((bonds[:,1], bonds[:,0]))]
            snap.bonds.group = bonds
            newt.append(snap)
    print(f"XML data written to {gsdfile}")
