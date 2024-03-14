import warnings
from tempfile import NamedTemporaryFile

import freud
import gsd.hoomd
import networkx as nx
import numpy as np
from boltons.setutils import IndexedSet

from cmeutils.geometry import moit


def frame_to_freud_system(frame, ref_length=None):
    """Creates a freud system given a gsd.hoomd.Frame.

    Parameters
    ----------
    frame : gsd.hoomd.Frame, required
        Frame used to get box and particle positions.
    ref_length : float, optional, default None
        Set a reference length to convert from reduced units to real units.
        If None, uses 1 by default.
    """
    if ref_length is None:
        ref_length = 1
    box = frame.configuration.box
    box[0:3] *= ref_length
    xyz = frame.particles.position * ref_length
    return freud.locality.NeighborQuery.from_system(system=(box, xyz))


def get_type_position(
    typename, gsd_file=None, snap=None, gsd_frame=-1, images=False
):
    """Get the positions of a particle type.

    This function returns the positions of a particular particle type from a
    frame of a gsd trajectory file or from a snapshot.
    Pass in either a gsd file or a snapshot, but not both.

    Parameters
    ----------
    typename : str or list of str
        Name of particles of which to get the positions (found in
        gsd.hoomd.Frame.particles.types)
        If you want the positions of multiple types, pass in a list
        e.g., ['ca', 'c3']
    gsd_file : str, default None
        Filename of the gsd trajectory
    snap : gsd.hoomd.Frame, default None
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
    snap : gsd.hoomd.Frame, default None
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
    snap : gsd.hoomd.Frame, default None
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
            with gsd.hoomd.open(name=gsd_file, mode="r") as f:
                snap = f[gsd_frame]
        except Exception as e:
            print("Unable to open the gsd_file")
            raise e
    elif snap:
        assert isinstance(snap, gsd.hoomd.Frame)
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
    snap : gsd.hoomd.Frame
        The snapshot to read in

    Returns
    -------
    gsd.hoomd.Frame
        The new snapshot with particles deleted.
    """
    new_snap = gsd.hoomd.Frame()
    delete_ids = [snap.particles.types.index(i) for i in delete_types]
    selection = np.where(~np.isin(snap.particles.typeid, delete_ids))[0]
    new_snap.particles.N = len(selection)
    new_snap.particles.types = [
        i for i in snap.particles.types if i not in delete_types
    ]
    typeid_map = {
        i: new_snap.particles.types.index(e)
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
            inds = {e: i for i, e in enumerate(selection)}
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
    snapshot : gsd.hoomd.Frame
        The snapshot returned from create_hoomd_forcefield
        or create_hoomd_simulation in mBuild
    mb_compound : mbuild.Compound, required
        mBuild compound containing the rigid body information
        of the complete system

    """
    import hoomd

    rigid_ids = [p.rigid_id for p in mb_compound.particles()]
    rigid_bodies = set(rigid_ids)
    # Total number of rigid body particles
    N_mols = len(rigid_bodies)
    N_p = [rigid_ids.count(i) for i in rigid_bodies]
    # Right now, we're assuming each molecule has the same num of particles
    assert len(set(N_p)) == 1
    N_p = N_p[0]  # Number of particles per molecule
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

    rigid.body["R"] = {
        "constituent_types": c_types,
        "positions": c_pos,
        "orientations": c_orient,
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
    with gsd.hoomd.open(new_file, "w") as new_t:
        with gsd.hoomd.open(gsd_file) as old_t:
            for snap in old_t:
                snap.particles.type_shapes = [
                    {"type": "Ellipsoid", "a": lpar, "b": lperp, "c": lperp},
                    {"type": "Sphere", "diameter": 0.01},
                    {"type": "Sphere", "diameter": 0.01},
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
            overwrite=True,
        )
        hoomd.util.unquiet_status()
        with gsd.hoomd.open(f.name) as t, gsd.hoomd.open(gsdfile, "w") as newt:
            snap = t[0]
            bonds = snap.bonds.group
            bonds = bonds[np.lexsort((bonds[:, 1], bonds[:, 0]))]
            snap.bonds.group = bonds
            newt.append(snap)
    print(f"XML data written to {gsdfile}")


def trim_snapshot_molecules(parent_snapshot, mol_indices):
    """Given a snapshot of a system, trim the snapshot to only include
    a subset of the molecules.

    Parameters
    ----------
    parent_snapshot : gsd.hoomd.Frame
        The snapshot to read in.
    mol_indices : list of np.ndarray
        List of arrays where each array contains the indices
        of the particles in a molecule to include.

    Returns
    -------
    gsd.hoomd.Frame
        The new snapshot with only the specified molecules.

    Notes
    -----
    See cmetuils.gsd_utils.get_molecule_cluster for a method to obtain
    mol_indices.

    """
    new_snap = gsd.hoomd.Frame()
    new_snap.configuration.box = parent_snapshot.configuration.box
    new_snap.particles.N = sum(len(i) for i in mol_indices)

    # Write out particle info
    for attr in ["position", "mass", "velocity", "orientation", "image", "diameter", "angmom", "typeid"]:
        setattr(
            new_snap.particles,
            attr,
            np.concatenate(
                list(getattr(parent_snapshot.particles, attr)[i] for i in mol_indices)
            )
        )
    new_snap.particles.types = parent_snapshot.particles.types

    particle_index_map = dict()
    count = 0
    for indices in mol_indices:
        for i in indices:
            particle_index_map[i] = count
            count += 1

    # Write out bond info
    mol_bond_groups = []
    mol_bond_ids = []
    for count, indices in enumerate(mol_indices):
        mask = np.any(np.isin(parent_snapshot.bonds.group, indices.flatten()), axis=1)
        parent_mol_bonds = parent_snapshot.bonds.group[np.where(mask)[0]]
        parent_mol_bond_typeids = parent_snapshot.bonds.typeid[np.where(mask)[0]]
        new_mol_bonds = np.vectorize(particle_index_map.get)(parent_mol_bonds)
        mol_bond_groups.append(new_mol_bonds)
        mol_bond_ids.append(parent_mol_bond_typeids)

    new_snap.bonds.types = parent_snapshot.bonds.types
    new_snap.bonds.group = np.concatenate(mol_bond_groups)
    new_snap.bonds.typeid = np.concatenate(mol_bond_ids)
    new_snap.bonds.N = sum(len(i) for i in mol_bond_ids)

    # Write out angle info
    mol_angle_groups = []
    mol_angle_ids = []
    for count, indices in enumerate(mol_indices):
        mask = np.any(np.isin(parent_snapshot.angles.group, indices.flatten()), axis=1)
        parent_mol_angles = parent_snapshot.angles.group[np.where(mask)[0]]
        parent_mol_angle_typeids = parent_snapshot.angles.typeid[np.where(mask)[0]]
        new_mol_angles = np.vectorize(particle_index_map.get)(parent_mol_angles)
        mol_angle_groups.append(new_mol_angles)
        mol_angle_ids.append(parent_mol_angle_typeids)

    new_snap.angles.types = parent_snapshot.angles.types
    new_snap.angles.group = np.concatenate(mol_angle_groups)
    new_snap.angles.typeid = np.concatenate(mol_angle_ids)
    new_snap.angles.N = sum(len(i) for i in mol_angle_ids)

    # Write out dihedral info
    mol_dihedral_groups = []
    mol_dihedral_ids = []
    for count, indices in enumerate(mol_indices):
        mask = np.any(np.isin(parent_snapshot.dihedrals.group, indices.flatten()), axis=1)
        parent_mol_dihedrals = parent_snapshot.dihedrals.group[np.where(mask)[0]]
        parent_mol_dihedral_typeids = parent_snapshot.dihedrals.typeid[np.where(mask)[0]]
        new_mol_dihedrals = np.vectorize(particle_index_map.get)(parent_mol_dihedrals)
        mol_dihedral_groups.append(new_mol_dihedrals)
        mol_dihedral_ids.append(parent_mol_dihedral_typeids)

    new_snap.dihedrals.types = parent_snapshot.dihedrals.types
    new_snap.dihedrals.group = np.concatenate(mol_dihedral_groups)
    new_snap.dihedrals.typeid = np.concatenate(mol_dihedral_ids)
    new_snap.dihedrals.N = sum(len(i) for i in mol_dihedral_ids)

    new_snap.validate()
    return new_snap

def identify_snapshot_connections(snapshot):
    """Identify angle and dihedral connections in a snapshot from bonds.

    Parameters
    ----------
    snapshot : gsd.hoomd.Frame
        The snapshot to read in.

    Returns
    -------
    gsd.hoomd.Frame
        The snapshot with angle and dihedral information added.
    """
    if snapshot.bonds.N == 0:
        warnings.warn(
            "No bonds found in snapshot, hence, no angles or "
            "dihedrals will be identified."
        )
        return snapshot
    bond_groups = snapshot.bonds.group
    connection_matches = _find_connections(bond_groups)

    if connection_matches["angles"]:
        _fill_connection_info(
            snapshot=snapshot,
            connections=connection_matches["angles"],
            type_="angles",
        )
    if connection_matches["dihedrals"]:
        _fill_connection_info(
            snapshot=snapshot,
            connections=connection_matches["dihedrals"],
            type_="dihedrals",
        )
    return snapshot


def _fill_connection_info(snapshot, connections, type_):
    p_types = snapshot.particles.types
    p_typeid = snapshot.particles.typeid
    _connection_types = []
    _connection_typeid = []
    for conn in connections:
        conn_sites = [p_types[p_typeid[i]] for i in conn]
        sorted_conn_sites = _sort_connection_by_name(conn_sites, type_)
        type = "-".join(sorted_conn_sites)
        # check if type not in angle_types and types_inv not in angle_types:
        if type not in _connection_types:
            _connection_types.append(type)
            _connection_typeid.append(
                max(_connection_typeid) + 1 if _connection_typeid else 0
            )
        else:
            _connection_typeid.append(_connection_types.index(type))

    if type_ == "angles":
        snapshot.angles.N = len(connections)
        snapshot.angles.M = 3
        snapshot.angles.group = connections
        snapshot.angles.types = _connection_types
        snapshot.angles.typeid = _connection_typeid
    elif type_ == "dihedrals":
        snapshot.dihedrals.N = len(connections)
        snapshot.dihedrals.M = 4
        snapshot.dihedrals.group = connections
        snapshot.dihedrals.types = _connection_types
        snapshot.dihedrals.typeid = _connection_typeid


# The following functions are obtained from gmso/utils/connectivity.py with
# minor modifications.
def _sort_connection_by_name(conn_sites, type_):
    if type_ == "angles":
        site1, site3 = sorted([conn_sites[0], conn_sites[2]])
        return [site1, conn_sites[1], site3]
    elif type_ == "dihedrals":
        site1, site2, site3, site4 = conn_sites
        if site2 > site3 or (site2 == site3 and site1 > site4):
            return [site4, site3, site2, site1]
        else:
            return [site1, site2, site3, site4]


def _find_connections(bonds):
    """Identify all possible connections within a topology."""
    compound = nx.Graph()

    for b in bonds:
        compound.add_edge(b[0], b[1])

    compound_line_graph = nx.line_graph(compound)

    angle_matches = _detect_connections(compound_line_graph, type_="angle")
    dihedral_matches = _detect_connections(
        compound_line_graph, type_="dihedral"
    )

    return {
        "angles": angle_matches,
        "dihedrals": dihedral_matches,
    }


def _detect_connections(compound_line_graph, type_="angle"):
    EDGES = {
        "angle": ((0, 1),),
        "dihedral": ((0, 1), (1, 2)),
    }

    connection = nx.Graph()
    for edge in EDGES[type_]:
        assert len(edge) == 2, "Edges should be of length 2"
        connection.add_edge(edge[0], edge[1])

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        compound_line_graph, connection
    )

    formatter_fns = {
        "angle": _format_subgraph_angle,
        "dihedral": _format_subgraph_dihedral,
    }

    conn_matches = IndexedSet()
    for m in matcher.subgraph_isomorphisms_iter():
        new_connection = formatter_fns[type_](m)
        conn_matches.add(new_connection)
    if conn_matches:
        conn_matches = _trim_duplicates(conn_matches)

    # Do more sorting of individual connection
    sorted_conn_matches = list()
    for match in conn_matches:
        if match[0] < match[-1]:
            sorted_conn = match
        else:
            sorted_conn = match[::-1]
        sorted_conn_matches.append(list(sorted_conn))

    # Final sorting the whole list
    if type_ == "angle":
        return sorted(
            sorted_conn_matches,
            key=lambda angle: (
                angle[1],
                angle[0],
                angle[2],
            ),
        )
    elif type_ == "dihedral":
        return sorted(
            sorted_conn_matches,
            key=lambda dihedral: (
                dihedral[1],
                dihedral[2],
                dihedral[0],
                dihedral[3],
            ),
        )


def _get_sorted_by_n_connections(m):
    """Return sorted by n connections for the matching graph."""
    small = nx.Graph()
    for k, v in m.items():
        small.add_edge(k[0], k[1])
    return sorted(small.adj, key=lambda x: len(small[x])), small


def _format_subgraph_angle(m):
    """Format the angle subgraph.

    Since we are matching compound line graphs,
    back out the actual nodes, not just the edges

    Parameters
    ----------
    m : dict
        keys are the compound line graph nodes
        Values are the sub-graph matches (to the angle, dihedral, or improper)

    Returns
    -------
    connection : list of nodes, in order of bonding
        (start, middle, end)
    """
    (sort_by_n_connections, _) = _get_sorted_by_n_connections(m)
    ends = sorted([sort_by_n_connections[0], sort_by_n_connections[1]])
    middle = sort_by_n_connections[2]
    return (
        ends[0],
        middle,
        ends[1],
    )


def _format_subgraph_dihedral(m):
    """Format the dihedral subgraph.

    Since we are matching compound line graphs,
    back out the actual nodes, not just the edges

    Parameters
    ----------
    m : dict
        keys are the compound line graph nodes
        Values are the sub-graph matches (to the angle, dihedral, or improper)
    top : gmso.Topology
        The original Topology

    Returns
    -------
    connection : list of nodes, in order of bonding
        (start, mid1, mid2, end)
    """
    (sort_by_n_connections, small) = _get_sorted_by_n_connections(m)
    start = sort_by_n_connections[0]
    if sort_by_n_connections[2] in small.neighbors(start):
        mid1 = sort_by_n_connections[2]
        mid2 = sort_by_n_connections[3]
    else:
        mid1 = sort_by_n_connections[3]
        mid2 = sort_by_n_connections[2]

    end = sort_by_n_connections[1]
    return (start, mid1, mid2, end)


def _trim_duplicates(all_matches):
    """Remove redundant sub-graph matches.

    Is there a better way to do this? Like when we format the subgraphs,
    can we impose an ordering so it's easier to eliminate redundant matches?
    """
    trimmed_list = IndexedSet()
    for match in all_matches:
        if (
            match
            and match not in trimmed_list
            and match[::-1] not in trimmed_list
        ):
            trimmed_list.add(match)
    return trimmed_list
