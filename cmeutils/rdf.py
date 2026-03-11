import freud
import gsd
import gsd.hoomd
import networkx as nx
import numpy as np

from cmeutils.gsd_utils import snapshot_to_graph


def get_rdf(
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
    update_bond_graph=False,
):
    """Uses freud's RDF module to calculate an RDF averaged over a GSD file.


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
    """
    if any([A_name, B_name]) and not all([A_name, B_name]):
        raise ValueError(
            "If A_name or B_name is given, the other must be defined as well."
            "To calculate an RDF between the same bead type, set A_name and B_name equal to the same value."
            "To calculate an RDF between all possible pairs, leave both as ``None``."
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
        # Reuse these for each frame's RDF unless update_bond_graph = True
        # If the bonding topology isn't changing, we only need to get bond graph and excluded pairs once
        rdf_correction = 1
        if exclude_bond_depth:
            max_idx = snap.particles.N
            bond_graph = snapshot_to_graph(snap)
            # This gives a seuqunce of tuples [(1, 4), (5, 8)...(i, j)]
            excluded_pairs = get_excluded_pairs(bond_graph, exclude_bond_depth)
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
            #

            # Create new bond graph and excluded pairs for each frame.
            # Only needed if bond topology is changing, set by ``update_bond_graph``
            if update_bond_graph:
                bond_graph = snapshot_to_graph(snap)
                excluded_pairs = get_excluded_pairs(
                    bond_graph, exclude_bond_depth
                )
                excluded_pairs_encoded = np.array(
                    [i * max_idx + j for i, j in excluded_pairs]
                )
                n_excluded = len(excluded_pairs)

            # Filter excluded pairs from all pairs in nlist
            if exclude_bond_depth:
                nlist = filter_nlist(
                    nlist=nlist,
                    excluded_pairs_encoded=excluded_pairs_encoded,
                    query_indices=type_A_indices,
                    point_indices=type_B_indices,
                    max_idx=snap.particles.N,
                )

            rdf.compute(aq, neighbors=nlist, reset=False)
    return rdf, rdf_correction


def get_excluded_pairs(bond_graph, excluded_bond_depth):
    """Returns a set of (i, j) pairs to exclude based on step distance of a bond graph."""
    # TODO: Vecotrize this instead of doing 2 for loops? Maybe just don't worry about it until there is a reason to
    # Not a big deal if bonds aren't changing as its only done once.
    excluded_pairs = set()
    for i in bond_graph.nodes:
        lengths = nx.single_source_shortest_path_length(
            bond_graph, i, cutoff=excluded_bond_depth
        )
        for j, dist in lengths.items():
            # use j > 1 for 2 reasons: skip adding the same pair twice and its used in filter_nlist
            if j > i and dist <= excluded_bond_depth:
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
