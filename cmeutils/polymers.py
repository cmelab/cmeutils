from cmeutils import gsd_utils
from cmeutils.gsd_utils import snap_molecule_cluster
from cmeutils.plotting import plot_distribution
import freud
import matplotlib.pyplot as plt
import numpy as np

class System:
    """
    """
    def __init__(self,
            atoms_per_monomer,
            gsd_file=None,
            snap=None,
            gsd_frame=-1):
        self.atoms_per_monomer = atoms_per_monomer
        self.snap = gsd_utils._validate_inputs(gsd_file, snap, gsd_frame)
        self.clusters = snap_molecule_cluster(snap=self.snap)
        self.molecule_ids = set(self.clusters)
        self.n_molecules = len(self.molecule_ids)
        self.n_atoms = len(self.clusters)
        self.n_monomers = int(self.n_atoms / self.atoms_per_monomer)
        self.molecules = [Molecule(self, i) for i in self.molecule_ids] 
        self.box = gsd_utils.snap_box(gsd_file, snap, gsd_frame)
        assert len(self.molecules) == self.n_molecules

    def monomers(self):
        """Generate all of the monomers from each molecule
        in System.molecules.

        Yields:
        -------
        polymers.Monomer
     
        """
        for molecule in self.molecules:
            for monomer in molecule.monomers:
                yield monomer

    def segments(self):
        """Generate all of the segments from each molecule
        in System.

        Yields:
        -------
        polymers.Segment

        """
        for molecule in self.molecules:
            for segment in molecule.segments:
                yield segment
    
    def end_to_end_avg(self, squared=False):
        """Returns the end-to-end distance averaged over each
        molecule in the system.

        Parameters:
        -----------
        squared : bool, optional, default=False
            Set to True if you want the mean squared average
            end-to-end distance.

        Returns:
        --------
        numpy.ndarray, shape=(1,), dtype=float
            The average end-to-end distance averaged over all of the
            molecules in System.molecules

        """
        distances = [m.end_to_end_distance(squared) for m in self.molecules]
        return np.mean(distances)

    def radius_of_gyration_avg(self):
        """
        """
        pass

    def persistence_length_avg(self):
        """
        """
        pass

    def end_to_end_distribution(self, nbins, squared=False, plot=False):
        """
        """
        lengths = []
        lengths.extend(
                [mol.end_to_end_distance(squared) for mol in self.molecules]
                )
        if plot:
            plot_distribution(lengths, label="$R_E$", fit_line=True)
        return lengths

    def radius_of_gyration_distribution(self):
        """
        """
        pass

    def bond_length_distribution(
            self,
            normalize=False,
            use_monomers=True,
            use_segments=False,
            nbins='auto',
            plot=False
            ):
        """
        """
        bond_lengths = []
        for molecule in self.molecules:
            bond_lengths.extend(
                    [np.linalg.norm(vec) for vec in molecule.bond_vectors(
                        use_monomers=use_monomers,
                        use_segments=use_segments,
                        normalize=normalize,
                        )
                        ]
                    )
        if plot:
            plot_distribution(bond_lengths, label="Bond Length $(r)$")
        return bond_lengths

    def bond_angle_distribution(
            self,
            use_monomers=True,
            use_segments=False,
            nbins='auto',
            plot=False
            ):
        """
        """
        bond_angles = []
        for molecule in self.molecules:
            bond_angles.extend(molecule.bond_angles(
                use_monomers=use_monomers,
                use_segments=use_segments
                    )
                )

        if plot:
            plot_distribution(bond_angles, label="Bond Angle $(\phi)$")
        return bond_angles


class Structure:
    """Base class for the Molecule(), Segment(), and Monomer() classes.

    Parameters:
    -----------
    system : 'cmeutils.polymers.System', required
        The system object initially created from the input .gsd file.
    atom_indices : np.ndarray(n, 3), optional, default=None
        The atom indices in the system that belong to this specific structure.
    molecule_id : int, optional, default=None
        The ID number of the specific molecule from system.molecule_ids.

    Attributes:
    -----------
    system : 'cmeutils.polymers.System'
        The system that this structure belong to. Contains needed information
        about the box, and gsd snapshot which are used elsewhere.
    atom_indices : np.ndarray(n_atoms, 3)
        The atom indices in the system that belong to this specific structure
    n_atoms : int
        The number of atoms that belong to this specific structure
    atom_positions : np.narray(n_atoms, 3)
        The x, y, z coordinates of the atoms belonging to this structure.
        The positions are wrapped inside the system's box.
    center_of_mass : np.1darray(1, 3)
        The x, y, z coordinates of the structure's center of mass.

    """
    def __init__(self, system, atom_indices=None, molecule_id=None):
        self.system = system
        if molecule_id != None:
            self.atom_indices = np.where(self.system.clusters == molecule_id)[0]
            self.molecule_id = molecule_id
        else:
            self.atom_indices = atom_indices
        self.n_atoms = len(self.atom_indices)

    def generate_monomers(self):
        if isinstance(self, Monomer):
            return self
        structure_length = int(self.n_atoms / self.system.atoms_per_monomer)
        monomer_indices = np.array_split(self.atom_indices, structure_length)
        assert len(monomer_indices) == structure_length
        return [Monomer(self, i) for i in monomer_indices]

    @property
    def atom_positions(self):
        """The wrapped coordinates of every particle in the structure
        as they exist in the periodic box.

        Returns:
        --------
        numpy.ndarray, shape=(n, 3), dtype=float

        """
        return self.system.snap.particles.position[self.atom_indices]

    @property
    def center(self):
        """The (x,y,z) position of the center of the structure accounting
        for the periodic boundaries in the system.
        
        Returns:
        --------
        numpy.ndarray, shape=(3,), dtype=float

        """
        freud_box = freud.Box(
                Lx = self.system.box[0],
                Ly = self.system.box[1],
                Lz = self.system.box[2]
                )
        return freud_box.center_of_mass(self.atom_positions)

    @property 
    def unwrapped_atom_positions(self):
        """The unwrapped coordiantes of every particle in the structure.
        The positions are unwrapped using the images for each particle
        stored in the hoomd snapshot.

        Returns:
        --------
        numpy.ndarray, shape=(n, 3), dtype=float

        """
        images = self.system.snap.particles.image[self.atom_indices]
        return self.atom_positions + (images * self.system.box[:3]) 

    @property
    def unwrapped_center(self):
        """The (x,y,z) position of the center using the structure's
        unwrapped coordiantes.

        Returns:
        --------
        numpy.ndarray, shape=(3,), dtype=float

        """
        x_mean = np.mean(self.unwrapped_atom_positions[:,0])
        y_mean = np.mean(self.unwrapped_atom_positions[:,1])
        z_mean = np.mean(self.unwrapped_atom_positions[:,2])
        return np.array([x_mean, y_mean, z_mean])

class Molecule(Structure):
    """
    """
    def __init__(self, system, molecule_id):
        super(Molecule, self).__init__(system=system, molecule_id=molecule_id)
        self.monomers = self.generate_monomers() 
        self.n_monomers = len(self.monomers)
        self.segments = None
        self.n_segments = None

    def generate_segments(self, monomers_per_segment):
        """
        """
        segments_per_molecule = int(self.n_monomers / monomers_per_segment)
        segment_indices = np.array_split(
                self.atom_indices,
                segments_per_molecule
                )
        self.segments = [Segment(self, i) for i in segment_indices]
        self.n_segments = len(self.segments)
    
    def end_to_end_distance(self, squared=False):
        """Retruns the magnitude of the vector connecting the first and
        last monomer in Molecule.monomers. Uses each monomer's center
        coordinates.

        Parameters:
        -----------
        squared : bool, optional default=False
            Set to True if you want the squared end-to-end distance

        Returns:
        --------
        numpy.ndarray, shape=(1,), dtype=float

        """
        head = self.monomers[0]
        tail = self.monomers[-1]
        distance = np.linalg.norm(
                tail.unwrapped_center - head.unwrapped_center
                )
        if squared:
            distance = distance**2
        return distance
    
    def radius_of_gyration(self):
        pass
    
    def bond_vectors(
            self,
            use_monomers=True,
            use_segments=False,
            normalize=False
            ):
        """Generates a list of the vectors connecting subsequent monomer 
        or segment units.

        Uses the monomer or segment average center coordinates.
        In order to return the bond vectors between segments, the 
        Segment objects need to be created; see the `generate_segments`
        method in the `Molecule` class.

        Parameters:
        -----------
        use_monomers : bool, optional, default=True
            Set to True to return bond vectors between the Molecule's monomers.
        use_segments : bool, optional, default=False
            Set to True to return bond vectors between the Molecule's segments.
        normalize : bool, optional, default=False
            Set to True to normalize each vector by its magnitude.

        Returns:
        --------
        list of numpy.ndarray, shape=(3,), dtype=float

        """
        sub_structures = self._sub_structures(use_monomers, use_segments)

        vectors = []
        for idx, structure in enumerate(sub_structures):
            try:
                next_structure = sub_structures[idx+1]
                vector = (
                        next_structure.unwrapped_center -
                        structure.unwrapped_center
                        )
                if normalize:
                    vector /= np.linalg.norm(vector)
                vectors.append(vector)
            except IndexError:
                pass

        assert len(vectors) == len(sub_structures) - 1
        return vectors

    def bond_angles(
            self,
            bond_vector_list=None,
            use_monomers=True,
            use_segments=False):
        """Generates a list of the angles between subsequent monomer 
        or segment bond vectors.

        Uses the output returned by the `bond_vectors` method
        in the `Molecule` class.
        In order to return the angles between segments, the 
        Segment objects first need to be created; see the
        `generate_segments` method in the `Molecule` class.

        Parameters:
        -----------
        bond_vector_list : array like, optional, default=None
            If None, then the bond vectors are generated by calling the 
            `bond_vectors` method. Or, if already generated, the list
            of vectors can be passed in.
        use_monomers : bool, optional, default=True
            Set to True to return angles between the Molecule's monomers
        use_segments : bool, optional, default=False
            Set to True to return angles between the Molecule's segments

        Returns:
        --------
        list of numpy.ndarray, shape=(3,), dtype=float

        """
        if bond_vector_list is None:
            bond_vector_list = self.bond_vectors(use_monomers, use_segments)

        angles = []
        for idx, vector in enumerate(bond_vector_list):
            try:
                next_vector = bond_vector_list[idx+1]
                cos_angle = (
                        np.dot(vector, next_vector) /
                        (np.linalg.norm(vector) * np.linalg.norm(next_vector))
                        )
                angles.append(np.arccos(cos_angle))
            except IndexError:
                pass

        assert len(angles) == len(bond_vector_list) - 1
        return angles

    def persistence_length(self):
        ""
        ""
        pass

    def _sub_structures(self, use_monomers, use_segments):
        """
        """
        if all((use_monomers, use_segments)):
            raise ValueError(
                    "Only one of `use_monomers` and `use_segments` "
                    "can be set to `True`"
                    )
        if not any((use_monomers, use_segments)):
            raise ValueError(
                    "Set one of `use_monomers` or `use_segments` to "
                    "`True` depending on which structure bond vectors "
                    "you want returned."
                    )

        if use_monomers:
            sub_structures = self.monomers
        elif use_segments:
            if self.segments == None:
                raise ValueError(
                        "The segments for this molecule have not been "
                        "created. See the `generate_segments()` method for "
                        "the `Molecule` class."
                        )
            sub_structures = self.segments
        return sub_structures


class Monomer(Structure):
    """
    """
    def __init__(self, parent_structure, atom_indices):
        super(Monomer, self).__init__(
                system=parent_structure.system,
                atom_indices=atom_indices
                )
        self.parent = parent_structure
        self.components = {}
        assert self.n_atoms == self.system.atoms_per_monomer 
        
    def generate_components(self, index_mapping):
        for name, indices in index_mapping.items():
            component = Component(
                    monomer=self,
                    name=name,
                    atom_indices = self.atom_indices[indices]
                    )
            self.components[name] = component


class Segment(Structure):
    """
    """
    def __init__(self, molecule, atom_indices):
        super(Segment, self).__init__(
                system=molecule.system,
                atom_indices=atom_indices
                )
        self.molecule = molecule
        self.monomers = self.generate_monomers()
        assert len(self.monomers) ==  int(
                self.n_atoms / self.system.atoms_per_monomer
                )

    def end_to_end_distance(self):
        pass
     
    def bond_vectors(self):
        """Generates a list of the vectors connecting subsequent monomer units.
        Uses the monomer's average center coordinates.

        Returns:
        --------
        list of numpy.ndarray, shape=(3,), dtype=float

        """
        b_vectors = []
        for idx, monomer in enumerate(self.monomers):
            try:
                next_monomer = self.monomers[idx+1]
                vector = (
                        next_monomer.unwrapped_center -
                        monomer.unwrapped_center
                        )
                b_vectors.append(vector)
            except IndexError:
                pass

        assert len(b_vectors) == len(self.monomers) - 1
        return b_vectors


class Component(Structure):
    def __init__(self, monomer, atom_indices, name):
        super(Component, self).__init__(
                system=monomer.system,
                atom_indices=atom_indices
                )
        self.name = name
        self.monomer = monomer
        
