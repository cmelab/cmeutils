from cmeutils import gsd_utils
from cmeutils.gsd_utils import snap_molecule_cluster, _validate_inputs
import freud
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
        self.snap = _validate_inputs(gsd_file, snap, gsd_frame)
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
        """Returns the average of each molecule's end-to-end distance.

        Parameters:
        -----------
        squared : bool, optional, default=False
            Set to True if you want the mean squared average
            end-to-end distance.

        Returns:
        --------
        numpy.array
            The average end-to-end distance averaged over all of the
            molecules in System.molecules

        """
        distances = [m.end_to_end_distance(squared) for m in self.molecules]
        return np.mean(distances)

    def radius_of_gyration_avg(self):
        pass

    def persistence_length_avg(self):
        pass

    def end_to_end_distribution(self):
        pass

    def radius_of_gyration_distribution(self):
        pass

    def bond_length_distribution(self):
        pass

    def bond_angle_distribution(self):
        pass


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
        self.segments = None

    def generate_segments(self, segments_per_molecule):
        segment_indices = np.array_split(
                self.atom_indices,
                segments_per_molecule)
        self.segments = [Segment(self, i) for i in segment_indices]
    
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
            except:
                pass

        assert len(b_vectors) == len(self.monomers) - 1
        return b_vectors

    def persistence_length(self):
        pass


class Monomer(Structure):
    """
    """
    def __init__(self, parent_structure, atom_indices):
        super(Monomer, self).__init__(
                system=parent_structure.system,
                atom_indices=atom_indices
                )
        self.parent = parent_structure
        assert self.n_atoms == self.system.atoms_per_monomer 


class Segment(Structure):
    """
    """
    def __init__(self, molecule, atom_indices):
        super(Segment, self).__init__(system=molecule.system,
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
       pass

