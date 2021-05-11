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
        self.molecules = self.generate_molecules() 
        self.monomers = None
        self.segments = None
        self.box = gsd_utils.snap_box(gsd_file, snap, gsd_frame)
        assert len(self.molecules) == self.n_molecules

    def generate_molecules(self):
        molecules = [Molecule(self, i) for i in self.molecule_ids]
        return molecules

    def generate_monomers(self):
        # Call the monomers func in Molecule instead?
        monomers = [Monomer(self, mol)for mol in self.molecules]
        return monomers

    def generate_segments(self,
            atoms_per_segment=None,
            monomers_per_segment=None):
        pass
    
    def end_to_end_average(self):
        pass

    def radius_of_gyration_average(self):
        pass

    def persistance_length_average(self):
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

    @property
    def atom_positions(self):
        return self.system.snap.particles.position[self.atom_indices]

    @property 
    def unwrapped_atom_positions(self):
        images = snap.particles.image[self.atom_indices]
        return snap.atom_positions + (images * self.system.box[:3]) 

    @property
    def center_of_mass(self):
        freud_box = freud.Box(
                Lx = self.system.box[0],
                Ly = self.system.box[1],
                Lz = self.system.box[2]
                )
        return freud_box.center_of_mass(self.atom_positions)


class Molecule(Structure):
    """
    """
    def __init__(self, system, molecule_id):
        super(Molecule, self).__init__(system=system, molecule_id=molecule_id)
        self.monomers = self.generate_monomers() 
        self.segments = None

    def generate_monomers(self):
        atoms_per_monomer = self.system.atoms_per_monomer
        molecule_length = int(self.n_atoms / atoms_per_monomer)
        monomer_indices = np.array_split(
                self.atom_indices,
                molecule_length
                )
        assert len(monomer_indices) == molecule_length
        return [Monomer(self, i) for i in monomer_indices]

    def generate_segments(self, segments_per_molecule):
        segment_indices = np.array_split(
                self.atom_indices,
                segments_per_molecule)
        self.segments = [Segment(self, i) for i in segment_indices]
    
    @property
    def end_to_end_distance(self):
        pass
    
    @property
    def radius_of_gyration(self):
        pass
    
    def bond_vectors(self):
        pass

    def persistance_length(self):
        pass


class Monomer(Structure):
    """
    """
    def __init__(self, molecule, atom_indices):
        super(Monomer, self).__init__(system=molecule.system,
                atom_indices=atom_indices
                )
        self.molecule = molecule
        assert self.n_atoms == self.system.atoms_per_monomer 


class Segment(Structure):
    """
    """
    def __init__(self, molecule, atom_indices):
        super(Segment, self).__init__(system=molecule.system,
                atom_indices=atom_indices
                )
        self.molecule = molecule
        self.n_monomers = int(self.n_atoms / self.system.atoms_per_monomer)

    def end_to_end_distance(self):
        pass
     
    def bond_vectors(self):
       pass

