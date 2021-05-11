from cmeutils.gsd_utils import snap_molecule_cluster

class System:
    """
    """
    def __init__(self,
            atoms_per_monomer,
            gsd_file=None,
            snap=None,
            gsd_frame=-1):
        self.atoms_per_monomer = atoms_per_monomer
        self.clusters = snap.molecule_cluster(gsd_file, snap, gsd_frame)
        self.molecule_ids = set(self.clusters)
        self.n_compounds = len(self.molecule_ids)
        self.n_atoms = len(self.clusters)
        self.n_monomers = int(self.n_compounds / self.atoms_per_monomer)
        assert self.n_atoms == (self.n_monomers * self.atoms_per_monomer)
        self.molecules = self.generate_molecules() 
        self.monomers = None
        self.segments = None

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

    def end_to_end_distribution(self):
        pass

    def bond_length_distribution(self):
        pass

    def bond_angle_distribution(self):
        pass

    def radius_of_gyration_distribution(self):
        pass


class Molecule:
    """
    """
    def __init__(self, system, molecule_id):
        self.system = system
        self.molecule_id = molecule_id # Need this?
        self.atom_indices = np.where(system.clusters == self.molecule_id)[0]
        self.n_atoms = len(self.atom_indices)
        self.monomers = None
        self.segments = None

    def monomers(self):
        atoms_per_monomer = self.system.atoms_per_monomer
        molecule_length = int(self.n_atoms / atoms_per_monomer)
        monomer_indices = np.array_split(self.atom_indices,
                molecule_length
                )
        assert len(monomer_indices) == molecule_length
        self.monomers = [Monomer(self, i) for i in monomer_indices]

    def segments(self, segments_per_molecule):
        segment_indices = np.array(split(self.atom_indices,
            segments_per_molecule)
            )
        self.segments = [Segment(self, i) for i in segment_indices]


    def center_of_mass(self):
        pass

    def end_to_end_distance(self):
        pass
    
    def radius_of_gyration(self):
        pass

    def bond_vectors(self):
        pass


class Monomer:
    def __init__(self, molecule, atom_indices):
        self.molecule = molecule
        self.system = molecule.system
        self.atom_indices = atom_indices
        self.n_atoms = len(self.atom_indices)
        assert self.n_atoms == self.system.atoms_per_monomer 

    def center_of_mass(self):
        pass


class Segment:
    def __init__(self, molecule, atom_indices):
        self.molecule = molecule
        self.system = molecule.system
        self.atom_indices = atom_indices
        self.n_atoms = len(self.atom_indices)

    def center_of_mass(self):
        pass

    def end_to_end_distance(self):
        pass
   
   def bond_vectors(self):
       pass
