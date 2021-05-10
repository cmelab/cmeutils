from cmeutils.gsd_utils import snap_molecule_cluster

class System:
    """
    """
    def __init__(self, gsd_file=None, snap=None, gsd_frame=-1):
        self.clusters = snap.molecule_cluster(gsd_file, snap, gsd_frame)
        self.molecule_ids = set(self.clusters)
        self.n_compounds = len(self.molecule_ids)
        self.n_atoms = len(self.molecule_ids)
        self.molecules = None
        self.monomers = None
        self.segments = None

    def generate_molecules(self):
        self.molecules = [Molecule(self, i) for i in self.molecule_ids]
        assert len(self.molecules) == self.n_compounds

    def generate_monomers(self, monomers_per_molecule):
        pass

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
        self.molecule_id = molecule_id
        self.atom_indices = np.where(system.clusters == self.molecule_id)[0]
        self.n_atoms = len(self.atom_indices)
        self.monomers = None
        self.segments = None

    def monomers(self, monomers_per_molecule):
        monomer_indices = np.array_split(self.atom_indices,
                monomers_per_molecule
                )
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
        self.atom_indices = atom_indices
        self.n_atoms = len(self.atom_indices)
        

    def center_of_mass(self):
        pass


class Segment:
    def __init__(self, molecule, atom_indices):
        self.molecule = molecule
        self.atom_indices = atom_indices
        self.n_atoms = len(self.atom_indices)

    def center_of_mass(self):
        pass

    def end_to_end_distance(self):
        pass

