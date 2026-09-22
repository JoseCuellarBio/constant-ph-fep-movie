import numpy as np
import math
import random


class Protein:
    def __init__(self, source, coords=None, list_charged_residues=None, pH=7.0,
                 use_polar=True, use_electrostatic=True):

        self.type_dict = {
            'ALA': 'N', 'ARG': 'B', 'ASN': 'P', 'ASP': 'A', 'CYS': 'A', 'GLU': 'A',
            'GLN': 'P', 'GLY': 'G', 'HIS': 'B', 'ILE': 'N', 'LEU': 'N', 'LYS': 'B',
            'MET': 'N', 'PHE': 'N', 'PRO': 'N', 'SER': 'P', 'THR': 'P', 'TRP': 'N',
            'TYR': 'A', 'VAL': 'N', 'NTR': 'B', 'CTR': 'A',
            'ACE': 'C', 'NME': 'C'
        }

        self.list_charged_residues = list_charged_residues or []

        # ---- Detect input type ----
        if isinstance(source, str):
            self.data = self._parse_pdb(source)

        elif isinstance(source, list) and coords is not None:
            # OpenAWSEM input
            self.data = self._parse_openawsem(source, coords)
            self._row_atom_refs = None

        else:
            # dataframe
            self.data = self._parse_dataframe(source)
            self._build_row_atom_refs(source)

        if not self.list_charged_residues:
            self._generate_provisional_charges()

        self.mc = MonteCarloResidue(self)
        self.neighborhood = Neighborhood(self)
        self.protonation_mc = ProtonationMC(
            self, pH=pH, use_polar=use_polar,
            use_electrostatic=use_electrostatic
        )
        self._build_representative_cache()

    # -----------------------------------

    def _generate_provisional_charges(self):
        charges = []

        for resid, info in self.data.items():

            if info["type"] == "A":
                info["charge"] = -1.0
                charges.append((resid, -1.0))

            elif info["type"] == "B":
                info["charge"] = 1.0
                charges.append((resid, 1.0))

            else:
                info["charge"] = 0.0

        self.list_charged_residues = charges

    # -----------------------------------

    def _parse_openawsem(self, target_atoms_info, coords):

        data = {}
        charge_map = dict(self.list_charged_residues)

        for (atom_index, atom_name, resid, resname), pos in zip(target_atoms_info, coords):

            x, y, z = pos

            if resid not in data:

                data[resid] = {
                    "resname": resname,
                    "type": self.type_dict.get(resname, "X"),
                    "charge": charge_map.get(resid, 0.0),
                    "atoms": {}
                }

            data[resid]["atoms"][atom_name] = {
                "coords": [float(x), float(y), float(z)],
                "index": atom_index
            }

        return data
    # -----------------------------------

    def _parse_dataframe(self, rows):

        data = {}
        charge_map = dict(self.list_charged_residues)

        for row in rows:

            resid = int(row["residue_number"])
            resname = row["residue_name"]
            atom = row["atom"]
            x, y, z = row["position_xyz"]

            if resid not in data:

                data[resid] = {
                    "resname": resname,
                    "type": self.type_dict.get(resname, "X"),
                    "charge": charge_map.get(resid, 0.0),
                    "atoms": {}
                }

            data[resid]["atoms"][atom] = {
                "coords": [float(x), float(y), float(z)]
                }

        return data

    def _build_row_atom_refs(self, rows):
        self._row_atom_refs = []
        for atom_index, row in enumerate(rows):
            resid = int(row["residue_number"])
            atom = row["atom"]
            atom_info = self.data[resid]["atoms"][atom]
            atom_info["index"] = atom_index
            self._row_atom_refs.append(
                (resid, row["residue_name"], atom, atom_info)
            )

    def _get_representative_atom(self, info):
        for atom_name in ("CB", "CA", "O"):
            if atom_name in info["atoms"]:
                return atom_name
        return None

    def _build_representative_cache(self):
        representative_refs = []
        representative_resids = []
        representative_resid_to_index = {}

        for resid, info in self.data.items():
            atom_name = self._get_representative_atom(info)
            if atom_name is None:
                continue
            atom_info = info["atoms"][atom_name]
            representative_resid_to_index[resid] = len(representative_refs)
            representative_resids.append(resid)
            representative_refs.append(atom_info)

        self._representative_refs = representative_refs
        self._representative_resids = representative_resids
        self._representative_resid_to_index = representative_resid_to_index
        self._charge_resids = [resid for resid, _ in self.list_charged_residues]
        self._geometry_version = 0
        self._refresh_representative_arrays()

    def _refresh_representative_arrays(self):
        self.representative_coords = np.asarray(
            [atom_info["coords"] for atom_info in self._representative_refs],
            dtype=float,
        )
        self.representative_charges = np.asarray(
            [self.data[resid]["charge"] for resid in self._representative_resids],
            dtype=float,
        )
        self.representative_types = np.asarray(
            [self.data[resid]["type"] for resid in self._representative_resids],
            dtype=object,
        )

    def update_from_rows(self, rows):
        if self._row_atom_refs is None:
            raise ValueError("In-place updates are only available for row-based frames.")
        if len(rows) != len(self._row_atom_refs):
            raise ValueError("The frame does not match the initial topology.")

        for row, (expected_resid, expected_resname, expected_atom, atom_info) in zip(rows, self._row_atom_refs):
            resid = int(row["residue_number"])
            resname = row["residue_name"]
            atom = row["atom"]
            if resid != expected_resid or resname != expected_resname or atom != expected_atom:
                raise ValueError("The frame does not match the initial topology.")
            x, y, z = row["position_xyz"]
            atom_info["coords"] = [float(x), float(y), float(z)]
        self._refresh_representative_arrays()
        self._geometry_version += 1

    def refresh_charge_state(self, charged_residues):
        self.list_charged_residues = charged_residues
        for resid, charge in charged_residues:
            if resid in self.data:
                self.data[resid]["charge"] = charge
            representative_index = self._representative_resid_to_index.get(resid)
            if representative_index is not None:
                self.representative_charges[representative_index] = charge

class MonteCarloResidue:
    def __init__(self, protein):
        self.protein = protein
        self.current_resid = None

    def _get_charged_resids(self):
        return self.protein._charge_resids

    def choose_residue(self):
        charged = self._get_charged_resids()
        self.current_resid = random.choice(charged) if charged else None
        return self.current_resid

    def get_current_residue_info(self):
        if self.current_resid is None:
            return None
        return {self.current_resid: self.protein.data[self.current_resid]}

class Neighborhood:
    def __init__(self, protein, cutoff=6.0):
        self.protein = protein
        self.cutoff = cutoff
        self._geometry_version = -1
        self._distance_rows = {}
        self._masks = {}

    def _prepare_frame(self):
        if self._geometry_version == self.protein._geometry_version:
            return
        coords = self.protein.representative_coords
        center_indices = [
            self.protein._representative_resid_to_index[resid]
            for resid in self.protein._charge_resids
            if resid in self.protein._representative_resid_to_index
        ]
        if center_indices:
            displacement = (
                coords[np.asarray(center_indices), np.newaxis, :]
                - coords[np.newaxis, :, :]
            )
            distances = np.linalg.norm(displacement, axis=2)
            self._distance_rows = dict(zip(center_indices, distances))
        else:
            self._distance_rows = {}
        self._masks.clear()
        self._geometry_version = self.protein._geometry_version

    def get_neighbor_data(self, center_resid):
        self._prepare_frame()
        center_index = self.protein._representative_resid_to_index.get(center_resid)
        if center_index is None:
            return None

        coords = self.protein.representative_coords
        if coords.size == 0 or len(coords) <= 1:
            return None

        mask = self._masks.get(center_index)
        if mask is None:
            mask = np.ones(len(coords), dtype=bool)
            mask[center_index] = False
            self._masks[center_index] = mask
        distance_row = self._distance_rows.get(center_index)
        if distance_row is None:
            distance_row = np.linalg.norm(coords - coords[center_index], axis=1)
            self._distance_rows[center_index] = distance_row
        distances = distance_row[mask]
        neighbor_charges = self.protein.representative_charges[mask]
        neighbor_types = self.protein.representative_types[mask]
        return neighbor_charges, neighbor_types, distances

def calculate_new_charge(resid, acid_basic, charge):
    if acid_basic == -1:
        new_charge = 0.0 if charge == -1 else -1.0
    elif acid_basic == 1:
        new_charge = 1.0 if charge == 0 else 0.0
    else:
        new_charge = charge
    return new_charge, new_charge - charge

def calculate_delta_term_elec(neighbor_charges, distances, residue_mc, new_charge):
    K_elec = 2.43232 / 10
    L = 1.0
    resid_mc = list(residue_mc.keys())[0]
    old_charge = residue_mc[resid_mc]["charge"]
    valid = (distances != 0.0) & (neighbor_charges != 0.0)
    if not np.any(valid):
        return 0.0

    distances = distances[valid]
    neighbor_charges = neighbor_charges[valid]
    signs = np.sign(neighbor_charges)
    prefactor = np.sum((signs / distances) * np.exp(-distances / L))
    return (new_charge - old_charge) * prefactor * K_elec

def calculate_delta_term_polar(resname, acid_basic, delta_q, neighbor_types, distances):
    R_max = 5 / 10
    R_max_nonpolar = 7 / 10
    tau = 0.1 * 100

    polar_parameters = {
        'ASP': (0.72426, 5.57075),
        'CTR': (0.72426, 5.57075),
        'GLU': (0.380763, 9.44846),
        'LYS': (0.0110311, 5.65882),
        'ARG': (0.0110311, 5.65882),
        'NTR': (0.0110311, 5.65882),
        'HIS': (0.602896, 9.36507),
        'CYS': (0.037 * 0.001987 * 300, 85.91 * 0.001987 * 300),
        'TYR': (0.0, 0.0),
    }

    if resname not in polar_parameters or acid_basic == 0 or delta_q == 0:
        return 0.0

    neighbor_types = np.asarray(neighbor_types, dtype=object)
    distances = np.asarray(distances, dtype=float)

    is_polar = np.isin(neighbor_types, ["P", "B", "A"])
    is_nonpolar = neighbor_types == "N"

    polar_weights = np.where(
        is_polar,
        np.where(distances <= R_max, 1.0, np.exp(-tau * (distances - R_max) ** 2)),
        0.0,
    )
    nonpolar_weights = np.where(
        is_nonpolar,
        np.where(
            distances <= R_max_nonpolar,
            1.0,
            np.exp(-tau * (distances - R_max_nonpolar) ** 2),
        ),
        0.0,
    )

    environment_polar = np.sum(polar_weights)
    environment_no_polar = np.sum(nonpolar_weights)

    Bp, Bnp = polar_parameters[resname]
    Npmax = 3.9
    Nnpmax = 17.78
    alfaP = 0.416683
    alfanp = 0.049

    Up = math.exp(-alfaP * (environment_polar - Npmax) ** 2) if environment_polar <= Npmax else 1.0
    Unp = math.exp(-alfanp * (environment_no_polar - Nnpmax) ** 2) if environment_no_polar <= Nnpmax else 1.0

    term_polar = acid_basic * (Bnp * Unp - Bp * Up)
    return delta_q * term_polar

def calculate_delta_term_pH(resname, delta_q, pH, T=300):
    pKa = {'ASP':4.0,'GLU':4.5,'LYS':10.6,'ARG':12.0,'HIS':6.4,'CYS':8.3,'TYR':11.0,'NTR':7.5,'CTR':3.5}
    kb = 0.001987
    return delta_q * (pH - pKa[resname]) * kb * T * np.log(10)

def accept_or_reject(resid, protein, charged_residues,
                     dE_pH, dE_elec, dE_polar, new_charge):

    kb, T = 0.001987, 300
    dE = dE_pH + dE_elec + dE_polar

    new_list = charged_residues.copy()

    if dE < 0 or random.random() < math.exp(-dE/(kb*T)):

        # Update the list of charged residues.
        new_list = [(r, new_charge if r == resid else q) for r, q in charged_residues]

        # Get the particle index (CB or CA).
        atom_dict = protein.data[resid]["atoms"]
        atom_name = list(atom_dict.keys())[0]
        particle_index = atom_dict[atom_name]["index"]

        return new_list, (particle_index, new_charge)

    else:
        return charged_residues, None

class ProtonationMC:

    def __init__(self, protein, pH=7.0, T=300, use_polar=True,
                 use_electrostatic=True):
        self.protein = protein
        self.pH = pH
        self.T = T
        self.use_polar = use_polar
        self.use_electrostatic = use_electrostatic
        self._geometry_version = -1
        self._neighbor_geometry = {}
        self._elec_weights = {}
        self._polar_unit = {}


    def attempt_charge_flip(self, charged_residues):

        if self._geometry_version != self.protein._geometry_version:
            self._neighbor_geometry.clear()
            self._elec_weights.clear()
            self._polar_unit.clear()
            self._geometry_version = self.protein._geometry_version

        residue_mc = self.protein.mc.get_current_residue_info()

        if residue_mc is None:
            return charged_residues, None

        resid = list(residue_mc.keys())[0]
        info = residue_mc[resid]
        neighbor_data = self._neighbor_geometry.get(resid)
        if neighbor_data is None:
            fresh_neighbor_data = self.protein.neighborhood.get_neighbor_data(resid)
            if fresh_neighbor_data is not None:
                _, neighbor_types, distances = fresh_neighbor_data
                neighbor_data = (neighbor_types, distances)
                self._neighbor_geometry[resid] = neighbor_data

        if neighbor_data is None:
            return charged_residues, None

        neighbor_types, distances = neighbor_data
        center_index = self.protein._representative_resid_to_index[resid]
        mask = self.protein.neighborhood._masks[center_index]
        neighbor_charges = self.protein.representative_charges[mask]

        old_charge = info["charge"]

        acid_base = -1 if info["type"] == "A" else 1 if info["type"] == "B" else 0

        if acid_base == 0:
            return charged_residues, None

        new_charge, delta_q = calculate_new_charge(resid, acid_base, old_charge)

        dE_elec = 0.0
        if self.use_electrostatic:
            elec_weights = self._elec_weights.get(resid)
            if elec_weights is None:
                elec_weights = np.zeros_like(distances)
                valid_distances = distances != 0.0
                elec_weights[valid_distances] = (
                    np.exp(-distances[valid_distances]) / distances[valid_distances]
                )
                self._elec_weights[resid] = elec_weights
            dE_elec = (
                (new_charge - old_charge)
                * (2.43232 / 10)
                * float(np.dot(np.sign(neighbor_charges), elec_weights))
            )
        dE_polar = 0.0
        if self.use_polar:
            polar_unit = self._polar_unit.get(resid)
            if polar_unit is None:
                polar_unit = calculate_delta_term_polar(
                    info["resname"], acid_base, 1.0, neighbor_types, distances
                )
                self._polar_unit[resid] = polar_unit
            dE_polar = delta_q * polar_unit
        dE_pH = calculate_delta_term_pH(info["resname"], delta_q, self.pH, self.T)

        new_list, particle_info = accept_or_reject(
            resid,
            self.protein,
            charged_residues,
            dE_pH,
            dE_elec,
            dE_polar,
            new_charge
        )

        accepted = new_list != charged_residues

        if accepted:
            self.protein.data[resid]["charge"] = new_charge
            representative_index = self.protein._representative_resid_to_index.get(resid)
            if representative_index is not None:
                self.protein.representative_charges[representative_index] = new_charge

        return new_list, particle_info
        
#############################################
##### ADDITIONAL FUNCTIONS ##################
#############################################

def process_charged_residue_file(filename):
    with open(filename, 'r') as file:
        residues = []
        for line in file:
            parts = line.split()
            residue = int(parts[0])
            charge = float(parts[1])
            residues.append((residue, charge))

    # Keep residues with nonzero charge.
    charged_residues = [(residue, charge) for residue, charge in residues if charge != 0.0]
    
    return charged_residues

def get_target_atom_indices_and_info(oa):
    """Return indices and complete CB/CA atom data for efficient filtering."""
    target_indices = []  
    target_atoms_info = []  
      
    one_to_three = {  
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP",  
        "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",  
        "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",  
        "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",  
        "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"  
    }  
      
    for residue in oa.pdb.topology.residues():  
        cb_atom = None  
        ca_atom = None  
          
        for atom in residue.atoms():  
            if atom.name == 'CB':  
                cb_atom = atom  
            elif atom.name == 'CA':  
                ca_atom = atom  
          
        target_atom = cb_atom if cb_atom is not None else ca_atom  
          
        if target_atom is not None:  
            # Index for fast position access.
            target_indices.append(target_atom.index)  
              
            # Complete atom information.
            if residue.index < len(oa.seq):  
                real_resname_one = oa.seq[residue.index]  
                real_resname_three = one_to_three.get(real_resname_one, "UNK")  
            else:  
                real_resname_three = "UNK"  
              
            target_atoms_info.append(  
                (target_atom.index, target_atom.name, residue.index, real_resname_three)  
            )  
      
    return target_indices, target_atoms_info


# Backward-compatible aliases for earlier experimental naming.
Proteina = Protein
procesador_de_archivo_con_residuos_cargados = process_charged_residue_file
