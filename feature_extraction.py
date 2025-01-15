import pandas as pd
import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import MACCSkeys
import networkx as nx
from torch.utils.data import Dataset
from torch_geometric import data as DATA
import torch
from Bio.PDB import PDBParser
from torchdrug import core, data, datasets, utils
from torch_geometric.data import InMemoryDataset


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)    
    c_size = mol.GetNumAtoms()    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )
    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])        
    return c_size, features, edge_index

def drug_graph_construct(smile):
    c_size, x, edge_index = smile_to_graph(smile)  
    feature_mat = np.array(x)
    adjacency_mat = np.zeros((c_size,c_size))
    for i in range(len(edge_index)):
        link = edge_index[i]
        in_node_index = link[0]
        out_node_index = link[1]
        adjacency_mat[in_node_index][out_node_index] = 1
    feature_mat = torch.from_numpy(feature_mat).float()    
    adjacency_mat = torch.from_numpy(adjacency_mat).float()     
    return feature_mat, adjacency_mat

def protein_feature_extraction(prot_fasta,max_seq_len,seq_dict):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot_fasta[:max_seq_len]): 
        x[i] = seq_dict[ch]
    x = torch.LongTensor(x)
    x = torch.unsqueeze(x,0)    
    return x

CHARPROTSET = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
            "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
            "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
            "U": 19, "T": 20, "W": 21,
            "V": 22, "Y": 23, "X": 24,
            "Z": 25 }

def label_chars(chars, max_len, char_set):
    X = torch.zeros(max_len, dtype=torch.long)
    for i, ch in enumerate(chars[:max_len]):
        X[i] = char_set[ch]
    return X

def smiles_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return torch.tensor([int(_) for _ in fp.ToBitString()[1:]])

class GraphDataset(InMemoryDataset):
    def __init__(self, root='/tmp', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        pass

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass

    def process(self, graphs_dict):
        data_list = []
        for key in graphs_dict:
            size, features, edge_index = graphs_dict[key]
            if self.dttype == 'drug':
                GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index).transpose(1, 0))
            else:
                GCNData = DATA.Data(x=torch.Tensor(features), edge_index=torch.LongTensor(edge_index))
            GCNData.__setitem__(f'{self.dttype}_size', torch.LongTensor([size]))
            data_list.append(GCNData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class multi_DRP_dataset(Dataset):
    def __init__(self, drug_atom_dict, drug_bond_dict):
        super(multi_DRP_dataset, self).__init__()
        self.drug_atom, self.drug_bond_dict = drug_atom_dict, drug_bond_dict
        self.length = len(self.drug_atom)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return (self.drug_atom[index], self.drug_bond_dict[index])

def bio_load_pdb(pdb):
    parser = PDBParser(QUIET=True)
    protein = parser.get_structure(0, pdb)
    residues = [residue for residue in protein.get_residues()][:256]
    residue_ids = {residue.full_id for residue in residues}
    residue_type = [data.Protein.residue2id.get(residue.get_resname(), 0) for residue in residues]
    chain_id = [data.Protein.alphabet2id.get(residue.get_parent().id, 0) for residue in residues]
    insertion_code = [data.Protein.alphabet2id.get(residue.full_id[3][2], -1) for residue in residues]
    residue_number = [residue.full_id[3][1] for residue in residues]
    id2residue = {residue.full_id: i for i, residue in enumerate(residues)}

    atoms = [atom for atom in protein.get_atoms() if atom.get_parent().full_id in residue_ids]
    occupancy = [atom.get_occupancy() for atom in atoms]
    b_factor = [atom.get_bfactor() for atom in atoms]
    atom_type = [data.feature.atom_vocab.get(atom.get_name()[0], 0) for atom in atoms]
    atom_name = [data.Protein.atom_name2id.get(atom.get_name(), 37) for atom in atoms]
    node_position = np.stack([atom.get_coord() for atom in atoms], axis=0)
    node_position = torch.as_tensor(node_position)
    atom2residue = [id2residue[atom.get_parent().full_id] for atom in atoms]

    edge_list = [[0, 0, 0]]
    bond_type = [0]

    return data.Protein(edge_list, atom_type=atom_type, bond_type=bond_type, residue_type=residue_type,
                num_node=len(atoms), num_residue=len(residues), atom_name=atom_name,
                atom2residue=atom2residue, occupancy=occupancy, b_factor=b_factor, chain_id=chain_id,
                residue_number=residue_number, node_position=node_position, insertion_code=insertion_code, # residue_feature=residue_feature
            ), "".join([data.Protein.id2residue_symbol[res] for res in residue_type])

def protein_graph():
    folder_path = 'protein_structrue\davis\structure'
    file_path = 'Data\davis\\targets.xlsx'
    proteins = []
    df = pd.read_excel(file_path)
    column_values = df['Protein_Name'].tolist()
    pdb_files = [f"{folder_path}\\{value}.pdb" for value in column_values]
    for i, pdb_file in enumerate(pdb_files):
        protein, _ = bio_load_pdb(pdb_file)
        proteins.append(protein)
    return proteins