import dgl
import pandas as pd
import numpy as np
import rdkit
import torch
from rdkit import rdBase, Chem, DataStructs
from dgl import DGLGraph
from sklearn.preprocessing import MinMaxScaler
from rdkit.Chem import rdMolDescriptors
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# # # ==============--->>> -------------------- Prepare Data for Fingerprint ------------------ <<<---============== # # #

def get_fingerprint(smi):
    """
    Generate MACCS fingerprints using RDKit.

    Args:
        smi (str): SMILES string of the molecule.

    Returns:
        np.ndarray: The MACCS fingerprint as a numpy array.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
        arr = np.zeros((167,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    else:
        return None


def get_finger_date(data):
    """
    Get MACCS fingerprints for a dataset and preprocess them.

    Args:
        data (pd.DataFrame): DataFrame with a 'smiles' column containing SMILES strings.

    Returns:
        pd.DataFrame: DataFrame containing processed MACCS fingerprints.
    """
    fingerprints = [get_fingerprint(smi) for smi in data['smiles']]

    if len(fingerprints) == 0:
        return pd.DataFrame()

    # If any invalid SMILES still exist, replace with zero vector to keep row alignment
    fingerprints = [
        fp if fp is not None else np.zeros((167,), dtype=int)
        for fp in fingerprints
    ]

    fingerprint_df = pd.DataFrame(fingerprints)

    means = fingerprint_df.mean()
    stds = fingerprint_df.std()

    delete = means[means == 0].index.tolist()
    delete1 = stds[stds == 0].index.tolist()
    delete_cols = list(set(delete + delete1))

    fin1 = fingerprint_df.drop(columns=delete_cols)

    # Prevent empty dataframe after dropping columns
    if fin1.shape[1] == 0:
        fin1 = fingerprint_df.copy()

    cormatrix = fin1.corr(method='pearson')

    delecor = []
    for col in range(len(cormatrix.columns)):
        for row in range(col):
            if abs(cormatrix.iloc[row, col]) >= 0.95:
                delecor.append(cormatrix.columns[col])

    delecor = list(set(delecor))

    fin2 = fin1.drop(columns=delecor)

    # Prevent empty dataframe after removing highly correlated columns
    if fin2.shape[1] == 0:
        fin2 = fin1.copy()

    fin = [np.array(fin2.iloc[i], dtype=np.float32) for i in range(len(fin2))]

    fingers = np.array(fin, dtype=np.float32)
    fingers_df = pd.DataFrame(fingers)

    return fingers_df


# # # ==============--->>> -------------------- Prepare Data for ChemBERTa ------------------ <<<---============== # # #

def mean_pooling(last_hidden_state, attention_mask):
    """
    Mean pooling for transformer outputs.
    """
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    sum_embeddings = masked_embeddings.sum(dim=1)
    sum_mask = mask.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def get_chemberta_date(
    data,
    smiles_col='smiles',
    model_dir='/root/autodl-tmp/neuro/ChemBERTa-zinc-base-v1',
    batch_size=64,
    max_length=128,
    device=None
):
    """
    Generate ChemBERTa embeddings for a dataset.

    Args:
        data (pd.DataFrame): DataFrame containing SMILES.
        smiles_col (str): Column name of SMILES.
        model_dir (str): Local ChemBERTa model directory.
        batch_size (int): Batch size for inference.
        max_length (int): Max token length.
        device (torch.device): cuda/cpu device.

    Returns:
        pd.DataFrame: ChemBERTa embedding dataframe.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smiles_list = data[smiles_col].astype(str).fillna("").tolist()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir)
    model = model.to(device)
    model.eval()

    all_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="ChemBERTa embedding"):
            batch_smiles = smiles_list[i:i + batch_size]

            encoded = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}

            outputs = model(**encoded)

            if hasattr(outputs, 'last_hidden_state'):
                batch_embeddings = mean_pooling(outputs.last_hidden_state, encoded['attention_mask'])
            else:
                raise ValueError("ChemBERTa model output does not contain last_hidden_state.")

            batch_embeddings = batch_embeddings.detach().cpu().numpy().astype(np.float32)
            all_embeddings.append(batch_embeddings)

    if len(all_embeddings) == 0:
        return pd.DataFrame()

    chemberta_embeddings = np.vstack(all_embeddings)
    chemberta_df = pd.DataFrame(chemberta_embeddings)

    return chemberta_df


# # # ==============--->>> -------------------- Prepare Data for Graph ------------------ <<<---============== # # #

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(atom):
    possible_atom = ['C', 'N', 'O', 'F', 'P', 'Cl', 'Br', 'I', 'DU']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), possible_atom)
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 1])
    atom_features += one_of_k_encoding_unk(
        atom.GetHybridization(),
        [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D
        ]
    )
    return np.array(atom_features, dtype=np.float32)


def get_bond_features(bond):
    bond_type = bond.GetBondType()
    bond_feats = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    return np.array(bond_feats, dtype=np.float32)


def get_graph_date(data):
    data_smiles = data['smiles'].tolist()

    molgraph = []

    for molecule_smiles in data_smiles:
        molecule = Chem.MolFromSmiles(molecule_smiles)

        if molecule is None:
            # keep row alignment; create a minimal dummy graph
            G = dgl.graph(([0], [0]), num_nodes=1)
            G.ndata['x'] = torch.zeros((1, 26), dtype=torch.float32)
            G.edata['w'] = torch.zeros((1, 6), dtype=torch.float32)
            molgraph.append(G)
            continue

        molecule = rdkit.Chem.AddHs(molecule)

        src_list = []
        dst_list = []
        edge_features = []
        node_features = []

        num_atoms = molecule.GetNumAtoms()

        for i in range(num_atoms):
            atom_i = molecule.GetAtomWithIdx(i)
            atom_i_features = get_atom_features(atom_i)
            node_features.append(atom_i_features)

        for i in range(num_atoms):
            for j in range(num_atoms):
                if i == j:
                    continue
                bond_ij = molecule.GetBondBetweenAtoms(i, j)
                if bond_ij is not None:
                    src_list.append(i)
                    dst_list.append(j)
                    bond_features_ij = get_bond_features(bond_ij)
                    edge_features.append(bond_features_ij)

        G = dgl.graph((src_list, dst_list), num_nodes=num_atoms)
        G = dgl.add_self_loop(G)

        node_features = torch.from_numpy(np.array(node_features)).float()

        if len(edge_features) > 0:
            edge_features = torch.from_numpy(np.array(edge_features)).float()
        else:
            edge_features = torch.zeros((0, 6), dtype=torch.float32)

        # add self-loop edge features
        num_self_loops = num_atoms
        self_loop_features = torch.zeros((num_self_loops, 6), dtype=torch.float32)

        if edge_features.shape[0] > 0:
            full_edge_features = torch.cat([edge_features, self_loop_features], dim=0)
        else:
            full_edge_features = self_loop_features

        G.ndata['x'] = node_features
        G.edata['w'] = full_edge_features

        molgraph.append(G)

    molgraph_df = pd.DataFrame(molgraph)
    return molgraph_df


# # # ==============--->>> -------------------- Prepare Data for CCS ------------------ <<<---============== # # #

# def get_CCS_date(data):
#     scaler = MinMaxScaler()
#     data['CCS_normalized'] = scaler.fit_transform(data['CCS'].values.reshape(-1, 1)).squeeze()
#     ccs_data = np.array(data['CCS_normalized'].tolist())
#     ccs_df = pd.DataFrame(ccs_data, columns=['CCS_normalized'])
#     return ccs_df