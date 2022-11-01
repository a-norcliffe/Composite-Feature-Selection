"""Library to turn chemistry datasets into vectors and save them as numpy files."""

# third party
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

# List of SMARTS adapted from GlobalChem - Open Smiles
functional_groups_smarts = {
    "acetic anydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",  # 0
    "acetylenic carbon": "[$([CX2]#C)]",  # Alkyne
    "acyl bromide": "[CX3](=[OX1])[Br]",
    "acyl chloride": "[CX3](=[OX1])[Cl]",
    "acyl fluoride": "[CX3](=[OX1])[F]",
    "acyl iodide": "[CX3](=[OX1])[I]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "alkane": "[CX4]",
    "unbranched alkene": "[R0;D2,D1][R0;D2][R0;D2,D1]",
    "allenic carbon": "[$([CX2](=C)=C)]",
    "amide": "[NX3][CX3](=[OX1])[#6]",  # 10
    "amidium": "[NX3][CX3]=[NX3+]",
    "amino acid": "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-,N]",
    "azo nitrogen": "[NX2]=N",
    "azole": "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",
    "azoxy nitrogen": "[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]",
    "diazene": "[NX2]=[NX2]",
    "diazo nitrogen": "[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]",
    "benzene": "[cR1]1[cR1][cR1][cR1][cR1][cR1]1",
    "bromine": "[Br]",
    "carbamate": "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",  # 20
    "carbamic ester": "[NX3][CX3](=[OX1])[OX2H0]",
    "carbamic acid": "[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]",
    "carbo azosulfone": "[SX4](C)(C)(=O)=N",
    "carbo thiocarboxylate": "[S-][CX3](=S)[#6]",
    "carbo thioester": "S([#6])[CX3](=O)[#6]",
    "carboxylate ion": "[CX3](=O)[O-]",
    "carbonic acid": "[CX3](=[OX1])(O)O",
    "carbonic ester": "C[OX2][CX3](=[OX1])[OX2]C",
    "carbonyl group": "[CX3]=[OX1]",
    "carbonyl with carbon": "[CX3](=[OX1])C",  # 30
    "carbonyl with nitrogen": "[OX1]=CN",
    "carbonyl with oxygen": "[CX3](=[OX1])O",
    "carboxylic acid": "[CX3](=O)[OX1H0-,OX2H1]",
    "chlorine": "[Cl]",
    "cyanamide": "[NX3][CX2]#[NX1]",
    "di sulfide": "[#16X2H0][#16X2H0]",
    "enamine": "[NX3][CX3]=[CX3]",
    "enol": "[OX2H][#6X3]=[#6]",
    "ester": "[#6][CX3](=O)[OX2H0][#6]",
    "ether": "[OD2](C)C",  # 40
    "fluorine": "[F]",
    "hydrogen": "[H]",
    "hydrazine": "[NX3][NX3]",
    "hydrazone": "[NX3][NX2]=[*]",
    "hydroxyl in carboxylic acid": "[OX2H][CX3]=[OX1]",
    "isonitrile": "[CX1-]#[NX2+]",
    "imide": "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
    "imine": "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",
    "iminium": "[NX3+]=[CX3]",
    "ketone": "[#6][CX3](=O)[#6]",
    "peroxide": "[OX2,OX1-][OX2,OX1-]",  # 50
    "phenol": "[OX2H][cX3]:[c]",
    "phosphoric acid": "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]",
    "phosphoric ester": "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]",
    "primary alcohol": "[OX2H]",  # Also known as hydroxl
    "primary amine": "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
    "proton": "[H+]",
    "mono sulfide": "[#16X2H0][!#16]",
    "nitrate": "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
    "nitrile": "[NX1]#[CX2]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",  # 60
    "nitroso": "[NX2]=[OX1]",
    "n-oxide": "[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])]",
    "secondary amine": "[NX3;H1;!$(NC=O)]",
    "sulfate": "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]",
    "sulfamate": "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]",
    "sulfamic acid": "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]",
    "sulfenic acid": "[#16X2][OX2H,OX1H0-]",
    "sulfenate": "[#16X2][OX2H0]",
    "sulfide": "[#16X2H0]",
    "sulfonate": "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]",  # 70
    "sulfinate": "[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]",
    "sulfinic acid": "[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
    "sulfonamide": "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]",
    "sulfone": "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]",
    "sulfonic acid": "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
    "sulfoxide": "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]",
    "sulfur": "[#16!H0]",
    "sulfuric acid ester": "[$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)]",
    "sulfuric acid diester": "[$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6]),$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6])]",
    "thioamide": "[NX3][CX3]=[SX1]",  # 80
    "thiol": "[#16X2H]",
    "vinylic carbon": "[$([CX3]=[CX3])]",  # Also known as alkene
}
functional_groups = [
    Chem.MolFromSmarts(smi) for smi in list(functional_groups_smarts.values())
]


def featurise_mols(mols, substruct_list):
    feature_vectors = np.zeros((len(mols), len(substruct_list)))
    for i, mol in enumerate(mols):
        for j, substruct in enumerate(substruct_list):
            if len(mol.GetSubstructMatch(substruct)) > 0:
                feature_vectors[i][j] = 1
    return feature_vectors


def generate_y(mols, logic):
    """Generate corresponding label (y) given feature (x).

    Args:
      - mols: features
      - logic: data scenario (logic_4, logic_10, logic_13)
    Returns:
      - y: corresponding labels
    """
    # Logit computation
    if logic == "logic_4":
        patt1 = Chem.MolFromSmarts("[OD2](C)C")  # Ether
        patt2 = Chem.MolFromSmarts("[CX2]#[CX2]")  # Alkyne
        Y = [
            1
            if len(mol.GetSubstructMatch(patt1)) > 0
            or len(mol.GetSubstructMatch(patt2)) < 1
            else 0
            for mol in mols
        ]
    elif logic == "logic_10":
        patt1 = Chem.MolFromSmarts(
            "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
        )  # Primary amine not amide
        patt2 = Chem.MolFromSmarts("[OD2](C)C")  # Ether
        patt3 = Chem.MolFromSmarts("[cR1]1[cR1][cR1][cR1][cR1][cR1]1")  # Benzene
        Y = [
            1
            if (
                len(mol.GetSubstructMatch(patt1)) > 0
                and len(mol.GetSubstructMatch(patt2)) < 1
            )
            or (
                len(mol.GetSubstructMatch(patt2)) < 1
                and len(mol.GetSubstructMatch(patt3)) < 1
            )
            else 0
            for mol in mols
        ]
    elif logic == "logic_13":
        patt1 = Chem.MolFromSmarts("[cR1]1[cR1][cR1][cR1][cR1][cR1]1")  # Benzene
        patt2 = Chem.MolFromSmarts("[CX3]=O")  # Carbonyl
        patt3 = Chem.MolFromSmarts("[CX2]#[CX2]")  # Alkyne
        patt4 = Chem.MolFromSmarts("[OD2](C)C")  # Ether
        Y = [
            1
            if (
                len(mol.GetSubstructMatch(patt1)) > 0
                and len(mol.GetSubstructMatch(patt2)) < 1
            )
            or (
                len(mol.GetSubstructMatch(patt3)) > 0
                and len(mol.GetSubstructMatch(patt4)) < 1
            )
            else 0
            for mol in mols
        ]

    return Y


def prepare_chem_dataset(train_path, test_path, logic):
    # Load SMILES strings from csv files
    train = list(pd.read_csv(train_path)["smiles"].values)
    test = list(pd.read_csv(test_path)["smiles"].values)

    train_mols = [Chem.MolFromSmiles(smi) for smi in train]
    test_mols = [Chem.MolFromSmiles(smi) for smi in test]

    X_train = featurise_mols(train_mols, functional_groups)
    X_test = featurise_mols(test_mols, functional_groups)

    Y_train = generate_y(train_mols, logic)
    Y_test = generate_y(test_mols, logic)

    return X_train, X_test, Y_train, Y_test


def make_chem_data(logic_id):
    # Load in the csv files and turn into numpy files for the datasets.
    path = Path(__file__).parent
    basepath = path / "chem_data"
    outpath = path / "chem_data"
    try:
        X_train, X_test, Y_train, Y_test = prepare_chem_dataset(
            "{}/logic_{}_train.csv".format(basepath, logic_id),
            "{}/logic_{}_test.csv".format(basepath, logic_id),
            "logic_{}".format(logic_id),
        )
        np.save(outpath / f"logic_{logic_id}_X_train.npy", X_train)
        np.save(outpath / f"logic_{logic_id}_X_test.npy", X_test)
        np.save(outpath / f"logic_{logic_id}_Y_train.npy", Y_train)
        np.save(outpath / f"logic_{logic_id}_Y_test.npy", Y_test)
    except FileNotFoundError:
        print(
            "Data not found, please download at https://github.com/google-research/graph-attribution/raw/main/data/all_16_logics_train_and_test.zip and place in datasets/chem_data",
        )
        raise
