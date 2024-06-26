import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem


def extract_ECFP_dataset(init_train_set_path, num_train_samples):
    """
        Load background training data used to pre-train the predictive model    
    """
    
    print("Loading D0")
    train_set = pd.read_csv(init_train_set_path)
    feature_cols = [f"bit{i}" for i in range(2048)]
    target_col = ["activity"]
    smiles_train = train_set["smiles"].values.reshape(-1)
    x_train = train_set[feature_cols].values
    y_train = train_set[target_col].values.reshape(-1)
    sample_weight = np.array([1. for i in range(len(x_train))])
    print("The feature matrix shape: ", x_train.shape)
    print("The labels shape: ", y_train.shape)

    train_sample = train_set[train_set["activity"] == 1].sample(num_train_samples).smiles.tolist()
    return x_train, y_train, sample_weight, smiles_train, train_sample

def fingerprints_from_mol(mol, type = "counts", size = 2048, radius = 3):
    "and kwargs"

    if type == "binary":
        if isinstance(mol, list):
            fps = [AllChem.GetMorganFingerprintAsBitVect(m, radius, useFeatures=True) for m in mol if m is not None]
            l = len(mol)
        else:
            fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius, useFeatures=True)]
            l = 1
        nfp = np.zeros((l, size), np.int32)
        for i in range(l):
            for idx,v in enumerate(fps[i]):
                nfp[i, idx] += int(v)

    if type == "counts":
        if isinstance(mol, list):
            fps = [AllChem.GetMorganFingerprint(m, radius, useCounts=True, useFeatures=True) for m in mol if m is not None]
            l = len(mol)
        else:
            fps = [AllChem.GetMorganFingerprint(mol, radius, useCounts=True, useFeatures=True)]
            l = 1
        nfp = np.zeros((l, size), np.int32)
        for i in range(l):
            for idx,v in fps[i].GetNonzeroElements().items():
                nidx = idx%size
                nfp[i, nidx] += int(v)
    
    return nfp

def double_sigmoid(x, low, high, alpha_1, alpha_2):
    return 10**(x*alpha_1)/(10**(x*alpha_1)+10**(low*alpha_1)) - 10**(x*alpha_2)/(10**(x*alpha_2)+10**(high*alpha_2))
