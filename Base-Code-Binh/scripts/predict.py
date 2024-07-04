import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

def compute_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    else:
        return np.zeros((2048,), dtype=int)

def predict_proba_from_model(feedback_type, feedback_model, smiles):

    # This is not computationally extensive yet
    # choose float 32
    features = np.array([compute_fingerprints(smiles) for smiles in smiles], dtype=np.float32)
    features = torch.tensor(features, dtype=torch.float32)
    pred_label_proba = feedback_model.predict_proba(features).cpu().detach().numpy()
    return pred_label_proba


def predict_feedback_from_model(feedback_type, feedback_model, smiles, batch_pred=10000):
    
    # This is not computationally extensive yet
    # choose float 32
    features = np.array([compute_fingerprints(smiles) for smiles in smiles], dtype=np.float32)

    count_batch = 1
    if feedback_type == "scoring":
        pred_label_feedback = []
        for batch_idx in range(0, len(features), batch_pred):
            features_batch = torch.tensor(features[batch_idx:batch_idx+batch_pred], dtype=torch.float32)
            pred_label_feedback_batch = feedback_model.forward(features_batch).cpu().detach().numpy()
            pred_label_feedback_batch.flatten()
            pred_label_feedback.extend(pred_label_feedback_batch)
            print(f"Batch {count_batch} processed")
            count_batch += 1
        return pred_label_feedback
    
    elif feedback_type == "comparing":
        from itertools import product
        # Generate all repeated combinations of 2 out of len(smiles)

        smiles_len, features_dim = features.shape
        # comb is an iterator
        comb = list(product(range(smiles_len), repeat=2))

        # Remove indices that compare to itself
        comb = [(i, j) for (i, j) in comb if i != j]
        num_comb = len(comb) 
        
        pred_label_feedback = np.zeros(smiles_len)
        
        batch_count = 1

        for batch_idx in range(0, num_comb, batch_pred):
            features_1 = np.zeros((batch_pred, features_dim), dtype=np.float32) 
            features_2 = np.zeros((batch_pred, features_dim), dtype=np.float32)
            comb_batch = comb[batch_idx:batch_idx+batch_pred]

            for i, (idx1, idx2) in enumerate(comb_batch):
                features_1[i, :] = features[idx1, :]
                features_2[i, :] = features[idx2, :]

            features_1 = torch.tensor(features_1, dtype=torch.float32)
            features_2 = torch.tensor(features_2, dtype=torch.float32)
            pred_label_feedback_batch = feedback_model.forward(features_1, features_2).cpu().detach().numpy()

            for i, (idx1, idx2) in enumerate(comb_batch):
                pred_label_feedback[idx1] += 1 if pred_label_feedback_batch[i] > 0.5 else 0

            print(f"B{batch_count} processed", end=" | ")
            batch_count += 1

        # Computing the average
        pred_label_feedback = pred_label_feedback/(smiles_len - 1)
        return pred_label_feedback

    elif feedback_type == "ranking":

        # This is a potentially very computationally extensive task
        # nCr could easily be in the order of 10^9
        # Example of nCr(1000, 3) = 166,167,000 combinations

        from itertools import combinations
        smiles_len, features_dim = features.shape
        comb = combinations(range(smiles_len), 3)
        
        nCr = lambda n, r: np.math.factorial(n) // (np.math.factorial(r) * np.math.factorial(n - r))
        num_comb = int(nCr(smiles_len, 3))

        pred_label_feedback = np.zeros(smiles_len)
        
        batch_count = 1

        for batch_idx in range(0, num_comb, batch_pred):
            if batch_idx + batch_pred > num_comb:
                true_batch_pred = num_comb - batch_idx
            else:
                true_batch_pred = batch_pred
            features_1 = np.zeros((true_batch_pred, features_dim), dtype=np.float32) 
            features_2 = np.zeros((true_batch_pred, features_dim), dtype=np.float32)
            features_3 = np.zeros((true_batch_pred, features_dim), dtype=np.float32)
            comb_batch = []
            for i in range(true_batch_pred):
                comb_batch.append(next(comb))

            for i, (idx1, idx2, idx3) in enumerate(comb_batch):
                features_1[i, :] = features[idx1, :]
                features_2[i, :] = features[idx2, :]
                features_3[i, :] = features[idx3, :]

            features_1 = torch.tensor(features_1, dtype=torch.float32)
            features_2 = torch.tensor(features_2, dtype=torch.float32)
            features_3 = torch.tensor(features_3, dtype=torch.float32)
            pred_label_feedback_batch = feedback_model.forward(features_1, features_2, features_3).cpu().detach().numpy()
            
            # After sorting we have the ranking in [0, 1, 2]
            # Then we normalize it to [0, 0.5, 1]
            pred_label_ranking_batch = np.argsort(pred_label_feedback_batch, axis=1)/(3 - 1)

            for i, (idx1, idx2, idx3) in enumerate(comb_batch):
                pred_label_feedback[idx1] += pred_label_ranking_batch[i, 0]
                pred_label_feedback[idx2] += pred_label_ranking_batch[i, 1]
                pred_label_feedback[idx3] += pred_label_ranking_batch[i, 2]

            print(f"B{batch_count} processed", end=" | ")
            batch_count += 1

        # Computing the average
        count_each_index = nCr(smiles_len - 1, 2)
        pred_label_feedback = pred_label_feedback/count_each_index

    return pred_label_feedback