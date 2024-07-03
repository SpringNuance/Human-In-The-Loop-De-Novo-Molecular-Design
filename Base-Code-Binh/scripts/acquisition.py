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
    
def predict_proba_from_model(feedback_type, feedback_model, smiles, batch_pred=10000):
    
    # This is not computationally extensive yet
    # choose float 32
    features = np.array([compute_fingerprints(smiles) for smiles in smiles], dtype=np.float32)

    count_batch = 1
    if feedback_type == "scoring":
        pred_label_proba = []
        for batch_idx in range(0, len(features), batch_pred):
            features_batch = torch.tensor(features[batch_idx:batch_idx+batch_pred], dtype=torch.float32)
            pred_label_proba_batch = feedback_model.forward(features_batch).cpu().detach().numpy()
            pred_label_proba_batch.flatten()
            pred_label_proba.extend(pred_label_proba_batch)
            print(f"Batch {count_batch} processed")
            count_batch += 1
        return pred_label_proba
    
    elif feedback_type == "comparing":
        from itertools import product
        # Generate all repeated combinations of 2 out of len(smiles)

        smiles_len, features_dim = features.shape
        # comb is an iterator
        comb = list(product(range(smiles_len), repeat=2))

        # Remove indices that compare to itself
        comb = [(i, j) for (i, j) in comb if i != j]
        num_comb = len(comb) 
        
        pred_label_proba = np.zeros(smiles_len)
        
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
            pred_label_proba_batch = feedback_model.forward(features_1, features_2).cpu().detach().numpy()

            for i, (idx1, idx2) in enumerate(comb_batch):
                pred_label_proba[idx1] += pred_label_proba_batch[i]

            print(f"B{batch_count} processed", end=" | ")
            batch_count += 1

        # Computing the average
        pred_label_proba = pred_label_proba/(smiles_len - 1)
        return pred_label_proba

    elif feedback_type == "ranking":

        # This is a potentially very computationally extensive task
        # nCr could easily be in the order of 10^9
        # Example of nCr(1000, 3) = 166,167,000 combinations

        from itertools import combinations
        smiles_len, features_dim = features.shape
        comb = combinations(range(smiles_len), 3)
        
        nCr = lambda n, r: np.math.factorial(n) // (np.math.factorial(r) * np.math.factorial(n - r))
        num_comb = int(nCr(smiles_len, 3))

        pred_label_proba = np.zeros(smiles_len)
        
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
            pred_label_proba_batch = feedback_model.forward(features_1, features_2, features_3).cpu().detach().numpy()

            for i, (idx1, idx2, idx3) in enumerate(comb_batch):
                pred_label_proba[idx1] += pred_label_proba_batch[i, 0]
                pred_label_proba[idx2] += pred_label_proba_batch[i, 1]
                pred_label_proba[idx3] += pred_label_proba_batch[i, 2]

            print(f"B{batch_count} processed", end=" | ")
            batch_count += 1

        # Computing the average
        count_each_index = nCr(smiles_len - 1, 2)
        pred_label_proba = pred_label_proba/count_each_index

    return pred_label_proba

def random_sampling(feedback_type, feedback_model, scaffold_df, num_queries, 
                     smiles, already_selected_indices, rng):
    
    df_len = len(scaffold_df)

    # Get all possible indices
    all_indices = np.arange(df_len)
    
    # Remove already selected indices from the pool
    not_selected_indices = np.setdiff1d(all_indices, already_selected_indices)

    # Randomly select `num_queries` indices from the pool of not selected indices
    selected_indices = rng.choice(not_selected_indices, size=num_queries, replace=False)

    return selected_indices

def greedy_sampling(feedback_type, feedback_model, scaffold_df, num_queries, 
                     smiles, already_selected_indices, rng):
    
    predicted_scores = predict_proba_from_model(feedback_type, feedback_model, smiles)
    df_len = len(scaffold_df)

    # Get all possible indices
    all_indices = np.arange(df_len)

    # Remove already selected indices from the pool
    not_selected_indices = np.setdiff1d(all_indices, already_selected_indices)

    # Sort the indices by the predicted scores
    sorted_indices = np.argsort(predicted_scores[not_selected_indices])[::-1]

    # Select the top `num_queries` indices
    selected_indices = not_selected_indices[sorted_indices[:num_queries]]

    return selected_indices

def uncertainty_sampling(feedback_type, feedback_model, scaffold_df, num_queries,
                            smiles, already_selected_indices, rng):
        
        predicted_scores = predict_proba_from_model(feedback_type, feedback_model, smiles)

        df_len = len(scaffold_df)
    
        # Get all possible indices
        all_indices = np.arange(df_len)
    
        # Remove already selected indices from the pool
        not_selected_indices = np.setdiff1d(all_indices, already_selected_indices)
    
        if feedback_type == "scoring":
            # Choose indices that maximize this value: abs(score - 0.5)
            # 0.5 is the most uncertain because the model is unsure whether smiles have drd2 or not

            # Sort the indices by the uncertainty
            sorted_indices = np.argsort(np.abs(predicted_scores[not_selected_indices] - 0.5))

        elif feedback_type == "comparing":
            # Choose indices that maximize this value: abs(score - 0.5)
            # 0.5 is the most uncertain because the model is unsure if this smiles is better or worse than others

            # Sort the indices by the uncertainty
            sorted_indices = np.argsort(np.abs(predicted_scores[not_selected_indices] - 0.5))

        elif feedback_type == "ranking":
            # Choose indices that maximize this value: abs(score - 2/3)
            # smiles with highest rates of being 2nd rank in all combinations of rankings is the most uncertain 

            # Sort the indices by the uncertainty
            sorted_indices = np.argsort(np.abs(predicted_scores[not_selected_indices] - 2/3))

        # Select the top `num_queries` indices
        selected_indices = not_selected_indices[sorted_indices[:num_queries]]
    
        return selected_indices

def select_query_feedback(feedback_type, feedback_model, scaffold_df, num_queries, smiles, already_selected_indices, acquisition = 'random', rng=None):
    '''
    Parameters
    ----------
    feedback_type: "scoring", "comparing" or "ranking"
    feedback_model: the corresponding pretrained model for the feedback type
    scaffold_df: the dataframe pool of unlabelled molecules with scores higher than the threshold from REINVENT
     column 'scaffold' contains the scaffold pattern
     column 'smiles' contains the SMILES string
     column 'human_component' contains the human feedback score (from REINVENT)
     column 'raw_human_component' contains the raw human feedback (from REINVENT)
     column 'total_score' contains the total score (from REINVENT)

    num_queries: number of queries to select
    smiles: list of high-scoring smiles string. This is the same as scaffold_df['smiles']
    already_selected_indices: selected smiles in previous feedback HITL iterations
    acquisition: acquisition criterion to select the queries
    rng: random number generator

    Returns
    -------
    list[int]: 
        list of indices of the next query

    '''

    # select acquisition: (TODO: EIG, other strategies)
    if acquisition == 'uncertainty':
        acquisition_func = uncertainty_sampling
    elif acquisition == 'greedy':
        acquisition_func = greedy_sampling
    elif acquisition == 'random':
        acquisition_func = random_sampling
    else:
        print("Warning: unknown acquisition criterion. Using random sampling.")
        acquisition_func = random_sampling
    new_queried_smiles_indices = acquisition_func(feedback_type, feedback_model, scaffold_df, num_queries, smiles, already_selected_indices, rng)
    return new_queried_smiles_indices
