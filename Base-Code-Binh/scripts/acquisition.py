import numpy as np
from rdkit import Chem
from utils import fingerprints_from_mol
import torch

def random_sampling(U, X_train, n, model, acquisition, X_updated, y_updated, L=None,t=None):
    ret = np.random.choice(U, n, replace=False)
    return ret

def uncertainty_sampling(U, X_train, n, model, acquisition, X_updated, y_updated, L, t=None):
    X_pool = X_train[U,:]
    preds_p, var = model.predict_f(X_pool)
    idx = np.argsort(np.squeeze(var))[-n:]
    return U[idx]

# exploitation
def greedy_algorithm(U, X_train, n, model, acquisition, X_updated, y_updated, L=None, t=None):
    X_pool = X_train[U,:]
    preds_p,_ = model.predict_f(X_pool)
    preds_p=np.squeeze(preds_p)
    idx = np.argsort(preds_p)[::-1]
    assert len(idx)>0
    selected = idx[:n]
    return U[selected]


def thompson_sampling(U, X_U, n, model, acquisition, X_updated, y_updated, L=None, t=None):
    X_pool =X_U[U,:]
    # a sample from posterior
    preds_p = model.predict_f_samples(X_pool) 
    preds_p = np.squeeze(preds_p)
    # greedily maximize with respect to the randomly sampled belief
    idx = np.argsort(preds_p)[::-1] 
    assert len(idx)>0
    selected = idx[:n]
    return U[selected]


def local_idx_to_full_scaffold_df_idx(num_queries, already_selected_indices, idx):
    all_idx = np.arange(num_queries)
    mask = np.ones(num_queries, dtype=bool)
    mask[already_selected_indices] = False
    pred_idx = all_idx[mask]
    try:
        pred_idx[idx]
        return pred_idx[idx]
    except:
        valid_idx = [i if 0 <= i < len(pred_idx) else len(pred_idx) - 1 for i in idx]
        return pred_idx[valid_idx]

def predict_proba_from_model(feedback_type, feedback_model, smiles):
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    fps = fingerprints_from_mol(mols)
    
    if feedback_type == "scoring":
        features = torch.tensor(fps, dtype=torch.float32)
        pred_activity_score = feedback_model.forward(features).cpu().detach().numpy()
        pred_activity_score = pred_activity_score.flatten()
        return pred_activity_score
    
    elif feedback_type == "comparing":
        from itertools import product
        # Generate all repeated combinations of 2 out of len(smiles)

        smiles_len, fps_dim = fps.shape
        comb = list(product(range(smiles_len), repeat=2))
        C = len(comb) 

        features_1 = np.zeros((C, fps_dim))
        features_2 = np.zeros((C, fps_dim))
        pred_activity_score = np.zeros(smiles_len)

        for i, (idx1, idx2) in enumerate(comb):
            features_1[i, :] = fps[idx1, :]
            features_2[i, :] = fps[idx2, :]
        
        features_1 = torch.tensor(features_1, dtype=torch.float32)
        features_2 = torch.tensor(features_2, dtype=torch.float32)
        score_pred = feedback_model.forward(features_1, features_2).cpu().detach().numpy()

        for i, (idx1, idx2) in enumerate(comb):
            pred_activity_score[idx1] += (1 if score_pred[i] > 0.5 else 0)
        
        pred_activity_mean = pred_activity_score/smiles_len
        return pred_activity_mean

    elif feedback_type == "ranking":
        from itertools import combinations
        # Generate all unrepeated combinations of 3 out of len(new_queried_smiles)
        smiles_len, fps_dim = fps.shape
        comb = list(combinations(range(smiles_len), 3))
        C = len(comb)  # This is the number of combinations, which is binom(num_new_queried_smiles, 3)

        features_1 = np.zeros((C, fps_dim))
        features_2 = np.zeros((C, fps_dim))
        features_3 = np.zeros((C, fps_dim))

        label_proba = np.zeros(C)

        for i, (idx1, idx2, idx3) in enumerate(comb):
            features_1[i, :] = fps[idx1, :]
            features_2[i, :] = fps[idx2, :]
            features_3[i, :] = fps[idx3, :]

        features_1 = torch.tensor(features_1, dtype=torch.float32)
        features_2 = torch.tensor(features_2, dtype=torch.float32)
        features_3 = torch.tensor(features_3, dtype=torch.float32)

        score_pred = feedback_model.forward(features_1, features_2, features_3).cpu().detach().numpy()

        # Returning the ordinal ranks, 1, 2, 3
        # 1 is worst and 3 is best
        outputs_ranks = np.argsort(score_pred, axis=1) + 1 # shape (C, 3)

        # We need to normalize the ranks to 0-1, or 0.33, 0.66, 1.0
        outputs_ranks_normalized = outputs_ranks / 3

        # Initialize a list to store the scores
        pred_activity_score = np.zeros(smiles_len)

        count = len(list(combinations(range(smiles_len - 1), 2)))  # Number of times each index appears in the combinations, equal to binom(batch_size - 1, num_ranking-1)

        # Aggregate the scores
        for i, (idx1, idx2, idx3) in enumerate(comb):
            pred_activity_score[idx1] += outputs_ranks_normalized[i, 0]
            pred_activity_score[idx2] += outputs_ranks_normalized[i, 1]
            pred_activity_score[idx3] += outputs_ranks_normalized[i, 2]

        # Compute the average scores
        pred_activity_mean = pred_activity_score / count
    return pred_activity_mean

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

def thompson_sampling(feedback_type, feedback_model, scaffold_df, num_queries,
                            smiles, already_selected_indices, rng):
        
        predicted_scores = predict_proba_from_model(feedback_type, feedback_model, smiles)

        df_len = len(scaffold_df)
    
        # Get all possible indices
        all_indices = np.arange(df_len)
    
        # Remove already selected indices from the pool
        not_selected_indices = np.setdiff1d(all_indices, already_selected_indices)
    
        if feedback_type == "scoring":
            # Assuming we can get multiple samples from the model
            # For simplicity, using a normal distribution around predicted scores
            # In practice, this should be samples from the posterior distribution of the model

            samples = np.array([rng.normal(loc=predicted_scores[not_selected_indices], scale=0.1) for _ in range(num_samples)])
            
        elif feedback_type == "comparing":
            # Use a similar approach but ensure the distribution reflects pairwise comparison
      
            samples = np.array([rng.normal(loc=predicted_scores[not_selected_indices], scale=0.1) for _ in range(num_samples)])
            
        elif feedback_type == "ranking":
            # Similar approach, but for ranking, ensure the samples reflect rank uncertainties
      
            samples = np.array([rng.normal(loc=predicted_scores[not_selected_indices], scale=0.1) for _ in range(num_samples)])
        
        # Take the mean of predictions for Thompson Sampling
        preds_p = samples.mean(axis=0)
        
        # Sort the indices by the sampled values
        sorted_indices = np.argsort(preds_p)[::-1]
        
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
    elif acquisition == 'thompson':
        acquisition_func = thompson_sampling
    elif acquisition == 'greedy':
        acquisition_func = greedy_sampling
    elif acquisition == 'random':
        acquisition_func = random_sampling
    else:
        print("Warning: unknown acquisition criterion. Using random sampling.")
        acquisition_func = random_sampling
    new_queried_smiles_indices = acquisition_func(feedback_type, feedback_model, scaffold_df, num_queries, smiles, already_selected_indices, rng)
    return new_queried_smiles_indices
