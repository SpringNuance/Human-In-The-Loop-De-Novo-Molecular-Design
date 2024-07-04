import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import torch

import sys
import os

from predict import predict_feedback_from_model

def compute_fingerprints(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    else:
        return np.zeros((2048,), dtype=int)

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
    
    predicted_scores = predict_feedback_from_model(feedback_type, feedback_model, smiles)
    # Convert to numpy array
    predicted_scores = np.array(predicted_scores)

    df_len = len(scaffold_df)

    # Get all possible indices
    all_indices = np.arange(df_len)

    # Remove already selected indices from the pool
    not_selected_indices = np.setdiff1d(all_indices, already_selected_indices)

    # Sort the indices by the predicted scores in descending order
    sorted_indices = np.argsort(predicted_scores[not_selected_indices])[::-1]

    # Select the top `num_queries` indices
    selected_indices = not_selected_indices[sorted_indices[:num_queries]]

    # flatten
    selected_indices = selected_indices.flatten()
    
    return selected_indices

def uncertainty_sampling(feedback_type, feedback_model, scaffold_df, num_queries,
                            smiles, already_selected_indices, rng):
        
    predicted_scores = predict_feedback_from_model(feedback_type, feedback_model, smiles)
    # Convert to np
    predicted_scores = np.array(predicted_scores)

    df_len = len(scaffold_df)

    # Get all possible indices
    all_indices = np.arange(df_len)

    # Remove already selected indices from the pool
    not_selected_indices = np.setdiff1d(all_indices, already_selected_indices)

    if feedback_type == "scoring":
        # Choose indices that minimize this value: abs(score - 0.5)
        # 0.5 is the most uncertain because the model is unsure whether smiles have drd2 or not

        # Sort the indices by the uncertainty
        sorted_indices = np.argsort(np.abs(predicted_scores[not_selected_indices] - 0.5))

    elif feedback_type == "comparing":
        # Choose indices that minimize this value: abs(score - 0.5)
        # 0.5 is the most uncertain because the model is unsure if this smiles is better or worse than others

        # Sort the indices by the uncertainty
        sorted_indices = np.argsort(np.abs(predicted_scores[not_selected_indices] - 0.5))

    elif feedback_type == "ranking":
        # Choose indices that minimize this value: abs(score - 0.5)
        # smiles with highest rates of being 2nd rank (0.5 normalized) in all combinations of rankings is the most uncertain 

        # Sort the indices by the uncertainty
        sorted_indices = np.argsort(np.abs(predicted_scores[not_selected_indices] - 0.5))

    # Select the top `num_queries` indices
    selected_indices = not_selected_indices[sorted_indices[:num_queries]] # shape (N, 1)

    # flatten
    selected_indices = selected_indices.flatten()
    
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
