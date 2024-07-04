import os
import json
import math
import torch

import pandas as pd
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict
from reinvent_chemistry.descriptors import Descriptors
from reinvent_scoring.scoring.predictive_model.base_model_container import BaseModelContainer
from itertools import combinations

class RankListNetContainer(BaseModelContainer):
    def __init__(self, activity_model, specific_parameters):
        """
        :type activity_model: Pytorch type of model object
        :type model_type: "classification"
        """
        self._molecules_to_descriptors = self._load_descriptor(specific_parameters) #ecfp_counts
        # Convert the molecules to fingerprints 
        self._activity_model = activity_model

    def predict(self, molecules: List, parameters: Dict) -> np.array:
        """
        Takes as input RDKit molecules and uses Rank ListNet model to rank the activities of three molecules
        :param molecules: This is a list of rdkit.Chem.Mol objects
        :param parameters: Those are descriptor-specific parameters.
        :return: numpy.array with activity predictions
        """
        return self.predict_from_mols(molecules, parameters)

    def predict_from_mols(self, molecules: List, parameters: dict):
        if len(molecules) == 0:
            return np.empty([])
        
        num_ranking = 3

        features = self._molecules_to_descriptors(molecules, parameters)
        # features_1 is a list of np.array of shape (2048, )
        # Now we would convert them to 2D array
        features = np.array(features) # Shape (batch_size, 2048)
        batch_size, features_dim = features.shape
        
        # Generate all combinations of 3 out of 64
        comb = list(combinations(range(batch_size), num_ranking))
        num_comb = len(comb)  # This is the number of combinations, which is binom(batch_size, 3)

        # Initialize three tensors of shape (C, 2048)
        features_1 = np.zeros((num_comb, features_dim))
        features_2 = np.zeros((num_comb, features_dim))
        features_3 = np.zeros((num_comb, features_dim))

        # Fill the tensors with the corresponding rows from original_tensor
        for i, (idx1, idx2, idx3) in enumerate(comb):
            features_1[i, :] = features[idx1, :]
            features_2[i, :] = features[idx2, :]
            features_3[i, :] = features[idx3, :]

        # Forward pass to get the softmax scores (C3)
        outputs_scores = self.predict_from_fingerprints(features_1, features_2, features_3) # shape (C, 3)

        # Returning the ordinal ranks, 0, 1, 2
        # 0 is worst and 2 is best
        outputs_ranks = np.argsort(outputs_scores, axis=1) # shape (C, 3)

        # We need to normalize the ranks to 0-1, or 0.0, 0.5, 1.0
        outputs_ranks_normalized = outputs_ranks / (num_ranking - 1)

        # Initialize a list to store the scores
        pred_label_proba = np.zeros(batch_size)

        count = len(list(combinations(range(batch_size - 1), num_ranking - 1)))  # Number of times each index appears in the combinations, equal to binom(batch_size - 1, num_ranking-1)

        # Aggregate the scores
        for i, (idx1, idx2, idx3) in enumerate(comb):
            pred_label_proba[idx1] += outputs_ranks_normalized[i, 0]
            pred_label_proba[idx2] += outputs_ranks_normalized[i, 1]
            pred_label_proba[idx3] += outputs_ranks_normalized[i, 2]

        # Compute the average scores
        pred_label_proba = [score / count for score in pred_label_proba]

        return pred_label_proba

    def predict_from_fingerprints(self, features_1, features_2, features_3): 

        features_1_torch_tensor = torch.tensor(features_1, dtype=torch.float32)
        features_2_torch_tensor = torch.tensor(features_2, dtype=torch.float32)
        features_3_torch_tensor = torch.tensor(features_3, dtype=torch.float32)

        preds = self._activity_model.forward(features_1_torch_tensor, features_2_torch_tensor, features_3_torch_tensor)
        
        final_preds = preds.cpu().detach().numpy()

        return final_preds

    def _load_descriptor(self, parameters: {}):
        descriptors = Descriptors()
        descriptor = descriptors.load_descriptor(parameters)
        return descriptor
