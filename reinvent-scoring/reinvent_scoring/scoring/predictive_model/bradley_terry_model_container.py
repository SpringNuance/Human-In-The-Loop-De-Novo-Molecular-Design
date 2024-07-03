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

from itertools import product

class BradleyTerryModelContainer(BaseModelContainer):
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
        Takes as input RDKit molecules and uses Bradley Terry model to predict activities of molecules compared to others
        :param molecules: This is a list of rdkit.Chem.Mol objects
        :param parameters: Those are descriptor-specific parameters.
        :return: numpy.array with activity predictions
        """
        return self.predict_from_mols(molecules, parameters)

    def predict_from_mols(self, molecules: List, parameters: dict):
        if len(molecules) == 0:
            return np.empty([])
        
        features = self._molecules_to_descriptors(molecules, parameters)
        # features_1 is a list of np.array of shape (2048, )
        # Now we would convert them to 2D array
        features = np.array(features) # Shape (125, 2048)
        
        batch_size, features_dim = features.shape
        # Generate all repeated combinations of 2 out of len(smiles)
        comb = list(product(range(batch_size), repeat=2))

        # Remove all combinations where the same molecule is compared to itself
        comb = [(i, j) for (i, j) in comb if i != j]
        num_comb = len(comb) 

        features_1 = np.zeros((num_comb, features_dim))
        features_2 = np.zeros((num_comb, features_dim))

        # Fill the tensors with the corresponding rows from original_tensor
        for i, (idx1, idx2) in enumerate(comb):
            features_1[i, :] = features[idx1, :]
            features_2[i, :] = features[idx2, :]
        
        compare_proba = self.predict_from_fingerprints(features_1, features_2) # shape (C, 1)

        # If value > 0.5 then 1 else 0
        compare_binary = np.where(compare_proba > 0.5, 1, 0)

        # Initialize a list to store the scores
        pred_label_proba = np.zeros(batch_size)

        # Aggregate the scores, exluding comparing smiles with itself
        for i, (idx1, idx2) in enumerate(comb):
            pred_label_proba[idx1] += compare_binary[i]

        # Compute the average scores
        pred_label_proba = pred_label_proba / (batch_size - 1)

        return pred_label_proba











        pred_activity_mean = []

        for i, current_fps in enumerate(features_1):
            # excluse current_fps from features_1
            features_2 = np.delete(features_1, i, axis=0)  # Shape (124, 2048)
            
            # Repeat current_fps n-1 times to match the shape of features_2
            current_fps_repeated = np.tile(current_fps, (features_2.shape[0], 1))  # Shape (124, 2048)
            
            # Calculate preference scores for current_fps against each fps in features_2
            preference_scores = self.predict_from_fingerprints(current_fps_repeated, features_2)

            # Apply thresholding to preference scores
            rounded_scores = np.where(preference_scores > 0.5, 1, 0)
            
            # Calculate the mean activity
            activity = np.sum(rounded_scores) / len(features_2)
            pred_activity_mean.append(activity)
        
        # Convert the predicted_activity_mean to numpy
        pred_activity_mean = np.array(pred_activity_mean)
        return pred_activity_mean

    def predict_from_fingerprints(self, features_1, features_2): # return only one prediction value that is the probability of the first molecule being more active than the second

        features_1_torch_tensor = torch.tensor(features_1, dtype=torch.float32)
        features_2_torch_tensor = torch.tensor(features_2, dtype=torch.float32)
        preds = self._activity_model.forward(features_1_torch_tensor, features_2_torch_tensor)
        
        final_preds = preds.cpu().detach().numpy()

        return final_preds

    def _load_descriptor(self, parameters: {}):
        descriptors = Descriptors()
        descriptor = descriptors.load_descriptor(parameters)
        return descriptor
