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
        
        fps1 = self._molecules_to_descriptors(molecules, parameters)
        # fps1 is a list of np.array of shape (2048, )
        # Now we would convert them to 2D array
        fps1 = np.array(fps1) # Shape (125, 2048)
        
        pred_activity_mean = []

        for i, current_fps in enumerate(fps1):
            # excluse current_fps from fps1
            fps2 = np.delete(fps1, i, axis=0)  # Shape (124, 2048)
            
            # Repeat current_fps n-1 times to match the shape of fps2
            current_fps_repeated = np.tile(current_fps, (fps2.shape[0], 1))  # Shape (124, 2048)
            
            # Calculate preference scores for current_fps against each fps in fps2
            preference_scores = self.predict_from_fingerprints(current_fps_repeated, fps2)

            # Apply thresholding to preference scores
            rounded_scores = np.where(preference_scores > 0.5, 1, 0)
            
            # Calculate the mean activity
            activity = np.sum(rounded_scores) / len(fps2)
            pred_activity_mean.append(activity)
        
        # Convert the predicted_activity_mean to numpy
        pred_activity_mean = np.array(pred_activity_mean)
        return pred_activity_mean

    def predict_from_fingerprints(self, fps1, fps2): # return only one prediction value that is the probability of the first molecule being more active than the second

        fps1_torch_tensor = torch.tensor(fps1, dtype=torch.float32)
        fps2_torch_tensor = torch.tensor(fps2, dtype=torch.float32)
        preds = self._activity_model.forward(fps1_torch_tensor, fps2_torch_tensor)
        
        final_preds = preds.cpu().detach().numpy()

        return final_preds

    def _load_descriptor(self, parameters: {}):
        descriptors = Descriptors()
        descriptor = descriptors.load_descriptor(parameters)
        return descriptor
