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


class ScoreRegressionModelContainer(BaseModelContainer):
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
        Takes as input RDKit molecules and uses simple scoring model to predict activities of molecules
        :param molecules: This is a list of rdkit.Chem.Mol objects
        :param parameters: Those are descriptor-specific parameters.
        :return: numpy.array with activity predictions
        """
        return self.predict_from_mols(molecules, parameters)

    def predict_from_mols(self, molecules: List, parameters: dict):
        if len(molecules) == 0:
            return np.empty([])
        
        fps = self._molecules_to_descriptors(molecules, parameters)
        # fps1 is a list of np.array of shape (2048, )
        # Now we would convert them to 2D array
        fps = np.array(fps) # Shape (125, 2048)

        # Calculate preference scores for current_fps against each fps in fps2
        preference_scores = self.predict_from_fingerprints(fps)

        return preference_scores

    def predict_from_fingerprints(self, fps): # return only one prediction value that is the probability of the first molecule being more active than the second

        fps_torch_tensor = torch.tensor(fps, dtype=torch.float32)
        preds = self._activity_model.forward(fps_torch_tensor)
        
        final_preds = preds.cpu().detach().numpy()

        return final_preds

    def _load_descriptor(self, parameters: {}):
        descriptors = Descriptors()
        descriptor = descriptors.load_descriptor(parameters)
        return descriptor
