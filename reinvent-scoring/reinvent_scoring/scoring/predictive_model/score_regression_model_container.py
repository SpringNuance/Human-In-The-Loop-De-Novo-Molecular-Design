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
        
        features = self._molecules_to_descriptors(molecules, parameters)
        # features1 is a list of np.array of shape (2048, )
        # Now we would convert them to 2D array
        features = np.array(features) # Shape (125, 2048)
        
        pred_label_proba = self.predict_from_fingerprints(features)

        return pred_label_proba

    def predict_from_fingerprints(self, features): # return only one prediction value that is the probability of the first molecule being more active than the second

        features_tensor = torch.tensor(features, dtype=torch.float32)
        pred_label_proba = self._activity_model.forward(features_tensor)
        
        pred_label_proba = pred_label_proba.cpu().detach().numpy()

        return pred_label_proba

    def _load_descriptor(self, parameters: {}):
        descriptors = Descriptors()
        descriptor = descriptors.load_descriptor(parameters)
        return descriptor
