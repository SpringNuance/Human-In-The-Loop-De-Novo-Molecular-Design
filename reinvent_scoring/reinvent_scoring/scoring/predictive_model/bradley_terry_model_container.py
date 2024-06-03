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
        :type activity_model: stan type of model object
        :type model_type: can be "classification" or "regression"
        """
        self._molecules_to_descriptors = self._load_descriptor(specific_parameters) #ecfp_counts
        # Convert the molecules to fingerprints 
        self._activity_model = activity_model

    def predict(self, molecules: List, parameters: Dict) -> np.array:
        """
        Takes as input RDKit molecules and uses a stan model to predict activities.
        :param molecules: This is a list of rdkit.Chem.Mol objects
        :param parameters: Those are descriptor-specific parameters.
        :return: numpy.array with activity predictions
        """
        return self.predict_from_mols(molecules, parameters)

    def predict_from_mols(self, molecules: List, parameters: dict):
        if len(molecules) == 0:
            return np.empty([])
        fps1 = self._molecules_to_descriptors(molecules, parameters)
        
        activity_mean = []
        for current_fps in fps1:
            fps2 = fps1.copy()
            fps2.remove(current_fps)
            # excluse current_fps from fps1
            preference_list = []
            for fps in fps2:
                preference_list.append(self.predict_from_fingerprints(current_fps, fps)) # can be done in batches later
            activity = np.mean(preference_list)
            activity_mean.append(activity)
        return activity_mean

    def predict_from_fingerprints(self, fps1, fps2): # return only one prediction value that is the probability of the first molecule being more active than the second

        preds = self._activity_model(torch.tensor(fps1, dtype=torch.float32), torch.tensor(fps2, dtype=torch.float32))
        
        final_preds = preds.cpu().detach().numpy()

        return final_preds

    def _load_descriptor(self, parameters: {}):
        descriptors = Descriptors()
        descriptor = descriptors.load_descriptor(parameters)
        return descriptor
