import pickle

from typing import List

from reinvent_scoring.scoring.predictive_model.model_container import ModelContainer
from reinvent_scoring.scoring.component_parameters import ComponentParameters
from reinvent_scoring.scoring.score_components import BaseScoreComponent
from reinvent_scoring.scoring.score_summary import ComponentSummary
from reinvent_scoring.scoring.score_transformations import TransformationFactory
from reinvent_scoring.scoring.enums import TransformationTypeEnum, TransformationParametersEnum

import torch
import torch.nn as nn

class PredictivePropertyComponent(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.activity_model = self._load_model(parameters)
        self._transformation_function = self._assign_transformation(parameters.specific_parameters)

    def calculate_score(self, molecules: List, step=-1) -> ComponentSummary:
        score, raw_score = self._predict_and_transform(molecules)
        score_summary = ComponentSummary(total_score=score, parameters=self.parameters, raw_score=raw_score)
        return score_summary

    def _predict_and_transform(self, molecules: List):
        score = self.activity_model.predict(molecules, self.parameters.specific_parameters)
        transformed_score = self._apply_transformation(score, self.parameters.specific_parameters)
        return transformed_score, score

    def _load_model(self, parameters: ComponentParameters):
        activity_model = self._load_container(parameters)
        return activity_model

    def _load_container(self, parameters: ComponentParameters):
        model_path = self.parameters.specific_parameters.get(self.component_specific_parameters.MODEL_PATH, "")

        #initialize
        # out_dim = self.parameters.specific_parameters.get(self.component_specific_parameters.OUT_DIM, "")
        # dropout = self.parameters.specific_parameters.get(self.component_specific_parameters.DROPOUT, "")
        # input_dim = self.parameters.specific_parameters.get(self.component_specific_parameters.SIZE, "")
        # layers = self.parameters.specific_parameters.get(self.component_specific_parameters.LAYERS, "")
        
        out_dim = 1
        input_dim = 2048

        model = nn.Sequential(
            nn.Linear(input_dim, 512),  
            nn.ReLU(),
            nn.Linear(512, 128),  
            nn.ReLU(),
            nn.Linear(128, out_dim)   # Output a single strength score per entity (lambda)
        )
    
        # def forward(self, features_A, features_B):
        #     theta_A = self.fc(features_A) # Features A is Morgan Fingerprints of smiles 1
        #     theta_B = self.fc(features_B) # Features B is Morgan Fingerprints of smiles 2
        #     return torch.sigmoid(theta_A - theta_B)

        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        packaged_model = ModelContainer(model, parameters.specific_parameters)
        return packaged_model

    def _apply_transformation(self, predicted_activity, parameters: dict):
        transform_params = parameters.get(self.component_specific_parameters.TRANSFORMATION)
        if transform_params:
            activity = self._transformation_function(predicted_activity, transform_params)
        else:
            activity = predicted_activity
        return activity

    def _assign_transformation(self, specific_parameters: dict):
        transformation_type = TransformationTypeEnum()
        transform_params = specific_parameters.get(self.component_specific_parameters.TRANSFORMATION)
        if not transform_params:
            specific_parameters[self.component_specific_parameters.TRANSFORMATION] = {
                    TransformationParametersEnum.TRANSFORMATION_TYPE: transformation_type.NO_TRANSFORMATION
                }
        factory = TransformationFactory()
        transform_function = factory.get_transformation_function(transform_params)
        return transform_function
