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

import importlib.util

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
    
    ###############################
    # METHOD CHANGED BY XUAN BINH #
    ###############################

    def _load_container(self, parameters: ComponentParameters):
        model_pretrained_path = self.parameters.specific_parameters["model_pretrained_path"]
        model_name = self.parameters.specific_parameters["model_name"]
        model_definition_path = self.parameters.specific_parameters["model_definition_path"]
        
        print("\n(predictive_property_component.py) _load_container is called")

        if model_name == "score_regression":
            # This code import the module file by using absolute path
            spec = importlib.util.spec_from_file_location(model_name, model_definition_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            # Constructing the Score Regression model
            feedback_model = model_module.ScoreRegressionModel()
            # Loading the state dict
            feedback_model.load_state_dict(torch.load(model_pretrained_path))
            activity_model = ModelContainer(feedback_model, parameters.specific_parameters)
        elif model_name == "bradley_terry":
            # This code import the module file by using absolute path
            spec = importlib.util.spec_from_file_location(model_name, model_definition_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            # Constructing the Bradley Terry model
            feedback_model = model_module.BradleyTerryModel()
            # Loading the state dict
            feedback_model.load_state_dict(torch.load(model_pretrained_path))
            activity_model = ModelContainer(feedback_model, parameters.specific_parameters)
        elif model_name == "rank_listnet":
            # This code import the module file by using absolute path
            spec = importlib.util.spec_from_file_location(model_name, model_definition_path)
            model_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_module)
            # Constructing the Rank ListNet model
            feedback_model = model_module.RankListNetModel()
            # Loading the state dict
            feedback_model.load_state_dict(torch.load(model_pretrained_path))
            activity_model = ModelContainer(feedback_model, parameters.specific_parameters)
        else:
            raise ValueError(f"Model {model_name} not recognized")
        
        print("\n(predictive_property_component.py) Model has been loaded successfully from path: ", model_pretrained_path)
        
        return activity_model

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
