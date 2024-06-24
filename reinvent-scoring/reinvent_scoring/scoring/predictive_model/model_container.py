from typing import Dict, Any

from reinvent_scoring.scoring.enums.container_type_enum import ContainerType
from reinvent_scoring.scoring.enums.component_specific_parameters_enum import ComponentSpecificParametersEnum

from reinvent_scoring.scoring.predictive_model.base_model_container import BaseModelContainer
from reinvent_scoring.scoring.predictive_model.optuna_container import OptunaModelContainer
from reinvent_scoring.scoring.predictive_model.scikit_model_container import ScikitModelContainer

from reinvent_scoring.scoring.predictive_model.bradley_terry_model_container import BradleyTerryModelContainer

class ModelContainer:

    def __new__(cls, activity_model: Any, specific_parameters: Dict) -> BaseModelContainer:

        print("\nreinvent_scoring/scoring/predictive_model/model_container.py is called")
        
        _component_specific_parameters = ComponentSpecificParametersEnum() # BRADLEY_TERRY = "bradley_terry"
        _container_type = ContainerType() # BRADLEY_TERRY_CONTAINER = "bradley_terry_container"
        
        # This means that the container type is not specified, so we default to scikit
        container_type = specific_parameters.get(_component_specific_parameters.CONTAINER_TYPE,
                                                 _container_type.SCIKIT_CONTAINER) # bradley_terry_container
        
        print(f"The container_type is {container_type}") # bradley_terry_container
        
        # Given two molecules, what is the probability that the first molecule is better than the second molecule?
        # Better here can be defined in anyway as you want

        if container_type == _container_type.BRADLEY_TERRY_CONTAINER: # bradley_terry_container
            print(f"BradleyTerryModelContainer from reinvent_scoring/scoring/predictive_model/bradley_terry_model_container.py has been loaded")
            container_instance = BradleyTerryModelContainer(activity_model, specific_parameters)
        
        # if container_type == _container_type.RANDOM_FOREST_CONTAINER: # random_forest_container
        #     container_instance = RandomForestModelContainer(activity_model, specific_parameters)
        
        elif container_type == _container_type.SCIKIT_CONTAINER:
            container_instance = ScikitModelContainer(activity_model,
                                                      specific_parameters[_component_specific_parameters.SCIKIT],
                                                      specific_parameters)
        else:
            # TODO: possibly a good spot for error try/catching
            container_instance = OptunaModelContainer(activity_model)

        return container_instance
