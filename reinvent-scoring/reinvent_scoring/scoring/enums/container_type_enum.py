
from dataclasses import dataclass

# This decorator is applied to the ContainerType class. 
# The frozen=True parameter makes the class immutable, meaning once 
# an instance of the class is created, its attributes cannot be changed. 
#If you try to modify one of its attributes, Python will raise an AttributeError.

@dataclass(frozen=True)
class ContainerType:
    SCIKIT_CONTAINER = "scikit_container"
    OPTUNA_CONTAINER = "optuna_container"
    STAN_CONTAINER = "stan_container"
    TORCH_CONTAINER = "torch_container"
    ENSEMBLE_CONTAINER = "ensemble_container"
    SCORE_REGRESSION_CONTAINER = "score_regression_container"
    BRADLEY_TERRY_CONTAINER = "bradley_terry_container"
    RANK_LISTNET_CONTAINER = "rank_listnet_container"
