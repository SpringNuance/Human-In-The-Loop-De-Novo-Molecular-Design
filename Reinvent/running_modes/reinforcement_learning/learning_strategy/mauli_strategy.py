from running_modes.reinforcement_learning.configurations.learning_strategy_configuration import LearningStrategyConfiguration
from running_modes.reinforcement_learning.learning_strategy import BaseLearningStrategy


class MAULIStrategy(BaseLearningStrategy):

    def __init__(self, critic_model, optimizer, configuration: LearningStrategyConfiguration, logger=None):
        """
        TODO: Provide description of the current strategy
        """
        super().__init__(critic_model, optimizer, configuration, logger)

        #TODO: Create a StrategySpecificEnums
        self._sigma = self._configuration.parameters.get("sigma", 120)

    def _calculate_loss(self, scaffold_batch, decorator_batch, score, actor_nlls):

        critic_nlls = self.critic_model.likelihood(*scaffold_batch, *decorator_batch)
        negative_critic_nlls = -critic_nlls
        negative_actor_nlls = -actor_nlls
        augmented_nlls = negative_critic_nlls + self._sigma * self._to_tensor(score)
        loss = -(sum(augmented_nlls) * sum(negative_actor_nlls))
        return loss, negative_actor_nlls, negative_critic_nlls, augmented_nlls
