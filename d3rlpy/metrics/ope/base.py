import numpy as np

from abc import abstractmethod
from d3rlpy.base import ImplBase, LearnableBase
from d3rlpy.dataset import Transition, TransitionMiniBatch


class OPEImplBase(ImplBase):
    @abstractmethod
    def predict(self, x, action):
        pass


class OPEBase(LearnableBase):
    def predict(self, x, action):
        """ Returns predicted reward and log probability of action.

        Args:
            x (numpy.ndarray): observation.
            action (numpy.ndarray): action.

        Returns:
            tuple:
                tuple of predicted observation and log probability of action.

        """
        rewards, log_probs = self.impl.predict(x, action)
        return rewards, log_probs

    def evaluate_episode(self, algo, transitions):
        """ Returns result of off-policy evaluation in a single episode.

        Args:
            algo (d3rlpy.algos.AlgoBase): algorithm.
            transitions (list(d3rlpy.dataset.Transition)):
                all transitions in a single episode.

        Returns:
            float: result of off-policy evaluation.

        """
        raise NotImplementedError

    def evaluate(self, algo, episodes):
        """ Returns result of off-policy evaluation.

        Args:
            algo (d3rlpy.algos.AlgoBase): algorithm.
            episodes (list(d3rlpy.dataset.Episode)): list of episodes.

        Returns:
            float: mean off-policy evaluation result.

        """
        evaluations = []
        for episode in episodes:
            transitions = episode.transitions
            evaluations.append(self.evaluate_episode(algo, transitions))
        return float(np.mean(evaluations))

    def __call__(self, algo, episodes):
        return self.evaluate(algo, episodes)
