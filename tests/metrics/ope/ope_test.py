import numpy as np
import os
import torch
import pickle

from unittest.mock import Mock
from tests.base_test import base_tester, base_update_tester
from d3rlpy.metrics.ope.torch.base import TorchImplBase
from d3rlpy.dataset import Episode, Transition, TransitionMiniBatch
from d3rlpy.logger import D3RLPyLogger


class DummyImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size):
        self.observation_shape = observation_shape
        self.action_size = action_size

    def save_model(self, fname):
        pass

    def load_model(self, fname):
        pass

    def predict(self, x, action):
        pass


class DummyAlgo:
    def __init__(self, action_size, discrete_action):
        self.action_size = action_size
        self.discrete_action = discrete_action

    def predict(self, x):
        if self.discrete_action:
            return np.random.randint(0, self.action_size, size=x.shape[0])
        return np.random.random((x.shape[0], self.action_size))


def ope_tester(ope, observation_shape, action_size):
    # dummy impl object
    impl = DummyImpl(observation_shape, action_size)

    base_tester(ope, impl, observation_shape, action_size)

    ope.impl = impl

    # check predict
    x = np.random.random((2, 3)).tolist()
    action = np.random.random((2, 3)).tolist()
    ref_reward = np.random.random((2, 1)).tolist()
    ref_log_prob = np.random.random((2, 1)).tolist()
    impl.predict = Mock(return_value=(ref_reward, ref_log_prob))
    reward, log_prob = ope.predict(x, action)
    assert reward == ref_reward
    assert log_prob == ref_log_prob
    impl.predict.assert_called_with(x, action)


def ope_update_tester(ope, observation_shape, action_size, discrete=False):
    transitions = base_update_tester(ope, observation_shape, action_size,
                                     discrete)

    # dummy algo
    algo = DummyAlgo(action_size, discrete)

    # evaluate a single episode
    evaluation = ope.evaluate_episode(algo, transitions)
    assert isinstance(evaluation, float)

    observations = np.random.random((100, ) + observation_shape)
    if discrete:
        actions = np.random.randint(action_size, size=100)
    else:
        actions = np.random.random((100, action_size))
    rewards = np.random.random(100)
    episode = Episode(observation_shape, action_size, observations, actions,
                      rewards)
    evaluation = ope(algo, [episode])
    assert isinstance(evaluation, float)


def impl_tester(impl, discrete, with_reward, with_log_prob):
    # setup implementation
    impl.build()

    observations = np.random.random((100, ) + impl.observation_shape)
    if discrete:
        actions = np.random.randint(impl.action_size, size=100)
    else:
        actions = np.random.random((100, impl.action_size))

    # check predict
    rewards, log_probs = impl.predict(observations, actions)
    if with_reward:
        assert rewards.shape == (100, 1)
    else:
        assert rewards is None
    if with_log_prob:
        assert log_probs.shape == (100, 1)
    else:
        assert log_probs is None


def torch_impl_tester(impl, discrete, with_reward=True, with_log_prob=True):
    impl_tester(impl, discrete, with_reward, with_log_prob)

    # check save_model and load_model
    impl.save_model(os.path.join('test_data', 'model.pt'))
    impl.load_model(os.path.join('test_data', 'model.pt'))
