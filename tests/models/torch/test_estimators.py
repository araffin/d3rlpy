import pytest
import torch
import torch.nn.functional as F

from d3rlpy.models.torch.estimators import create_reward_estimator
from d3rlpy.models.torch.estimators import RewardEstimator
from d3rlpy.models.torch.estimators import EnsembleRewardEstimator
from .model_test import check_parameter_updates, DummyEncoder


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_ensembles', [5])
@pytest.mark.parametrize('use_batch_norm', [False, True])
@pytest.mark.parametrize('discrete_action', [False, True])
@pytest.mark.parametrize('batch_size', [32])
def test_create_reward_estimator(observation_shape, action_size, n_ensembles,
                                 use_batch_norm, discrete_action, batch_size):
    estimator = create_reward_estimator(observation_shape, action_size,
                                        discrete_action, n_ensembles,
                                        use_batch_norm)

    assert isinstance(estimator, EnsembleRewardEstimator)
    assert len(estimator.estimators) == n_ensembles

    x = torch.rand((batch_size, ) + observation_shape)
    if discrete_action:
        action = torch.randint(0, action_size, size=(batch_size, 1))
    else:
        action = torch.rand(batch_size, action_size)
    reward = estimator(x, action)
    assert reward.shape == (batch_size, 1)


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('batch_size', [32])
def test_reward_estimator(feature_size, action_size, batch_size):
    encoder = DummyEncoder(feature_size, action_size, True)
    estimator = RewardEstimator(encoder)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = estimator(x, action)
    assert y.shape == (batch_size, 1)

    # check error
    reward = torch.rand(batch_size, 1)
    loss = estimator.compute_error(x, action, reward)
    assert torch.allclose(loss, F.mse_loss(y, reward))

    # check layer connection
    check_parameter_updates(estimator, (x, action, reward))


@pytest.mark.parametrize('feature_size', [100])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('n_ensembles', [5])
@pytest.mark.parametrize('batch_size', [32])
def test_ensemble_reward_estimator(feature_size, action_size, n_ensembles,
                                   batch_size):
    encoder = DummyEncoder(feature_size, action_size, True)
    estimators = []
    for _ in range(n_ensembles):
        estimators.append(RewardEstimator(encoder))

    estimator = EnsembleRewardEstimator(estimators)

    # check output shape
    x = torch.rand(batch_size, feature_size)
    action = torch.rand(batch_size, action_size)
    y = estimator(x, action)
    assert y.shape == (batch_size, 1)

    # check error
    reward = torch.rand(batch_size, 1)
    loss = estimator.compute_error(x, action, reward)

    # check layer connection
    check_parameter_updates(estimator, (x, action, reward))
