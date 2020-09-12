import pytest

from d3rlpy.augmentation import AugmentationPipeline
from d3rlpy.metrics.ope.torch.dm_impl import DMImpl
from tests.base_test import DummyScaler
from tests.metrics.ope.ope_test import torch_impl_tester


@pytest.mark.parametrize('observation_shape', [(100, ), (4, 84, 84)])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('learning_rate', [1e-3])
@pytest.mark.parametrize('n_ensembles', [5])
@pytest.mark.parametrize('eps', [1e-8])
@pytest.mark.parametrize('weight_decay', [1e-4])
@pytest.mark.parametrize('use_batch_norm', [True, False])
@pytest.mark.parametrize('discrete_action', [False, True])
@pytest.mark.parametrize('scaler', [None, DummyScaler()])
@pytest.mark.parametrize('augmentation', [AugmentationPipeline()])
@pytest.mark.parametrize('encoder_params', [{}])
def test_dm_impl(observation_shape, action_size, learning_rate, n_ensembles,
                 eps, weight_decay, use_batch_norm, discrete_action, scaler,
                 augmentation, encoder_params):
    impl = DMImpl(observation_shape,
                  action_size,
                  learning_rate,
                  n_ensembles,
                  eps,
                  weight_decay,
                  use_batch_norm,
                  discrete_action,
                  use_gpu=False,
                  scaler=scaler,
                  augmentation=augmentation,
                  encoder_params=encoder_params)
    torch_impl_tester(impl, discrete=discrete_action, with_log_prob=False)
