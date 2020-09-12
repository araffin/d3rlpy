import pytest

from d3rlpy.metrics.ope.dm import DM
from .ope_test import ope_tester, ope_update_tester


@pytest.mark.parametrize('observation_shape', [(100, )])
@pytest.mark.parametrize('action_size', [2])
@pytest.mark.parametrize('scaler', [None, 'standard'])
@pytest.mark.parametrize('discrete_action', [False, True])
def test_dm(observation_shape, action_size, scaler, discrete_action):
    dm = DM(scaler=scaler, discrete_action=discrete_action)
    ope_tester(dm, observation_shape, action_size)
    ope_update_tester(dm, observation_shape, action_size, discrete_action)
