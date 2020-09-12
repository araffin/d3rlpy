import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import create_encoder
from .q_functions import _reduce_ensemble


def create_reward_estimator(observation_shape,
                            action_size,
                            discrete_action=False,
                            n_ensembles=1,
                            use_batch_norm=False,
                            encoder_params={}):
    estimators = []
    for _ in range(n_ensembles):
        encoder = create_encoder(observation_shape,
                                 action_size,
                                 use_batch_norm=use_batch_norm,
                                 discrete_action=discrete_action,
                                 **encoder_params)
        estimator = RewardEstimator(encoder)
        estimators.append(estimator)
    return EnsembleRewardEstimator(estimators)


class RewardEstimator(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(encoder.feature_size, 1)

    def forward(self, x, action):
        h = self.encoder(x, action)
        return self.fc(h)

    def compute_error(self, x, action, reward, reduction='mean'):
        return F.mse_loss(self.forward(x, action), reward, reduction=reduction)


class EnsembleRewardEstimator(nn.Module):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = nn.ModuleList(estimators)

    def forward(self, x, action, reduction='mean'):
        values = []
        for estimator in self.estimators:
            values.append(estimator(x, action).view(1, x.shape[0], 1))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def compute_error(self, x, action, reward):
        loss_sum = 0.0
        for estimator in self.estimators:
            # bootstrapping
            loss = estimator.compute_error(x, action, reward, reduction='none')
            mask = torch.randint(0, 2, loss.shape, device=x.device)
            loss *= mask
            loss_sum += loss.mean()
        return loss_sum
