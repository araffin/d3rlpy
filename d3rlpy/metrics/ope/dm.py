import numpy as np

from d3rlpy.dataset import TransitionMiniBatch
from .base import OPEBase
from .torch.dm_impl import DMImpl


class DM(OPEBase):
    def __init__(self,
                 n_epochs=30,
                 batch_size=100,
                 n_frames=1,
                 learning_rate=1e-3,
                 eps=1e-8,
                 weight_decay=1e-4,
                 n_ensembles=1,
                 use_batch_norm=False,
                 discrete_action=False,
                 scaler=None,
                 augmentation=[],
                 encoder_params={},
                 use_gpu=False,
                 impl=None,
                 **kwargs):
        super().__init__(n_epochs, batch_size, n_frames, scaler, augmentation,
                         use_gpu)
        self.learning_rate = learning_rate
        self.eps = eps
        self.weight_decay = weight_decay
        self.n_ensembles = n_ensembles
        self.use_batch_norm = use_batch_norm
        self.discrete_action = discrete_action
        self.encoder_params = encoder_params
        self.impl = impl

    def create_impl(self, observation_shape, action_size):
        self.impl = DMImpl(observation_shape=observation_shape,
                           action_size=action_size,
                           learning_rate=self.learning_rate,
                           n_ensembles=self.n_ensembles,
                           eps=self.eps,
                           weight_decay=self.weight_decay,
                           use_batch_norm=self.use_batch_norm,
                           discrete_action=self.discrete_action,
                           use_gpu=self.use_gpu,
                           scaler=self.scaler,
                           augmentation=self.augmentation,
                           encoder_params=self.encoder_params)
        self.impl.build()

    def evaluate_episode(self, algo, transitions):
        batch = TransitionMiniBatch(transitions, self.n_frames)
        observations = batch.observations
        actions = algo.predict(observations)
        rewards, _ = self.predict(observations, actions)
        return float(np.sum(rewards))

    def update(self, epoch, total_step, batch):
        loss = self.impl.update_estimator(batch.observations, batch.actions,
                                          batch.next_rewards)
        return (loss, )

    def _get_loss_labels(self):
        return ['estimator_loss']
