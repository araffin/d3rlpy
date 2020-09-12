from torch.optim import Adam
from .base import TorchImplBase
from d3rlpy.models.torch.estimators import create_reward_estimator
from d3rlpy.algos.torch.utility import torch_api, train_api


class DMImpl(TorchImplBase):
    def __init__(self, observation_shape, action_size, learning_rate,
                 n_ensembles, eps, weight_decay, use_batch_norm,
                 discrete_action, use_gpu, scaler, augmentation,
                 encoder_params):
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.n_ensembles = n_ensembles
        self.eps = eps
        self.weight_decay = weight_decay
        self.use_batch_norm = use_batch_norm
        self.discrete_action = discrete_action
        self.use_gpu = use_gpu
        self.scaler = scaler
        self.augmentation = augmentation
        self.encoder_params = encoder_params

    def build(self):
        self._build_estimator()

        if self.use_gpu:
            self.to_gpu(self.use_gpu)
        else:
            self.to_cpu()

        self._build_estimator_optim()

    def _build_estimator(self):
        self.estimator = create_reward_estimator(
            self.observation_shape,
            self.action_size,
            discrete_action=self.discrete_action,
            n_ensembles=self.n_ensembles,
            use_batch_norm=self.use_batch_norm,
            encoder_params=self.encoder_params)

    def _build_estimator_optim(self):
        self.estimator_optim = Adam(self.estimator.parameters(),
                                    self.learning_rate,
                                    eps=self.eps,
                                    weight_decay=self.weight_decay)

    def _predict(self, x, action):
        reward = self.estimator(x, action)
        return reward, None

    @train_api
    @torch_api
    def update_estimator(self, observations, actions, rewards):
        if self.scaler:
            observations = self.scaler.transform(observations)

        observations = self.augmentation.transform(observations)

        loss = self.estimator.compute_error(observations, actions, rewards)

        self.estimator_optim.zero_grad()
        loss.backward()
        self.estimator_optim.step()

        return loss.cpu().detach().numpy()
