from d3rlpy.datasets import get_pybullet
from d3rlpy.algos import CQL
from d3rlpy.metrics.ope import DM
from d3rlpy.metrics.scorer import ope_reward_prediction_error_scorer
from d3rlpy.metrics.scorer import evaluate_on_environment
from sklearn.model_selection import train_test_split

dataset, env = get_pybullet('HopperBulletEnv-v0')

train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# train reward estimator
dm = DM(n_epochs=30, use_gpu=True)
dm.fit(train_episodes,
       eval_episodes=test_episodes,
       scorers={'reward_error': ope_reward_prediction_error_scorer})

# train algorithm
cql = CQL(n_epochs=100, use_gpu=True)
cql.fit(train_episodes,
        eval_episodes=test_episodes,
        scorers={
            'environment': evaluate_on_environment(env),
            'dm': dm
        })
