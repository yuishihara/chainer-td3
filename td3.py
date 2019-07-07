from chainer import optimizers
from chainer import iterators
from chainer.datasets import tuple_dataset
from chainer.dataset import concat_examples
from chainer.distributions import Normal
import chainer.functions as F

from collections import deque

from models.td3_actor import TD3Actor
from models.td3_critic import TD3Critic

import numpy as np
import cupy as cp


class TD3(object):
    def __init__(self, state_dim, action_num, lr=1.0*1e-3, batch_size=100, device=-1):
        super(TD3, self).__init__()
        self._q1_optimizer = optimizers.Adam(alpha=lr)
        self._q2_optimizer = optimizers.Adam(alpha=lr)
        self._pi_optimizer = optimizers.Adam(alpha=lr)

        self._batch_size = batch_size
        self._q1 = TD3Critic(state_dim=state_dim, action_num=action_num)
        self._q2 = TD3Critic(state_dim=state_dim, action_num=action_num)
        self._pi = TD3Actor(action_num=action_num)
        self._target_q1 = TD3Critic(state_dim=state_dim, action_num=action_num)
        self._target_q2 = TD3Critic(state_dim=state_dim, action_num=action_num)
        self._target_pi = TD3Actor(action_num=action_num)

        self._q1_optimizer.setup(self._q1)
        self._q2_optimizer.setup(self._q2)
        self._pi_optimizer.setup(self._pi)

        xp = np if device < 0 else cp

        mean = xp.zeros(shape=(batch_size, action_num), dtype=xp.float32)
        sigma = xp.ones(shape=(batch_size, action_num), dtype=np.float32) * 0.2
        self._action_noise = Normal(loc=mean, scale=sigma)

        self._device = device
        self._clip_value = 0.5

    def train(self, replay_buffer, iterations, d):
        iterator = self._prepare_iterator(replay_buffer)
        for i in iterations:
            self._q1_optimizer.target.cleargrads()
            self._q2_optimizer.target.cleargrads()
            self._pi_optimizer.target.cleargrads()

            batch = iterator.next()
            s_current, action, r, s_next = concat_examples(
                batch, device=self._device)

            epsilon = F.clip(self._sample_action_noise(), -self._clip_value, self._clip_value)
            a_tilde = self._pi(s_current) + epsilon

            if i % d == 0:
                # update targets and pi
                pass

    def _sample_action_noise(self):
        return self._action_noise.sample()

    def _prepare_iterator(self, buffer):
        dataset = tuple_dataset.TupleDataset(buffer)
        return iterators.SerialIterator(dataset, self._batch_size)

if __name__ == "__main__":
    state_dim = 5
    action_num = 5
    td3 = TD3(state_dim=state_dim, action_num=action_num, batch_size=100)

    noise = td3._sample_action_noise()
    print('noise shape: ', noise.shape, ' noise: ', noise)

    mean = np.mean(noise.array)
    var = np.var(noise.array)

    print('mean: ', mean, ' sigma: ', np.sqrt(var))

    assert noise.shape == (td3._batch_size, action_num) 