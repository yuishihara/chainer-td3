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
# import cupy as cp


class TD3(object):
    def __init__(self, state_dim, action_num, lr=1.0*1e-3, batch_size=100, device=-1):
        super(TD3, self).__init__()
        self._q1_optimizer = optimizers.Adam(alpha=lr)
        self._q2_optimizer = optimizers.Adam(alpha=lr)
        self._pi_optimizer = optimizers.Adam(alpha=lr)

        self._batch_size = batch_size
        self._q1 = TD3Critic(state_dim=state_dim, action_num=action_num)
        self._q2 = TD3Critic(state_dim=state_dim, action_num=action_num)
        self._pi = TD3Actor(state_dim=state_dim, action_num=action_num)
        self._target_q1 = TD3Critic(state_dim=state_dim, action_num=action_num)
        self._target_q2 = TD3Critic(state_dim=state_dim, action_num=action_num)
        self._target_pi = TD3Actor(state_dim=state_dim, action_num=action_num)

        self._q1_optimizer.setup(self._q1)
        self._q2_optimizer.setup(self._q2)
        self._pi_optimizer.setup(self._pi)

        xp = np if device < 0 else cp

        mean = xp.zeros(shape=(batch_size, action_num), dtype=xp.float32)
        sigma = xp.ones(shape=(batch_size, action_num), dtype=np.float32) * 0.2
        self._action_noise = Normal(loc=mean, scale=sigma)

        self._device = device
        self._initialized = False

    def train(self, replay_buffer, iterations, d, clip_value, gamma, tau):
        if not self._initialized:
            self._initialize_target_networks()
        iterator = self._prepare_iterator(replay_buffer)
        for i in iterations:
            batch = iterator.next()
            s_current, action, r, s_next, done = concat_examples(
                batch, device=self._device)

            epsilon = F.clip(self._sample_action_noise(),
                             -clip_value, clip_value)
            a_tilde = self._target_pi(s_current) + epsilon

            target_q1 = self._target_q1(s_next, a_tilde)
            target_q2 = self._target_q2(s_next, a_tilde)

            y = r + gamma * done * F.min(target_q1, target_q2)
            # Remove reference to avoid unexpected gradient update
            y.unchain()

            q1 = self._q1(s_current, action)
            q1_loss = F.mean_squared_error(y, q1)
            q2 = self._q2(s_current, action)
            q2_loss = F.mean_squared_error(y, q2)
            critic_loss = q1_loss + q2_loss

            self._q1_optimizer.target.cleargrads()
            self._q2_optimizer.target.cleargrads()
            critic_loss.backward()
            critic_loss.unchain_backward()
            self._q1_optimizer.update()
            self._q2_optimizer.update()

            if i % d == 0:
                a = self._pi(s_current)
                q1 = self._q1(s_current, a)

                pi_loss = -F.mean(q1)

                self._pi_optimizer.target.cleargrads()
                pi_loss.backward()
                pi_loss.unchain_backward()
                self._pi_optimizer.update()

                self._update_target_network(self._target_q1, self._q1, tau)
                self._update_target_network(self._target_q2, self._q2, tau)
                self._update_target_network(self._target_pi, self._pi, tau)

    def _initialize_target_networks(self):
        self._update_target_network(self._target_q1, self._q1, 1.0)
        self._update_target_network(self._target_q2, self._q2, 1.0)
        self._update_target_network(self._target_pi, self._pi, 1.0)
        self._initialized = True

    def _update_target_network(self, target, origin, tau):
        for target_param, origin_param in zip(target.params(), origin.params()):
            target_param.data = tau * origin_param.data + \
                (1.0 - tau) * target_param.data

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

    for target_param, origin_param in zip(td3._target_pi.params(), td3._pi.params()):
        print('before target param shape: ', target_param.shape,
              ' origin param shape: ', origin_param.shape)
        print('target: ', target_param.data, ' origin: ', origin_param.data)
        is_equal = np.array_equal(target_param.data, origin_param.data)
        print('is target and origin equal?: ', is_equal)

    td3._update_target_network(td3._target_pi, td3._pi, tau=0.1)

    for target_param, origin_param in zip(td3._target_pi.params(), td3._pi.params()):
        print('after target param shape: ', target_param.shape,
              ' origin param shape: ', origin_param.shape)
        print('target: ', target_param.data, ' origin: ', origin_param.data)
        is_equal = np.array_equal(target_param.data, origin_param.data)
        print('is target and origin equal?: ', is_equal)
