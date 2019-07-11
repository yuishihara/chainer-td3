import chainer
import chainer.functions as F
from chainer import optimizers
from chainer import iterators
from chainer.datasets import tuple_dataset
from chainer.dataset import concat_examples
from chainer.distributions import Normal

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

        if not device < 0:
            self._q1.to_gpu()
            self._q2.to_gpu()
            self._pi.to_gpu()
            self._target_q1.to_gpu()
            self._target_q2.to_gpu()
            self._target_pi.to_gpu()

        self._q1_optimizer.setup(self._q1)
        self._q2_optimizer.setup(self._q2)
        self._pi_optimizer.setup(self._pi)

        xp = np if device < 0 else cp

        mean = xp.zeros(shape=(action_num), dtype=xp.float32)
        sigma = xp.ones(shape=(action_num), dtype=np.float32)
        self._exploration_noise = Normal(loc=mean, scale=sigma * 0.1)
        self._action_noise = Normal(loc=mean, scale=sigma * 0.2)

        self._device = device
        self._initialized = False

        self._action_num = action_num

    def act_with_policy(self, env, s):
        s = np.float32(s)
        state = chainer.Variable(np.reshape(s, newshape=(1, ) + s.shape))
        if not self._device < 0:
            state.to_gpu()

        a = self._pi(state)
        if not self._device < 0:
            a.to_cpu()
        a = np.squeeze(a.data, axis=0)

        noise = self._sample_exploration_noise(shape=(1))
        assert a.shape == noise.shape
        s_next, r, done, _ = env.step(a + noise)

        s_next = np.float32(s_next)
        a = np.float32(a)
        r = np.float32(r)
        return s, a, r, s_next, done

    def act_randomly(self, env, s):
        s = np.float32(s)

        a = env.action_space.sample()
        s_next, r, done, _ = env.step(a)

        s_next = np.float32(s_next)
        a = np.float32(a)
        r = np.float32(r)
        return s, a, r, s_next, done

    def evaluate_policy(self, env):
        s = env.reset()
        rewards = []
        episode_reward = 0
        for _ in range(10):
            s = chainer.Variable(np.reshape(s, newshape=(1, ) + s.shape))
            if not self._device < 0:
                s.to_gpu()

            a = self._pi(s)
            if not self._device < 0:
                a.to_cpu()
            a = np.squeeze(a.data, axis=0)
            s, r, done, _ = env.step(a)
            episode_reward += r
            if done:
                rewards.append(episode_reward)
                episode_reward = 0
                s = env.reset()
        return rewards

    def train(self, replay_buffer, iterations, d, clip_value, gamma, tau):
        if not self._initialized:
            self._initialize_target_networks()
        iterator = self._prepare_iterator(replay_buffer)
        for i in iterations:
            batch = iterator.next()
            s_current, action, r, s_next, non_terminal = concat_examples(
                batch, device=self._device)

            epsilon = F.clip(self._sample_action_noise(shape=(self._batch_size)),
                             -clip_value, clip_value)
            target_pi = self._target_pi(s_current)
            assert target_pi.shape == epsilon.shape
            a_tilde = target_pi + epsilon

            target_q1 = self._target_q1(s_next, a_tilde)
            target_q2 = self._target_q2(s_next, a_tilde)

            y = r + gamma * non_terminal * F.min(target_q1, target_q2)
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

    def _sample_action_noise(self, shape):
        return self._action_noise.sample(shape)

    def _sample_exploration_noise(self, shape):
        return self._exploration_noise.sample(shape)

    def _prepare_iterator(self, buffer):
        dataset = tuple_dataset.TupleDataset(buffer)
        return iterators.SerialIterator(dataset, self._batch_size)


if __name__ == "__main__":
    state_dim = 5
    action_num = 5
    batch_size = 100
    td3 = TD3(state_dim=state_dim, action_num=action_num,
              batch_size=batch_size)

    a_noise = td3._sample_action_noise(shape=(batch_size))
    print('action noise shape: ', a_noise.shape, ' noise: ', a_noise)
    mean = np.mean(a_noise.array)
    var = np.var(a_noise.array)
    print('mean: ', mean, ' sigma: ', np.sqrt(var))
    assert a_noise.shape == (td3._batch_size, action_num)

    e_noise = td3._sample_action_noise(shape=(1))
    print('exploration noise shape: ', e_noise.shape, ' noise: ', e_noise)
    mean = np.mean(e_noise.array)
    var = np.var(e_noise.array)
    print('mean: ', mean, ' sigma: ', np.sqrt(var))
    assert a_noise.shape == (1, action_num)

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
