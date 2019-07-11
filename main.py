import gym

import argparse

import collections
import os

import numpy as np

from researchutils import files
from researchutils.chainer import serializers

from td3 import TD3


def build_env(args):
    return gym.make(args.env)


def save_params(td3, timestep, outdir, args):
    print('saving model params of iter: ', timestep)

    q1_filename = 'q1_iter-{}'.format(timestep)
    q2_filename = 'q2_iter-{}'.format(timestep)
    pi_filename = 'pi_iter-{}'.format(timestep)

    td3._q1.to_cpu()
    td3._q2.to_cpu()
    td3._pi.to_cpu()
    serializers.save_model(os.path.join(
        outdir, q1_filename), td3._q1)
    serializers.save_model(os.path.join(
        outdir, q2_filename), td3._q2)
    serializers.save_model(os.path.join(
        outdir, pi_filename), td3._pi)

    if not args.gpu < 0:
        td3._q1.to_gpu()
        td3._q2.to_gpu()
        td3._pi.to_gpu()


def run_training_loop(env, td3, args):
    replay_buffer = []
    s_current = env.reset()

    episode_steps = 0
    previous_evaluation = 0

    outdir = files.prepare_output_dir(base_dir=args.outdir, args=args)

    result_file = os.path.join(outdir, 'result.txt')
    if not files.file_exists(result_file):
        with open(result_file, "w") as f:
            f.write('timestep\tmean\tmedian\n')

    for timestep in range(args.total_timesteps):
        if timestep < args.start_timesteps:
            s_current, a, r, s_next, done = td3.act_randomly(env, s_current)
        else:
            s_current, a, r, s_next, done = td3.act_with_policy(env, s_current)
        non_terminal = np.float32(0 if done else 1)

        replay_buffer.append((s_current, a, r, s_next, non_terminal))

        episode_steps += 1
        s_current = s_next

        if done:
            td3.train(replay_buffer, episode_steps, args.d,
                      args.clip_value, args.gamma, args.tau)

            if args.evaluation_interval < timestep - previous_evaluation:
                print('evaluating policy at timestep: ', timestep)
                rewards = td3.evaluate_policy(env)
                print('rewards: ', rewards)
                mean = np.mean(rewards)
                median = np.median(rewards)

                print('mean: {mean}, median: {median}'.format(
                    mean=mean, median=median))
                with open(result_file, "a") as f:
                    f.write('{timestep}\t{mean}\t{median}\n'.format(
                        timestep=timestep, mean=mean, median=median))
                
                save_params(td3, timestep, outdir, args)
                previous_evaluation = timestep
            
            episode_steps = 0
            s_current = env.reset()


def start_training(args):
    env = build_env(args)

    td3 = TD3(state_dim=env.observation_space.shape[0],
              action_num=env.action_space.shape[0],
              lr=args.learning_rate,
              batch_size=args.batch_size,
              device=args.gpu)

    run_training_loop(env, td3, args)


def main():
    parser = argparse.ArgumentParser()

    # output
    parser.add_argument('--outdir', type=str, default='results')

    # Environment
    parser.add_argument('--env', type=str, default='Walker2d-v2')

    # Gpu
    parser.add_argument('--gpu', type=int, default=-1)

    # Training parameters
    parser.add_argument('--total-timesteps', type=float, default=1000000)
    parser.add_argument('--learning-rate', type=float, default=1.0*1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--clip_value', type=float, default=0.5)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--start-timesteps', type=int, default=1000)
    parser.add_argument('--evaluation-interval', type=float, default=5000)

    args = parser.parse_args()

    start_training(args)


if __name__ == "__main__":
    main()
