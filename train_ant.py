"""
Copyright 2018 ikeyasu.com
LICENSE: MIT License
Inspired from https://github.com/uchibe/ai-bs-summer17/blob/master/roboschool/train_ddpg_gym.py
"""
import argparse

import cupy
import sys
import os

import chainer
from chainer import optimizers
import gym
# noinspection PyUnresolvedReferences
import roboschool

from gym import spaces
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl.agents.ddpg import DDPG
from chainerrl.agents.ddpg import DDPGModel
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import policy
from chainerrl import replay_buffer

import qfunc
import policy
from env import GymFPS

CAM_SIZE = (3, 64, 64)

xp = np


def make_env(args):
    if args.use_fps_image:
        env = GymFPS(gym.make(args.env), fps_window=args.fps_window, cam_size=CAM_SIZE, random_flag_pos=args.random_flag_pos)
    else:
        env = gym.make(args.env)

    def clip_action_filter(a):
        return np.clip(a, env.action_space.low, env.action_space.high)

    def reward_filter(r):
        return r * args.reward_scale_factor

    if args.monitor:
        env = gym.wrappers.Monitor(env, args.outdir)
    if isinstance(env.action_space, spaces.Box):
        misc.env_modifiers.make_action_filtered(env, clip_action_filter)
    misc.env_modifiers.make_reward_filtered(env, reward_filter)
    if args.render:
        misc.env_modifiers.make_rendered(env)

    def __exit__(_self, *_args):
        pass

    env.__exit__ = __exit__
    return env


def main():
    import logging
    logging.basicConfig(level=logging.WARN)

    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='out')
    parser.add_argument('--env', type=str, default='RoboschoolAnt-v1')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--final-exploration-steps',
                        type=int, default=10 ** 6)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.add_argument('--n-hidden-channels', type=int, default=300)
    parser.add_argument('--n-hidden-layers', type=int, default=3)
    parser.add_argument('--replay-start-size', type=int, default=5000)
    parser.add_argument('--n-update-times', type=int, default=1)
    parser.add_argument('--target-update-interval',
                        type=int, default=1)
    parser.add_argument('--target-update-method',
                        type=str, default='soft', choices=['hard', 'soft'])
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--update-interval', type=int, default=4)
    parser.add_argument('--eval-n-runs', type=int, default=100)
    parser.add_argument('--eval-interval', type=int, default=10 ** 5)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--minibatch-size', type=int, default=200)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--fps-window', action='store_true')
    parser.add_argument('--random-flag-pos', action='store_true')
    parser.add_argument('--use-fps-image', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
    parser.add_argument('--dqn-out-len', type=int, default=512)
    args = parser.parse_args()

    if args.gpu > -1:
        global xp
        xp = cupy

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    if args.render and args.fps_window:
        raise Exception("Cannot specify --render and --fps-window at same time")
    if args.seed is not None:
        misc.set_random_seed(args.seed)

    env = make_env(args)

    timestep_limit = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')
    # TODO: replace env.observation_space at GymFPS
    obs_size = np.asarray(env.observation_space.shape).prod()
    if args.use_fps_image:
        obs_size += CAM_SIZE[0] * CAM_SIZE[1] * CAM_SIZE[2]
    action_space = env.action_space

    action_size = np.asarray(action_space.shape).prod()
    if args.use_fps_image:
        q_func = qfunc.CNNSAQFunction(
            obs_size, CAM_SIZE, action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            dqn_out_len=args.dqn_out_len, gpu=args.gpu)
        pi = policy.CNNDeterministicPolicy(
            obs_size, CAM_SIZE, action_size=action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            min_action=action_space.low, max_action=action_space.high,
            bound_action=True, dqn_out_len=args.dqn_out_len, gpu=args.gpu)
    else:
        q_func = qfunc.FCSAQFunction(
            obs_size, action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers)
        pi = policy.FCDeterministicPolicy(
            obs_size, action_size=action_size,
            n_hidden_channels=args.n_hidden_channels,
            n_hidden_layers=args.n_hidden_layers,
            min_action=action_space.low, max_action=action_space.high,
            bound_action=True)
    if args.gpu > -1:
        q_func.to_gpu(args.gpu)
        pi.to_gpu(args.gpu)
    else:
        q_func.to_cpu()
        pi.to_cpu()

    # draw computation graph
    fake_obs = xp.asarray(np.zeros(obs_size, dtype=np.float32)[None])
    fake_action = xp.asarray(np.zeros(action_size, dtype=np.float32)[None])
    with chainerrl.recurrent.state_reset(q_func):  # The state of the model is reset again after drawing the graph
        chainerrl.misc.draw_computational_graph([q_func(fake_obs, fake_action)],
                                                os.path.join(args.outdir, 'model_q_func'))
    with chainerrl.recurrent.state_reset(pi):  # The state of the model is reset again after drawing the graph
        chainerrl.misc.draw_computational_graph([pi(fake_obs)], os.path.join(args.outdir, 'model_policy'))

    model = DDPGModel(q_func=q_func, policy=pi)
    opt_a = optimizers.Adam(alpha=args.actor_lr)
    opt_c = optimizers.Adam(alpha=args.critic_lr)
    opt_a.setup(model['policy'])
    opt_c.setup(model['q_function'])
    opt_a.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_a')
    opt_c.add_hook(chainer.optimizer.GradientClipping(1.0), 'hook_c')

    rbuf = replay_buffer.ReplayBuffer(5 * 10 ** 5)

    def phi(obs):
        return obs.astype(np.float32)

    # def random_action():
    #    a = action_space.sample()
    #    if isinstance(a, np.ndarray):
    #        a = a.astype(np.float32)
    #    return a

    ou_sigma = (action_space.high - action_space.low) * 0.2
    explorer = explorers.AdditiveOU(sigma=ou_sigma)
    agent = DDPG(model, opt_a, opt_c, rbuf, gamma=args.gamma,
                 explorer=explorer, replay_start_size=args.replay_start_size,
                 target_update_method=args.target_update_method,
                 target_update_interval=args.target_update_interval,
                 update_interval=args.update_interval,
                 soft_update_tau=args.soft_update_tau,
                 n_times_update=args.n_update_times,
                 phi=phi, gpu=args.gpu, minibatch_size=args.minibatch_size)

    if len(args.load) > 0:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=env,
            agent=agent,
            n_runs=args.eval_n_runs,
            max_episode_len=timestep_limit)
        print('n_runs: {} mean: {} median: {} stdev {}'.format(
            args.eval_n_runs, eval_stats['mean'], eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_runs=args.eval_n_runs, eval_interval=args.eval_interval,
            outdir=args.outdir,
            max_episode_len=timestep_limit)


if __name__ == '__main__':
    main()
