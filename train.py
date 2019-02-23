"""
Copyright 2018 ikeyasu.com
LICENSE: MIT License
Inspired from https://github.com/uchibe/ai-bs-summer17/blob/master/roboschool/train_ddpg_gym.py
"""
import argparse

import sys
import os

import chainer
from chainer import functions as F
from chainer import optimizers
import gym
# noinspection PyUnresolvedReferences
import roboschool
from chainerrl.policies import FCDeterministicPolicy
from chainerrl.q_functions import FCSAQFunction

from gym import spaces
import gym.wrappers
import numpy as np

import chainerrl
from chainerrl.agents.ddpg import DDPG
from chainerrl.agents.ddpg import DDPGModel
from chainerrl import experiments
from chainerrl import explorers
from chainerrl import misc
from chainerrl import replay_buffer

from agents.ddpg_step import DDPGStep
from agents.trpo_step import TRPOStep
from env import urdf_env, mjcf_env, servo_env

xp = np
SM = 'SM_MODEL_DIR' in os.environ


def make_env(args):
    footlist = [] if args.foot_list is None else args.foot_list
    if args.urdf:
        env = urdf_env.make(model_urdf=os.path.abspath(args.urdf),
                            robot_name="base_link", footlist=[], action_dim=args.action_dim, obs_dim=args.obs_dim)
    elif args.mjcf:
        env = mjcf_env.make(model_xml=os.path.abspath(args.mjcf),
                            robot_name="torso", footlist=footlist, action_dim=args.action_dim, obs_dim=args.obs_dim)
    else:
        env = gym.make(args.env)

    if args.physical_with_sim:
        env = servo_env.make(env, (args.server_address, args.server_port))

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


def main(parser=argparse.ArgumentParser()):
    import logging
    logging.basicConfig(level=logging.WARN)

    parser.add_argument('--outdir', type=str, default='out')
    parser.add_argument('--save', type=str)
    parser.add_argument('--env', type=str, default='RoboschoolAnt-v1')
    parser.add_argument('--urdf', type=str, default=None)
    parser.add_argument('--mjcf', type=str, default=None, help="MuJoCo XML model")
    parser.add_argument('--foot-list', nargs='*', type=str, default=None, help="foot list")
    parser.add_argument('--physical-with-sim', action='store_true', help="Physical environment with simulator")
    parser.add_argument('--server-address', type=str, default="localhost", help="Server setting for physical environment")
    parser.add_argument('--server-port', type=int, default=8080, help="Server setting for physical environment")
    parser.add_argument('--action-dim', type=int, default=-1)
    parser.add_argument('--obs-dim', type=int, default=-1)
    parser.add_argument('--algorithm', type=str, default="DDPG", help="DDPG or TRPO")
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--final-exploration-steps',
                        type=int, default=10 ** 6)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--steps', type=int, default=10 ** 7)
    parser.add_argument('--skip-step', type=int, default=0, help="Skip steps for physical simulation")
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
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--monitor', action='store_true')
    parser.add_argument('--reward-scale-factor', type=float, default=1e-2)

    # Required for sagemaker
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'] if SM else None)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'] if SM else None)

    args = parser.parse_args()

    if args.urdf is not None:
        if args.action_dim <= 0 or args.obs_dim <= 0:
            raise Exception("--action-dim and --obs-dim are necessary when using --urdf")
    if args.mjcf is not None:
        if args.action_dim <= 0 or args.obs_dim <= 0:
            raise Exception("--action-dim and --obs-dim are necessary when using --mjcf")

    if args.gpu > -1:
        global xp
        # noinspection PyUnresolvedReferences
        import cupy
        xp = cupy

    if args.output_data_dir is not None:
        args.outdir = args.output_data_dir

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir, argv=sys.argv)
    print('Output files are saved in {}'.format(args.outdir))

    if args.seed is not None:
        misc.set_random_seed(args.seed)

    # because the argument may be concatinated (e.g sagemaker's hyperparameter)
    if ' ' in ''.join(args.foot_list):
        args.foot_list = ' '.join(args.foot_list).split()

    env = make_env(args)

    timestep_limit = env.spec.tags.get(
        'wrapper_config.TimeLimit.max_episode_steps')

    agent = make_agent(args, env)

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
        if args.save is not None:
            agent.save(args.save)


def make_agent(args, env):
    if args.algorithm.upper() == "DDPG":
        return make_agent_ddpg(args, env)
    elif args.algorithm.upper() == "TRPO":
        return make_agent_trpo(args, env)
    else:
        raise NotImplementedError()


def make_agent_trpo(args, env):
    obs_space = env.observation_space
    action_space = env.action_space
    # Normalize observations based on their empirical mean and variance
    obs_normalizer = chainerrl.links.EmpiricalNormalization(
        obs_space.low.size)
    policy = \
        chainerrl.policies.FCGaussianPolicyWithStateIndependentCovariance(
            obs_space.low.size,
            action_space.low.size,
            n_hidden_channels=64,
            n_hidden_layers=2,
            mean_wscale=0.01,
            nonlinearity=F.tanh,
            var_type='diagonal',
            var_func=lambda x: F.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        )
    # Use a value function to reduce variance
    vf = chainerrl.v_functions.FCVFunction(
        obs_space.low.size,
        n_hidden_channels=64,
        n_hidden_layers=2,
        last_wscale=0.01,
        nonlinearity=F.tanh,
    )
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        policy.to_gpu(args.gpu)
        vf.to_gpu(args.gpu)
        obs_normalizer.to_gpu(args.gpu)
    # TRPO's policy is optimized via CG and line search, so it doesn't require
    # a chainer.Optimizer. Only the value function needs it.
    vf_opt = chainer.optimizers.Adam()
    vf_opt.setup(vf)

    # Hyperparameters in http://arxiv.org/abs/1709.06560
    if args.skip_step == 0:
        agent = chainerrl.agents.TRPO(
            policy=policy,
            vf=vf,
            vf_optimizer=vf_opt,
            obs_normalizer=obs_normalizer,
            update_interval=5000,
            conjugate_gradient_max_iter=20,
            conjugate_gradient_damping=1e-1,
            gamma=0.995,
            lambd=0.97,
            vf_epochs=5,
            entropy_coef=0,
        )
    else:
        agent = TRPOStep(
            policy=policy,
            vf=vf,
            vf_optimizer=vf_opt,
            obs_normalizer=obs_normalizer,
            update_interval=5000,
            conjugate_gradient_max_iter=20,
            conjugate_gradient_damping=1e-1,
            gamma=0.995,
            lambd=0.97,
            vf_epochs=5,
            entropy_coef=0,
            skip_step=args.skip_step
        )

    return agent


def make_agent_ddpg(args, env):
    obs_size = np.asarray(env.observation_space.shape).prod()
    action_space = env.action_space

    action_size = np.asarray(action_space.shape).prod()
    q_func = FCSAQFunction(
        obs_size, action_size,
        n_hidden_channels=args.n_hidden_channels,
        n_hidden_layers=args.n_hidden_layers)
    pi = FCDeterministicPolicy(
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
    if args.skip_step == 0:
        agent = DDPG(model, opt_a, opt_c, rbuf, gamma=args.gamma,
                     explorer=explorer, replay_start_size=args.replay_start_size,
                     target_update_method=args.target_update_method,
                     target_update_interval=args.target_update_interval,
                     update_interval=args.update_interval,
                     soft_update_tau=args.soft_update_tau,
                     n_times_update=args.n_update_times,
                     phi=phi, gpu=args.gpu, minibatch_size=args.minibatch_size)
    else:
        agent = DDPGStep(model, opt_a, opt_c, rbuf, gamma=args.gamma,
                         explorer=explorer, replay_start_size=args.replay_start_size,
                         target_update_method=args.target_update_method,
                         target_update_interval=args.target_update_interval,
                         update_interval=args.update_interval,
                         soft_update_tau=args.soft_update_tau,
                         n_times_update=args.n_update_times,
                         phi=phi, gpu=args.gpu, minibatch_size=args.minibatch_size, skip_step=args.skip_step)
        if args.model_dir is not None:
            agent.save(args.model_dir)
    return agent


if __name__ == '__main__':
    main()
