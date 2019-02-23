"""
Copyright 2019 ikeyasu.com
LICENSE: MIT License
"""
from logging import getLogger

import chainer
from chainerrl.agents import TRPO
from chainerrl.misc import batch_states


class TRPOStep(TRPO):
    def __init__(self, policy, vf, vf_optimizer, obs_normalizer=None, gamma=0.99, lambd=0.95, phi=lambda x: x,
                 entropy_coef=0.01, update_interval=2048, max_kl=0.01, vf_epochs=3, vf_batch_size=64,
                 standardize_advantages=True, line_search_max_backtrack=10, conjugate_gradient_max_iter=10,
                 conjugate_gradient_damping=1e-2, act_deterministically=False, value_stats_window=1000,
                 entropy_stats_window=1000, kl_stats_window=100, policy_step_size_stats_window=100,
                 logger=getLogger(__name__), skip_step: int = 10):
        TRPO.__init__(self, policy, vf, vf_optimizer, obs_normalizer, gamma, lambd, phi,
                      entropy_coef, update_interval, max_kl, vf_epochs, vf_batch_size,
                      standardize_advantages, line_search_max_backtrack, conjugate_gradient_max_iter,
                      conjugate_gradient_damping, act_deterministically, value_stats_window,
                      entropy_stats_window, kl_stats_window, policy_step_size_stats_window,
                      logger)
        self.t = 0
        self.skip_step = skip_step

    def act_and_train(self, state, reward):

        xp = self.xp
        b_state = batch_states([state], xp, self.phi)

        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)

        if self.skip_step > 0 and self.last_action is not None and self.t % self.skip_step != 0:
            action = self.last_action
        else:
            # action_distrib will be recomputed when computing gradients
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                action_distrib = self.policy(b_state)
                action = chainer.cuda.to_cpu(action_distrib.sample().array)[0]
                self.entropy_record.append(float(action_distrib.entropy.array))
            self.logger.debug('action_distrib: %s', action_distrib)

        self.logger.debug('action: %s', action)

        if self.last_state is not None:
            self.last_episode.append({
                'state': self.last_state,
                'action': self.last_action,
                'reward': reward,
                'next_state': state,
                'nonterminal': 1.0,
            })
        self.last_state = state
        self.last_action = action
        self.t += 1

        self._update_if_dataset_is_ready()

        return action

    def act(self, state):
        xp = self.xp
        b_state = batch_states([state], xp, self.phi)
        if self.obs_normalizer:
            b_state = self.obs_normalizer(b_state, update=False)
        if self.skip_step > 0 and self.last_action is not None and self.t % self.skip_step != 0:
            return self.last_action
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            action_distrib = self.policy(b_state)
            if self.act_deterministically:
                action = chainer.cuda.to_cpu(
                    action_distrib.most_probable.array)[0]
            else:
                action = chainer.cuda.to_cpu(action_distrib.sample().array)[0]
        self.last_action = action
        return action
