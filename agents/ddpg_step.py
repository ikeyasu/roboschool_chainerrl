"""
Copyright 2018 ikeyasu.com
LICENSE: MIT License
"""
from chainerrl.agents import DDPG


class DDPGStep(DDPG):
    def __init__(self, model, actor_optimizer, critic_optimizer, replay_buffer,
                 gamma, explorer,
                 gpu=None, replay_start_size=50000,
                 minibatch_size=32, update_interval=1,
                 target_update_interval=10000,
                 phi= lambda x: x,
                 target_update_method='hard',
                 soft_update_tau=1e-2,
                 n_times_update=1, average_q_decay=0.999,
                 average_loss_decay=0.99,
                 episodic_update=False,
                 episodic_update_len=None,
                 skip_step: int = 25):
        DDPG.__init__(self, model, actor_optimizer, critic_optimizer, replay_buffer,
                      gamma, explorer, gpu, replay_start_size, minibatch_size,
                      update_interval, target_update_interval,
                      phi, target_update_method, soft_update_tau, n_times_update,
                      average_q_decay, average_loss_decay,
                      episodic_update, episodic_update_len)
        self.skip_step = skip_step

    def act_and_train(self, obs, reward):

        self.logger.debug('t:%s r:%s', self.t, reward)

        self.t += 1
        if self.skip_step > 0 and self.last_action is not None and (self.t - 1) % self.skip_step != 0:
            return self.last_action

        greedy_action = self.act(obs)

        action = self.explorer.select_action(self.t, lambda: greedy_action)

        # Update the target network
        if self.t % self.target_update_interval == 0:
            self.sync_target_network()

        if self.last_state is not None:
            assert self.last_action is not None
            # Add a transition to the replay buffer
            self.replay_buffer.append(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=obs,
                next_action=action,
                is_state_terminal=False)

        self.last_state = obs
        self.last_action = action

        self.replay_updater.update_if_necessary(self.t)

        return action
