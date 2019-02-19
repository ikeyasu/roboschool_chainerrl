"""
Copyright 2018 ikeyasu.com
LICENSE: MIT License
"""
from chainerrl.agents import DDPG


class DDPGStep(DDPG):
    def __init__(self,
                 skip_step: int = 0, **kwargs):
        DDPG.__init__(self, **kwargs)
        self.skip_step = skip_step

    def act_and_train(self, obs, reward):

        self.logger.debug('t:%s r:%s', self.t, reward)

        greedy_action = self.act(obs)

        if self.skip_step > 0 and self.last_action is not None and self.t % self.skip_step != 0:
            action = self.last_action
        else:
            action = self.explorer.select_action(self.t, lambda: greedy_action)

        self.t += 1

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
