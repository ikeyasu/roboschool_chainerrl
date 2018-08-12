import chainer
import numpy as np
from chainer import functions as F
from chainer import links as L
from chainerrl import distribution, links
from chainerrl.functions.bound_by_tanh import bound_by_tanh
from chainerrl.links import MLP
from chainerrl.policy import Policy
from chainerrl.recurrent import RecurrentChainMixin

from dqn import DQN


class FCDeterministicPolicy(chainer.Chain, Policy, RecurrentChainMixin):
    """Fully-connected deterministic policy.

    Args:
        n_input_channels (int): Number of input channels.
        n_hidden_layers (int): Number of hidden layers.
        n_hidden_channels (int): Number of hidden channels.
        action_size (int): Size of actions.
        min_action (ndarray or None): Minimum action. Used only if bound_action
            is set to True.
        min_action (ndarray or None): Minimum action. Used only if bound_action
            is set to True.
        bound_action (bool): If set to True, actions are bounded to
            [min_action, max_action] by tanh.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported. It is not used if n_hidden_layers is zero.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_input_channels, n_hidden_layers,
                 n_hidden_channels, action_size,
                 min_action=None, max_action=None, bound_action=True,
                 nonlinearity=F.relu,
                 last_wscale=1.):

        if bound_action:
            def action_filter(x):
                return bound_by_tanh(
                    x, min_action, max_action)
        else:
            action_filter = None

        model = MLP(n_input_channels,
                    action_size,
                    (n_hidden_channels,) * n_hidden_layers,
                    nonlinearity=nonlinearity,
                    last_wscale=last_wscale,
                    )
        super().__init__(model=model)
        self.action_filter = action_filter

    def __call__(self, x):
        # Model
        h = self.model(x)
        # Action filter
        if self.action_filter is not None:
            h = self.action_filter(h)
        # Wrap by Distribution
        return distribution.ContinuousDeterministicDistribution(h)


class CNNDeterministicPolicy(chainer.Chain, Policy, RecurrentChainMixin):
    """CNN based deterministic policy.

    Args:
        n_input_channels (int): Number of input channels.
        n_hidden_layers (int): Number of hidden layers.
        n_hidden_channels (int): Number of hidden channels.
        action_size (int): Size of actions.
        min_action (ndarray or None): Minimum action. Used only if bound_action
            is set to True.
        min_action (ndarray or None): Minimum action. Used only if bound_action
            is set to True.
        bound_action (bool): If set to True, actions are bounded to
            [min_action, max_action] by tanh.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported. It is not used if n_hidden_layers is zero.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_input_channels, rgb_array_size: tuple, n_hidden_layers,
                 n_hidden_channels, action_size,
                 min_action=None, max_action=None, bound_action=True,
                 nonlinearity=F.relu,
                 last_wscale=1.):

        self.rgb_array_size = rgb_array_size
        rgb_ary_len = np.array(rgb_array_size).prod()
        n_input_channels -= rgb_ary_len - 1 # 1 is output of cnn

        if bound_action:
            def action_filter(x):
                return bound_by_tanh(
                    x, min_action, max_action)
        else:
            action_filter = None

        model = MLP(n_input_channels,
                    action_size,
                    (n_hidden_channels,) * n_hidden_layers,
                    nonlinearity=nonlinearity,
                    last_wscale=last_wscale,
                    )
        super().__init__(model=model)
        self.dqn_model = DQN(n_input_channels=3, n_output_channels=1)
        self.action_filter = action_filter

    def __call__(self, x):
        rgb_size = self.rgb_array_size
        rgb_ary_len = np.array(self.rgb_array_size).prod()
        rgb_images = x[0][0:rgb_ary_len].reshape(1, rgb_size[0], rgb_size[1], rgb_size[2])
        other_input = x[0][rgb_ary_len:]
        other_input = other_input.reshape(1, len(other_input))
        dqn_out = self.dqn_model(rgb_images)
        x = F.concat((other_input, dqn_out), axis=1)
        h = self.model(x)
        # Action filter
        if self.action_filter is not None:
            h = self.action_filter(h)
        # Wrap by Distribution
        return distribution.ContinuousDeterministicDistribution(h)
