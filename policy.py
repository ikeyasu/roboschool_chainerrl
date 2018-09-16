import chainer
import numpy as np
from chainer import functions as F
from chainer.links import VGG16Layers
from chainerrl import distribution
from chainerrl.functions.bound_by_tanh import bound_by_tanh
from chainerrl.links import MLP
from chainerrl.policy import Policy
from chainerrl.recurrent import RecurrentChainMixin


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
                 nonlinearity=F.relu, last_wscale=1., gpu=-1):

        self.rgb_array_size = rgb_array_size
        rgb_ary_len = np.array(rgb_array_size).prod()
        self.cnn_model = VGG16Layers()
        if gpu > -1:
            self.cnn_model.to_gpu(gpu)
        n_input_channels -= rgb_ary_len - self.cnn_model.fc7.W.shape[0]

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
        rgb_size = self.rgb_array_size
        rgb_ary_len = np.array(self.rgb_array_size).prod()
        batchsize = x.shape[0]
        rgb_images = x[:, 0:rgb_ary_len].reshape(batchsize, rgb_size[0], rgb_size[1], rgb_size[2])
        cnn_out = self.cnn_model(rgb_images, layers=["fc7"])["fc7"].data
        other_input = self.xp.asarray(x[:, rgb_ary_len:])
        other_input = other_input.reshape(batchsize, other_input.shape[1])
        # TODO: need to evaluate features
        x = F.concat((other_input, cnn_out), axis=1)
        h = self.model(x)
        # Action filter
        if self.action_filter is not None:
            h = self.action_filter(h)
        # Wrap by Distribution
        return distribution.ContinuousDeterministicDistribution(h)
