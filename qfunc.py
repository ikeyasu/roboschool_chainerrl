import chainer
import numpy as np
from chainer import functions as F
from chainer import links as L

from chainerrl.initializers import LeCunNormal

from dqn import DQN


class FCSAQFunction(chainer.Chain):
    """Fully-connected (s,a)-input Q-function.

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported. It is not used if n_hidden_layers is zero.
        last_wscale (float): Scale of weight initialization of the last layer.
    """

    def __init__(self, n_dim_obs, n_dim_action, n_hidden_channels,
                 n_hidden_layers, nonlinearity=F.relu,
                 last_wscale=1.):
        in_size = n_dim_obs + n_dim_action
        out_size = 1
        self.hidden_sizes = [n_hidden_channels] * n_hidden_layers
        self.nonlinearity = nonlinearity

        super().__init__()
        with self.init_scope():
            if self.hidden_sizes:
                hidden_layers = [L.Linear(in_size, self.hidden_sizes[0])]
                for hin, hout in zip(self.hidden_sizes, self.hidden_sizes[1:]):
                    hidden_layers.append(L.Linear(hin, hout))
                self.hidden_layers = chainer.ChainList(*hidden_layers)
                self.output = L.Linear(self.hidden_sizes[-1], out_size,
                                       initialW=LeCunNormal(last_wscale))
            else:
                self.output = L.Linear(in_size, out_size,
                                       initialW=LeCunNormal(last_wscale))

    def __call__(self, state, action):
        h = F.concat((state, action), axis=1)
        if self.hidden_sizes:
            for l in self.hidden_layers:
                h = self.nonlinearity(l(h))
        return self.output(h)


class CNNSAQFunction(chainer.Chain):
    """CNN based(s,a)-input Q-function.

    Args:
        n_dim_obs (int): Number of dimensions of observation space.
        rgb_array_size (tuple): size of rgb image. (W, H)
        n_dim_action (int): Number of dimensions of action space.
        n_hidden_channels (int): Number of hidden channels.
        n_hidden_layers (int): Number of hidden layers.
        nonlinearity (callable): Nonlinearity between layers. It must accept a
            Variable as an argument and return a Variable with the same shape.
            Nonlinearities with learnable parameters such as PReLU are not
            supported. It is not used if n_hidden_layers is zero.
        last_wscale (float): Scale of weight initialization of the last layer.
    """
    # TODO: GPU support

    def __init__(self, n_dim_obs, rgb_array_size: tuple, n_dim_action, n_hidden_channels,
                 n_hidden_layers, nonlinearity=F.relu,
                 last_wscale=1.):
        self.rgb_array_size = rgb_array_size
        self.hidden_sizes = [n_hidden_channels] * n_hidden_layers
        self.nonlinearity = nonlinearity

        rgb_ary_len = np.array(rgb_array_size).prod()
        in_size = n_dim_obs - rgb_ary_len + 1 + n_dim_action # 1 is out of rgb

        super().__init__()
        with self.init_scope():
            self.dqn_model = DQN(n_input_channels=3, n_output_channels=1)
            hidden_layers = [L.Linear(in_size, self.hidden_sizes[0])]
            for hin, hout in zip(self.hidden_sizes, self.hidden_sizes[1:]):
                hidden_layers.append(L.Linear(hin, hout))
            self.hidden_layers = chainer.ChainList(*hidden_layers)
            self.output = L.Linear(self.hidden_sizes[-1], out_size=1,
                                   initialW=LeCunNormal(last_wscale))

    def __call__(self, state, action):
        rgb_size = self.rgb_array_size
        rgb_ary_len = np.array(self.rgb_array_size).prod()
        rgb_images = state[0][0:rgb_ary_len].reshape(1, rgb_size[0], rgb_size[1], rgb_size[2])
        dqn_out = self.dqn_model(rgb_images)
        other_state = state[0][rgb_ary_len:]
        other_state = other_state.reshape(1, len(other_state))
        h = F.concat((other_state, dqn_out, action), axis=1)
        if self.hidden_sizes:
            for l in self.hidden_layers:
                h = self.nonlinearity(l(h))
        return self.output(h)
