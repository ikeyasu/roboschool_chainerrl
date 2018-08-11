import chainer
from chainer import functions as F
from chainer import links as L

from chainerrl.initializers import LeCunNormal


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
