import chainer
from chainer import functions as F, links as L


class DQN(chainer.ChainList):
    """DQN's head (Nature version)"""

    def __init__(self, n_input_channels=4, n_output_channels=512,
                 activation=F.relu, bias=0.1):
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_output_channels = n_output_channels

        layers = [
            L.Convolution2D(n_input_channels, 32, 8, stride=4,
                            initial_bias=bias),
            L.Convolution2D(32, 64, 4, stride=2, initial_bias=bias),
            L.Convolution2D(64, 64, 3, stride=1, initial_bias=bias),
            L.Linear(1024, n_output_channels, initial_bias=bias),
        ]

        super(DQN, self).__init__(*layers)

    def __call__(self, state):
        h = state
        for layer in self:
            h = self.activation(layer(h))
        return h