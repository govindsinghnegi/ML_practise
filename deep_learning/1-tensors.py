import torch


def activation(x):
    """ Sigmoid activation function

        Arguments
        ---------
        x: torch.Tensor
    """
    return 1 / (1 + torch.exp(-x))

torch.manual_seed(7)
features = torch.randn((1, 5))
weights = torch.randn_like(features)
bias = torch.randn((1, 1))
y = activation(torch.mm(features, weights.view(5, 1)) + bias)
print('activation of y = {}'.format(y))

########### with hidden layers #######################

torch.manual_seed(7)
features = torch.randn((1, 3))
n_input = features.shape[1]
n_hidden = 2
n_output = 1
W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))
h = activation(torch.mm(features, W1) + B1)
output = activation(torch.mm(h, W2) + B2)
print('output of this NN = {}'.format(output))

######## numpy<->tensor
import numpy as np
a = np.random.rand(2,2)
print('numpy a = \n {}'.format(a))
b = torch.from_numpy(a)
print('tensor b from a = \n{}'.format(b))
print('numpy from tensor b = \n{}'.format(b.numpy()))