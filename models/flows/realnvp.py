import torch
import torch.nn as nn

from .modules import Compose, BatchNorm
from .coupling import AffineCoupling, AdditiveCoupling


class RealNVP(nn.Module):
    def __init__(self, in_dim, n_layers, coupling='affine', use_bn=True):
        super(RealNVP, self).__init__()

        self.in_dim = in_dim
        self.n_layers = n_layers
        if coupling == 'affine':
            coupling_layer = AffineCoupling
        elif coupling == 'additive':
            coupling_layer = AdditiveCoupling
        else:
            raise NotImplemented

        layers = []
        # for density samples
        for i in range(self.n_layers):
            if use_bn:
                layers.append(BatchNorm([in_dim, ], affine=False))
            layers.append(coupling_layer([in_dim, ], odd=i % 2 != 0))

        self.net = Compose(layers)

    def forward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net(z, log_df_dz)

    def backward(self, z):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net.backward(z, log_df_dz)
