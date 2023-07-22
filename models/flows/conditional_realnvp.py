import torch
import torch.nn as nn

from .modules import BatchNorm
from .modules import MLP


class Compose(nn.Module):
    """ compose flow layers """

    def __init__(self, layers):
        super(Compose, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, z, c, log_df_dz):
        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                z, log_df_dz = layer(z, log_df_dz)
            else:
                z, log_df_dz = layer(z, c, log_df_dz)
        return z, log_df_dz

    def backward(self, z, c, log_df_dz):
        for layer in reversed(self.layers):
            if isinstance(layer, BatchNorm):
                z, log_df_dz = layer.backward(z, log_df_dz)
            else:
                z, log_df_dz = layer.backward(z, c, log_df_dz)
        return z, log_df_dz


class ConditionalAffineCoupling(nn.Module):
    """
    affine coupling used in Real NVP
    """

    def __init__(self, in_dim, c_dim, odd=False):
        super(ConditionalAffineCoupling, self).__init__()

        self.register_parameter('s_log_scale', nn.Parameter(torch.randn(1) * 0.01))
        self.register_parameter('s_bias', nn.Parameter(torch.randn(1) * 0.01))

        in_chs = in_dim // 2
        self.out_chs = in_dim - in_chs
        self.net = MLP(in_chs + c_dim, self.out_chs * 2)
        from .squeeze import squeeze1d, unsqueeze1d
        self.squeeze = lambda z, odd=odd: squeeze1d(z, odd)
        self.unsqueeze = lambda z0, z1, odd=odd: unsqueeze1d(z0, z1, odd)

    def forward(self, z, c, log_df_dz):
        z0, z1 = self.squeeze(z)
        params = self.net(torch.cat([z1, c], dim=1))
        t = params[:, :self.out_chs]
        s = torch.tanh(params[:, self.out_chs:]) * self.s_log_scale + self.s_bias
        z0 = z0 * torch.exp(s) + t
        log_df_dz += torch.sum(s.view(z0.size(0), -1), dim=1)
        z = self.unsqueeze(z0, z1)
        return z, log_df_dz

    def backward(self, y, c, log_df_dz):
        y0, y1 = self.squeeze(y)
        params = self.net(torch.cat([y1, c], dim=1))
        t = params[:, :self.out_chs]
        s = torch.tanh(params[:, self.out_chs:]) * self.s_log_scale + self.s_bias

        y0 = torch.exp(-s) * (y0 - t)
        log_df_dz -= torch.sum(s.view(y0.size(0), -1), dim=1)
        y = self.unsqueeze(y0, y1)
        return y, log_df_dz


class ConditionalRealNVP(nn.Module):
    def __init__(self, in_dim, c_dim, n_layers):
        super(ConditionalRealNVP, self).__init__()

        self.n_layers = n_layers

        conditional_layer = ConditionalAffineCoupling
        layers = []
        # for density samples
        for i in range(self.n_layers):
            layers.append(BatchNorm([in_dim, ], affine=False))
            layers.append(conditional_layer(in_dim, c_dim, odd=i % 2 != 0))

        self.net = Compose(layers)

    def forward(self, z, c):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net(z, c, log_df_dz)

    def backward(self, z, c):
        log_df_dz = torch.zeros(z.size(0)).type_as(z).to(z.device)
        return self.net.backward(z, c, log_df_dz)
