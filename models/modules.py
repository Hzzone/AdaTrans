import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from .flows.realnvp import RealNVP


def mlp(in_c, hid_c, out_c, n_layer=0):
    layers = [nn.Linear(in_c, hid_c),
              nn.BatchNorm1d(hid_c),
              nn.ReLU(True)]
    for _ in range(n_layer):
        layers += [
            nn.Linear(hid_c, hid_c),
            nn.BatchNorm1d(hid_c),
            nn.ReLU(True)
        ]
    layers.append(nn.Linear(hid_c, out_c))
    return nn.Sequential(*layers)


class Classifier(nn.Module):
    def __init__(self, backbone='r34'):
        super(Classifier, self).__init__()
        import torchvision
        if backbone == 'r34':
            backbone = torchvision.models.resnet34(weights='ResNet34_Weights.IMAGENET1K_V1')
            in_dim = 512
        elif backbone == 'r50':
            backbone = torchvision.models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
            in_dim = 2048
        else:
            raise NotImplementedError
        self.extractor = copy.deepcopy(backbone)
        self.extractor.fc = nn.Identity()

        self.attr_heads = nn.ModuleList([mlp(in_dim, in_dim, 1, 2) for _ in range(40)])

        self.age_heads = copy.deepcopy(backbone)
        self.age_heads.fc = nn.Linear(in_dim, 6)

    def forward(self, x):
        outs = []

        features = self.extractor(x)
        for i, head in enumerate(self.attr_heads):
            outs.append(head(features))

        outs.append(self.age_heads(x))
        outs = torch.cat(outs, dim=1)

        preds = (torch.sigmoid(outs) > 0.5).float()
        preds = torch.cat([preds[:, :40], preds[:, 40:].sum(dim=1).unsqueeze(1)], dim=1)
        return outs, preds

    def forward_attr(self, x):
        features = self.extractor(x)
        outs = []
        for i, head in enumerate(self.attr_heads):
            outs.append(head(features))

        outs = torch.cat(outs, dim=1)
        preds = (torch.sigmoid(outs) > 0.5).float()
        return outs, preds

    def forward_age(self, x):
        outs = self.age_heads(x)
        preds = (torch.sigmoid(outs) > 0.5).float().sum(dim=1)
        return outs, preds


class FLOW(nn.Module):
    def __init__(self, style_dim=2, n_layer=10, n_styles=18):
        super(FLOW, self).__init__()
        self.flow = RealNVP(style_dim, n_layer)

    def forward(self, inputs):
        if inputs.ndim == 3:
            bs, n_styles, dim = inputs.size()
            latents = inputs.view(bs * n_styles, -1)
        elif inputs.ndim == 2:
            bs, dim = inputs.size()
            latents = inputs
        else:
            raise NotImplementedError
        z, log_det_jacobian = self.flow(latents)
        logz_sum = D.Normal(0, 1).log_prob(z).sum(dim=1)
        loss = - torch.mean(logz_sum + log_det_jacobian) / dim
        logz = logz_sum / dim
        return loss, logz, log_det_jacobian, z

    def backward(self, inputs):
        bs, n_styles, dim = inputs.size()
        inputs = inputs.view(bs * n_styles, -1)
        return self.flow.backward(inputs)[0].view(-1, n_styles, dim)


class Affine(nn.Module):
    def __init__(self, dim_out, dim_c):
        super(Affine, self).__init__()
        self.scale = mlp(dim_c, dim_out, dim_out)
        self.bias = mlp(dim_c, dim_out, dim_out)

    def forward(self, x, c):
        scale = self.scale(c).exp()
        bias = self.bias(c)
        return x * scale + bias


class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_c):
        super(ConcatSquashLinear, self).__init__()
        self.layer = mlp(dim_in, dim_in, dim_out, 0)
        self.affine = Affine(dim_out, dim_c)

    def forward(self, x, c):
        x = self.layer(x)
        x = self.affine(x, c)
        return x


class Func(nn.Module):

    def __init__(self, layer, style_dim, hid_c, c_dim, n_styles, n_layer=0):
        super(Func, self).__init__()
        layers = [layer(hid_c, hid_c, c_dim), ]
        for _ in range(n_layer - 1):
            layers.append(layer(hid_c, hid_c, c_dim))
        self.main = nn.ModuleList(layers)
        self.input_layer = mlp(style_dim * n_styles, hid_c, hid_c)
        self.output_layer = mlp(hid_c, hid_c, style_dim)
        self.step_layer = mlp(hid_c, hid_c, 1)
        self.rnn_layers = 2
        self.rnn = nn.LSTM(hid_c, hid_c, self.rnn_layers)
        self.n_styles = n_styles
        self.style_dim = style_dim

    def forward(self, trajectory, c):
        trajectory = torch.stack(trajectory).flatten(-2)
        y = self.input_layer(trajectory.view(-1, trajectory.size(-1))).view(trajectory.size(0), trajectory.size(1), -1)
        y = self.rnn(y, (torch.zeros(self.rnn_layers, y.size(1), y.size(2)).to(y),
                         torch.zeros(2, y.size(1), y.size(2)).to(y)))[0]
        y0 = y[-1]
        for i in range(len(self.main)):
            y0 = self.main[i](y0, c)

        step_size = torch.sigmoid(self.step_layer(y0))

        # dy = F.normalize(self.output_layer(y0).view(-1, self.n_styles, self.style_dim), dim=(-1, -2)) * np.sqrt(self.n_styles)
        dy = F.normalize(self.output_layer(y0), dim=-1).unsqueeze(1)

        return dy, step_size


class AdaTrans(nn.Module):
    def __init__(self,
                 style_dim=512,
                 hid_dim=512,
                 c_dim=4,
                 n_styles=18,
                 max_steps=20,
                 n_layers=10,
                 ):
        super(AdaTrans, self).__init__()
        self.n_layers = n_layers
        self.style_dim = style_dim
        self.func = Func(ConcatSquashLinear, style_dim, hid_dim, c_dim, n_styles, self.n_layers)
        self.c_dim = c_dim
        self.max_steps = max_steps

    def forward(self, x, target):
        target = target.float()

        y0 = x
        step_sizes = []
        trajectory = [y0]
        for t in range(self.max_steps):
            dy, step_size = self.func(trajectory, target)
            y1 = y0 + dy * step_size[:, :, None]
            y0 = y1
            step_sizes.append(step_size)
            trajectory.append(y0)
        step_sizes = torch.cat(step_sizes, dim=1)
        trajectory = torch.stack(trajectory)
        return trajectory, step_sizes
