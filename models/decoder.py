from torch import nn
import math
import torch.nn.functional as F
import torch
import itertools
import numpy as np


class StyleGANDecoder(nn.Module):

    def __init__(self,
                 stylegan_weights_path,
                 start_from_latent_avg=True,
                 output_size=128,
                 downsampling=True,
                 ):
        super(StyleGANDecoder, self).__init__()

        self.output_size = output_size
        # compute number of style inputs based on the output resolution
        self.start_from_latent_avg = start_from_latent_avg
        self.downsampling = downsampling

        self.load_weights(stylegan_weights_path)
        self.style_dim = self.decoder.w_dim
        self.n_styles = self.decoder.num_ws

    @torch.no_grad()
    def load_weights(self, stylegan_weights_path):
        print(f'Loading decoder weights from pretrained {stylegan_weights_path}!')

        from .stylegan2 import dnnlib, legacy
        with dnnlib.util.open_url(stylegan_weights_path) as fp:
            ckpt = legacy.load_network_pkl(fp)
        self.decoder = ckpt['G_ema']  # type: ignore

        if 'latent_avg' not in ckpt or 'principal_components' not in ckpt:
            print('compute avg...')
            z = torch.randn(100000, self.decoder.w_dim)
            import copy
            ws = copy.deepcopy(self.decoder.mapping).cuda()(z.cuda(), None)[:, 0]
            latent_avg = ws.mean(0, keepdim=True)
            principal_components = self.compute_pca(ws)

            ckpt['latent_avg'] = latent_avg.cpu()
            ckpt['principal_components'] = principal_components.cpu()
            with open(stylegan_weights_path, 'wb') as f:
                import pickle
                pickle.dump(ckpt, f)
        self.register_buffer('latent_avg', ckpt['latent_avg'].cpu())
        self.register_buffer('principal_components', ckpt['principal_components'].cpu())

    def sample_codes(self, bs):
        device = next(self.parameters()).device
        return self.decoder.mapping(torch.randn(bs, 512).to(device), None)[:, 0]

    @torch.no_grad()
    def compute_pca(self, X):
        X = X.cpu().numpy()
        from sklearn.decomposition import PCA
        import itertools
        n_components = 512
        transformer = PCA(n_components, svd_solver='full')
        transformer.fit(X)
        total_var = X.var(axis=0).sum()
        stdev = np.dot(transformer.components_, X.T).std(axis=1)
        idx = np.argsort(stdev)[::-1]
        stdev = stdev[idx]
        transformer.components_[:] = transformer.components_[idx]

        # Check orthogonality
        dotps = [np.dot(*transformer.components_[[i, j]])
                 for (i, j) in itertools.combinations(range(n_components), 2)]
        if not np.allclose(dotps, 0, atol=1e-4):
            print('IPCA components not orghogonal, max dot', np.abs(dotps).max())

        transformer.mean_ = X.mean(axis=0, keepdims=True)

        principal_components = torch.from_numpy(transformer.components_)
        return principal_components

    def forward(self, codes):
        if codes.ndim == 2:
            codes = codes.unsqueeze(1)
        if codes.size(1) == 1:
            codes = codes.repeat(1, self.n_styles, 1)
        # normalize with respect to the center of an average face
        if self.start_from_latent_avg:
            codes = codes + self.latent_avg

        images = self.decoder.synthesis(codes, noise_mode='const')
        if images.size(2) != self.output_size and self.downsampling:
            # images = F.interpolate(images, size=self.output_size, mode='bilinear', align_corners=True)
            from .ops import resize
            images = resize(images, self.output_size)

        return images
