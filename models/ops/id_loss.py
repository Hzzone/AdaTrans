import torch
from torch import nn
import torch.nn.functional as F


class IDLoss(nn.Module):
    def __init__(self, crop=False, backbone='r50'):
        super(IDLoss, self).__init__()
        from .insightface import iresnet50, iresnet34
        if backbone == 'r50':
            self.facenet = iresnet50(pretrained=True)
        else:
            self.facenet = iresnet34(pretrained=True)
        self.facenet.eval()
        self.crop = crop
        self.embeddings = None

    @torch.no_grad()
    def extract_dataset(self, loader):
        embeddings = []
        for inputs in loader:
            images = inputs[0].cuda()
            embeddings.append(F.normalize(self.extract_features(images), dim=1))
        self.embeddings = torch.cat(embeddings, dim=0)

    def extract_features(self, x):
        def resize(x):
            if x.size(-1) == 112:
                return x
            # return F.interpolate(x, size=112, mode='bilinear', align_corners=True)
            from . import resize
            return resize(x, 112)

        if self.crop:
            # x = x[:, :, 35:223, 32:220]
            w, h = x.size(-2), x.size(-1)
            assert w == h
            img_size = w
            scale = lambda x: int(x * img_size / 256)
            h, x1, x2 = scale(188), scale(35), scale(32)
            x = x[:, :, x1:x1 + h, x2:x2 + h]
        x = resize(x)
        # with torch.no_grad():
        #     from torchvision.utils import make_grid
        #     from torchvision.transforms.functional import to_pil_image
        #     to_pil_image(make_grid(x * 0.5 + 0.5)).save('tmp.png')
        #     exit(0)
        return self.facenet(x)

    def forward(self, input, recon):
        input = input * 0.5 + 0.5
        recon = recon * 0.5 + 0.5
        with torch.no_grad():
            e1 = F.normalize(self.extract_features(input), dim=1)
        e2 = F.normalize(self.extract_features(recon), dim=1)
        return - (e1 * e2).sum(dim=1).mean()
