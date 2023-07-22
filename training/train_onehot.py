import random
import torch.nn.functional as F
import sys
import torch
import argparse

sys.path.insert(0, '/home/zzhuang/AdaTrans')
sys.path.insert(0, '/home/zzhuang/AdaTrans/models/stylegan2')

parser = argparse.ArgumentParser('Default arguments for training of different methods')
parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
parser.add_argument('--run_name', type=str, default='AdaTrans')
parser.add_argument('--wandb', help='wandb', action='store_true')
parser.add_argument('--project_name', help='wandb project_name', type=str, default='FaceEditing')
parser.add_argument('--entity', help='wandb project_name', type=str, default='zzhuang')
parser.add_argument('--changes', nargs='+', type=int, default=[15, ])
parser.add_argument('--keeps', nargs='+', type=int, default=[20, -1])
parser.add_argument('--recon_loss_weight', type=float, default=1.0)
parser.add_argument('--nll_loss_weight', type=float, default=1.0)
parser.add_argument('--cls_loss_weight', type=float, default=1.0)
parser.add_argument('--id_loss_weight', type=float, default=0.1)
parser.add_argument('--lpips_loss_weight', type=float, default=0.1)
parser.add_argument('--pix_loss_weight', type=float, default=0.0)
parser.add_argument('--max_steps', type=int, default=5)
opts = parser.parse_args()
print(opts)

torch.autograd.set_grad_enabled(False)
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

import os.path as osp
from models.ops import resize, convert_to_cuda

torch.set_printoptions(precision=3)
torch.set_printoptions(sci_mode=False)
import torchvision.transforms as transforms
from models.ops import dataset_with_indices, load_network, age2group
from models.ops.loggerx import LoggerX
from models.ops.grad_scaler import NativeScalerWithGradNormCount
from models.dataset import dataset_dict

from models.decoder import StyleGANDecoder
from models.e4e.psp_encoders import Encoder4Editing
from models.modules import Classifier
from models.ops.lpips import LPIPS
from models.ops.id_loss import IDLoss
from models.modules import FLOW
from models.modules import AdaTrans

data_root = '/home/zzhuang/DATASET/FFHQ/images256x256'
stylegan2_checkpoint = '/home/zzhuang/AdaTrans/data/ffhq.pkl'
e4e_checkpoint = '/home/zzhuang/AdaTrans/data/e4e_ffhq_encode.pt'
classifier_checkpoint = '/home/zzhuang/AdaTrans/data/r34_a40_age_256_classifier.pth'
latents_checkpoint = '/home/zzhuang/AdaTrans/data/ffhq_train_latents.pth'
preds_checkpoint = '/home/zzhuang/AdaTrans/data/ffhq_train_preds.pth'
realnvp_checkpoint = '/home/zzhuang/AdaTrans/data/realnvp.pth'

output_size = 256
G = StyleGANDecoder(
    stylegan2_checkpoint,
    start_from_latent_avg=False,
    output_size=output_size,
)
G = G.cuda().eval()

encoder = Encoder4Editing(50, 'ir_se', stylegan_size=1024, checkpoint_path=e4e_checkpoint).cuda().eval()
image_classifier = Classifier().cuda()
image_classifier.load_state_dict(load_network(torch.load(classifier_checkpoint, map_location='cpu')))
image_classifier = image_classifier.eval().cuda()
LIPIPS_LOSS = LPIPS().cuda()
ID_LOSS = IDLoss(crop=True, backbone='r34').cuda()

all_latents = torch.load(latents_checkpoint, map_location='cpu').cuda() + G.latent_avg
all_preds = torch.load(preds_checkpoint, map_location='cpu').cuda()
realnvp = FLOW(style_dim=G.style_dim, n_styles=G.n_styles, n_layer=10).cuda().eval()
realnvp.load_state_dict(torch.load(realnvp_checkpoint, map_location='cpu'))

bs = 256
syn_bs = 16

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
])

dataset_type = dataset_dict['FFHQAge']
mem_ffhq_dataset = dataset_type(data_root, img_size=256, split='train', transform=test_transform)
mem_ffhq_loader = torch.utils.data.DataLoader(
    dataset_with_indices(mem_ffhq_dataset),
    batch_size=bs,
    shuffle=False,
    drop_last=False,
    num_workers=8)
train_ffhq_loader = torch.utils.data.DataLoader(
    dataset_with_indices(mem_ffhq_dataset),
    batch_size=bs,
    shuffle=True,
    drop_last=True,
    num_workers=8)

logger = LoggerX(save_root=osp.join('./ckpt', opts.run_name),
                 project=opts.project_name,
                 entity=opts.entity,
                 config=opts,
                 enable_wandb=opts.wandb,
                 name=opts.run_name)

c_dim = 0
for attr_num in opts.changes:
    if attr_num != -1:
        c_dim += 1
    else:
        c_dim += 6
model = AdaTrans(c_dim=c_dim, max_steps=opts.max_steps, n_layers=10, hid_dim=512, n_styles=G.n_styles,
                 style_dim=G.style_dim).cuda()
logger.modules = [model, ]

optimizer = torch.optim.AdamW(list(model.parameters()), lr=0.0001, weight_decay=0.0)
acc_grd_step = 8
max_iter = 10000

train_ffhq_loader_iter = iter(train_ffhq_loader)

scaler = NativeScalerWithGradNormCount()


def create_indices(l):
    indices = []
    for attr_num in l:
        if attr_num != -1:
            indices.append(attr_num)
        else:
            indices += list(range(40, 46))
    indices = torch.tensor(indices).long().cuda()
    return indices


changes_indices, keeps_indices = create_indices(opts.changes), create_indices(opts.keeps)


@torch.no_grad()
def test(n_iter):
    num_tests = 8
    indices = torch.arange(num_tests) + 48
    images = torch.stack([mem_ffhq_dataset[i][0] for i in indices]).cuda()

    latents = all_latents[indices].clone()
    sources = all_preds[indices].clone()

    model.eval()
    out_images = [images, G(latents), ]

    for i, attr_num in enumerate(opts.changes):
        targets = sources.clone()
        targets[:, opts.changes] = F.one_hot(torch.ones(len(targets)).long() * i, len(opts.changes)).float().cuda()
        targets_ = targets[:, changes_indices]
        targets_[targets_ == 0] = -1.
        out_images.append(G(model(latents, targets_)[0][-1]))

    grid_img = torch.stack(out_images).transpose(1, 0).reshape(-1, 3, 256, 256)
    grid_img = grid_img.clamp(-1, 1)
    grid_img = grid_img * 0.5 + 0.5
    grid_img = to_pil_image(make_grid(grid_img, len(out_images)))
    logger.save_image(grid_img, n_iter, sample_type='train')


for n_iter in range(1, 1 + max_iter):
    try:
        (images, _), indices = convert_to_cuda(next(train_ffhq_loader_iter))
    except StopIteration:
        train_ffhq_loader_iter = iter(train_ffhq_loader)
        (images, _), indices = convert_to_cuda(next(train_ffhq_loader_iter))

    with torch.autograd.set_grad_enabled(True):
        latents = all_latents[indices]
        sources = all_preds[indices]

        bs = latents.size(0)

        model.train()
        targets = sources.clone()

        rand_indices = torch.randint(0, len(opts.changes), (bs, ))
        targets[:, changes_indices] = F.one_hot(rand_indices, len(opts.changes)).float().cuda()
        targets_ = targets[:, changes_indices]
        targets_[targets_ == 0] = -1.
        trajectory, step_size = model(latents, targets_)
        new_styles = trajectory[-1]

        step_size = step_size.sum(dim=1).mean()
        new_styles = new_styles.float()

        norm = (new_styles - latents).norm(dim=-1).mean()

        new_images = G(new_styles[:syn_bs])
        image_classifier.eval()
        outs = image_classifier(new_images)[0]

        class_indices = torch.cat([changes_indices, keeps_indices])
        cls_loss = F.binary_cross_entropy_with_logits(outs[:, class_indices], targets[:syn_bs, class_indices])

        # target_images = images[:syn_bs]
        with torch.no_grad():
            target_images = G(latents[:syn_bs])
        id_loss = ID_LOSS(target_images, new_images)
        lpips_loss = LIPIPS_LOSS(target_images, new_images)
        pix_loss = F.mse_loss(target_images, new_images)

        nll_loss, logz, _, _ = realnvp(trajectory[1:].view(-1, new_styles.size(1), new_styles.size(2)))

        recon_loss = F.mse_loss(latents, new_styles)

        loss = recon_loss * opts.recon_loss_weight + \
               cls_loss * opts.cls_loss_weight + \
               nll_loss * opts.nll_loss_weight + \
               id_loss * opts.id_loss_weight + \
               lpips_loss * opts.lpips_loss_weight + \
               pix_loss * opts.pix_loss_weight
        update_params = (n_iter % acc_grd_step == 0)
        loss = loss / acc_grd_step
        scaler(loss, optimizer=optimizer, update_grad=update_params)

        logger.msg([recon_loss, id_loss, lpips_loss, pix_loss, norm, nll_loss, logz, cls_loss, step_size], n_iter)

        if n_iter % opts.save_freq == 0:
            test(n_iter)
            logger.checkpoints('latest')
