import time

import cv2
import sys
import json

import pandas as pd
import torch
import warnings
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor

torch.autograd.set_grad_enabled(False)

sys.path.insert(0, '/home/zzhuang/AdaTrans')
sys.path.insert(0, '/home/zzhuang/AdaTrans/models/stylegan2')

from models.decoder import StyleGANDecoder
from models.e4e.psp_encoders import Encoder4Editing
from models.modules import Classifier
from models.ops import load_network, age2group
from models.modules import AdaTrans
from models.face_align.dlib_face_align import face_alignment, face_alignment_inverse
from models.face_seg.deeplab import resnet101 as deeplab

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AdaTrans Demo",
    page_icon="🚀",
    # layout="wide",
)

device = "cuda" if torch.cuda.is_available() else 'cpu'
all_attributes_list = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                       'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
                       'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup',
                       'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                       'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
                       'Wearing_Necklace', 'Wearing_Necktie', 'Young'
                       ]
all_ages_list = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-69', '70-120']
all_hair_colors = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Unknown']
all_hair_types = ['Straight_Hair', 'Wavy_Hair', 'Unknown']

all_checkpoints = {
    'Eyeglasses': 'data/ckpt/15/save_models/model-latest',
    'Male': 'data/ckpt/20/save_models/model-latest',
    'Smiling': 'data/ckpt/31/save_models/model-latest',
    'Age': 'data/ckpt/Age/save_models/model-latest',
    'Hair_Color': 'data/ckpt/8_9_11/save_models/model-latest',
    'Hair_Type': 'data/ckpt/32_33/save_models/model-latest',
}
# max_steps = 5
max_steps = 10


class Trans(object):

    def __init__(self):
        stylegan2_checkpoint = 'data/ffhq.pkl'
        e4e_checkpoint = 'data/e4e_ffhq_encode.pt'
        classifier_checkpoint = 'data/r34_a40_age_256_classifier.pth'
        deeplab_checkpoint = 'data/deeplab_model.pth'

        G = StyleGANDecoder(
            stylegan2_checkpoint,
            start_from_latent_avg=False,
            output_size=256,
        ).eval().to(device)

        encoder = Encoder4Editing(50, 'ir_se', stylegan_size=1024, checkpoint_path=e4e_checkpoint).eval().to(device)
        image_classifier = Classifier().eval().to(device)
        image_classifier.load_state_dict(load_network(torch.load(classifier_checkpoint, map_location='cpu')))

        deeplab_classes = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear',
                           'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
        deeplab_model = deeplab(num_classes=len(deeplab_classes), num_groups=32, weight_std=True,
                                beta=False).eval().to(device)
        checkpoint = torch.load(deeplab_checkpoint, map_location='cpu')
        deeplab_model.load_state_dict({k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k})

        self.models = {}
        for a in all_checkpoints:
            if a == 'Age':
                c_dim = len(all_ages_list) - 1
            elif a == 'Hair_Color':
                c_dim = len(all_hair_colors) - 1
            elif a == 'Hair_Type':
                c_dim = len(all_hair_types) - 1
            else:
                c_dim = 1
            model = AdaTrans(c_dim=c_dim, max_steps=max_steps, n_layers=10,
                             hid_dim=512, n_styles=G.n_styles,
                             style_dim=G.style_dim).eval().to(device)
            model.load_state_dict(load_network(torch.load(all_checkpoints[a], map_location='cpu')))
            self.models[a] = model
        self.G = G
        self.encoder = encoder
        self.image_classifier = image_classifier
        self.deeplab_model = deeplab_model
        print('load complete...')


@st.cache_data()
@torch.no_grad()
def encode(image):
    st = time.time()
    ret = face_alignment(image, output_size=256)
    print(f'aligning image takes {time.time() - st}')

    if ret is None:
        return None
    input_image_np, inv_M = ret
    input_image = torch.from_numpy(input_image_np).permute((2, 0, 1)).to(device).unsqueeze(0).float()
    input_image = (input_image / 255. - 0.5) * 2

    st = time.time()
    latents = model.encoder(input_image) + model.G.latent_avg
    print(f'encoding image takes {time.time() - st}')

    st = time.time()
    attributes = classify(input_image.cpu().numpy())
    print(f'classifying image takes {time.time() - st}')

    st = time.time()
    seg_mask = face_segment(input_image_np)
    print(f'segmenting image takes {time.time() - st}')

    outs = {
        'latents': latents.cpu().numpy(),
        'attributes': attributes,
        'inv_M': inv_M,
        'seg_mask': seg_mask,
        'aligned_image': input_image_np,
        'image': image,
    }
    return outs


@torch.no_grad()
@st.cache_data()
def classify(image):
    image = torch.from_numpy(image).to(device)
    cur_attributes = model.image_classifier(image)[0].sigmoid().squeeze()
    attributes = {}
    attributes['Age'] = all_ages_list[(cur_attributes[40:] > 0.5).sum()]
    for i, a in enumerate(all_attributes_list):
        attributes[a] = round(float(cur_attributes[i]), 3)

    hair_color_preds = cur_attributes[[8, 9, 11]]
    c = torch.argmax(hair_color_preds)
    if hair_color_preds[c] > 0.5:
        c = all_hair_colors[c]
    else:
        c = 'Unknown'
    attributes['Hair_Color'] = c

    hair_type_preds = cur_attributes[[32, 33]]
    c = torch.argmax(hair_type_preds)
    if hair_type_preds[c] > 0.5:
        c = all_hair_types[c]
    else:
        c = 'Unknown'
    attributes['Hair_Type'] = c
    return attributes


@torch.no_grad()
@st.cache_data()
def face_segment(image):
    deeplab_input_size = 513
    input = to_tensor(Image.fromarray(image).resize((deeplab_input_size, deeplab_input_size), Image.BILINEAR))
    input = input.unsqueeze(0).to(device)
    outputs = model.deeplab_model(input)
    _, seg_mask = torch.max(outputs, 1)
    seg_mask = seg_mask.unsqueeze(1)
    seg_mask = (seg_mask != 0) & (seg_mask != 18)
    seg_mask = F.interpolate(seg_mask.float(), size=256)
    seg_mask = seg_mask.repeat(1, 3, 1, 1)
    seg_mask = seg_mask.squeeze(0).permute((1, 2, 0)).cpu().numpy()
    return seg_mask


@st.cache_data()
@torch.no_grad()
def transform(ret, opts, targets: dict):
    cur_latents = torch.from_numpy(ret['latents']).to(device)
    cur_attributes = ret['attributes']
    image = ret['image']
    aligned_image = ret['aligned_image']
    inv_M = ret['inv_M']
    seg_mask = ret['seg_mask']

    new_latents = cur_latents.clone()
    for t in targets:
        model.models[t].max_steps = int(opts['opt_steps'])
        if t == 'Age':
            if targets[t] != cur_attributes[t]:
                targets_ = torch.ones(1).to(device) * all_ages_list.index(targets[t])
                targets_ = age2group(groups=targets_, ordinal=True, age_group=7).float().to(device)
                targets_[targets_ == 0] = -1
                new_latents += model.models[t](cur_latents, targets_)[0][-1] - cur_latents
        elif t == 'Hair_Color':
            if targets[t] != cur_attributes[t]:
                targets_ = torch.ones(len(all_hair_colors) - 1).to(device) * 0
                if targets[t] != 'Unknown':
                    targets_[all_hair_colors.index(targets[t])] = 1.0
                targets_[targets_ == 0] = -1
                new_latents += model.models[t](cur_latents, targets_.unsqueeze(0))[0][-1] - cur_latents
        elif t == 'Hair_Type':
            if targets[t] != cur_attributes[t]:
                targets_ = torch.ones(len(all_hair_types) - 1).to(device) * 0
                if targets[t] != 'Unknown':
                    targets_[all_hair_types.index(targets[t])] = 1.0
                targets_[targets_ == 0] = -1
                new_latents += model.models[t](cur_latents, targets_.unsqueeze(0))[0][-1] - cur_latents
        else:
            if targets[t] != (float(cur_attributes[t]) > 0.5):
                # if abs(targets[t] - cur_attributes[t]) > 0.1:
                targets_ = torch.ones(1).to(device) * (1 if targets[t] == 1 else -1)
                new_latents += model.models[t](cur_latents, targets_.unsqueeze(1))[0][-1] - cur_latents
    st = time.time()
    edit_image = model.G(new_latents).clamp(-1., 1.0)
    print(f'decoding image takes {time.time() - st}')

    new_attributes = classify(edit_image.cpu().numpy())
    edit_image = edit_image * 0.5 + 0.5
    edit_image = (edit_image * 255).squeeze(0).permute((1, 2, 0)).cpu().numpy().astype(np.uint8)

    if opts['opt_seg']:
        from scipy.ndimage.filters import gaussian_filter
        bg = gaussian_filter(aligned_image.copy(), [19, 19, 0])
        # org_seg_mask = face_alignment_inverse(np.zeros_like(image), seg_mask, inv_M, output_size=256)
        new_seg_mask = face_segment(edit_image)
        m = (new_seg_mask.astype(bool) | seg_mask.astype(bool)).astype(np.float32)
        bg = bg * m + aligned_image * (1 - m)
        edit_image = (1 - new_seg_mask) * bg + new_seg_mask * edit_image
    new_image = edit_image
    if opts['opt_inverse']:
        new_image = face_alignment_inverse(image, new_image, inv_M, output_size=256)
    new_image = new_image.astype(np.uint8)

    return new_image, new_attributes


@st.cache_resource()
def load_model():
    model = Trans()
    return model


st.title("✨Adaptive Transformation 🏜")
st.info(' Let me help edit faces for any of your images. 😉')
model = load_model()
# option = st.sidebar.selectbox('Setting the editing strength to...', list(range(1, 16)), index=max_steps - 1)
# for name in model.models:
#     model.models[name].max_steps = int(option)
#     print(model.models[name].max_steps)

opt_seg = st.sidebar.checkbox('Use face segment model?', value=1)
opt_inverse = st.sidebar.checkbox('Paste to source image?', value=1)
opt_steps = st.sidebar.selectbox('Max steps?', list(range(max_steps - 6, max_steps + 7, 2)), index=3)
st.sidebar.markdown("# Attributes")


def show_attrs(attrs: dict):
    data = []
    for k, v in attrs.items():
        if k not in all_checkpoints:
            continue
        data.append([k, v])
    df = pd.DataFrame(data, columns=['Attributes', 'Values'])
    st.dataframe(df, use_container_width=True)


def show_image(col, image, attributes, content):
    with col:
        st.image(image, use_column_width=True)
        st.success(content)
        if attributes is not None:
            show_attrs(attributes)
        buf = BytesIO()
        Image.fromarray(image).save(buf, format="PNG")
        st.download_button(label="Download image", data=buf.getvalue(), file_name="image.png", mime="image/png")


def resize_image_with_max_long_side(img, max_long_side=1024):
    # Get the image dimensions
    height, width = img.shape[:2]

    # Calculate the scaling factor to resize the image
    scaling_factor = max_long_side / max(height, width)
    if scaling_factor >= 1.0:
        return img

    # Calculate the new dimensions
    new_height = int(height * scaling_factor)
    new_width = int(width * scaling_factor)

    # Resize the image using the calculated dimensions
    resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image


image_path = st.file_uploader("Upload Image 🚀", type=["png", "jpg", "bmp", "jpeg"],
                              key='image_path')

if image_path is None:
    st.warning('⚠ Please upload your Image! 😯')
else:
    with st.spinner("Working.. 💫"):
        print('read image...')
        image = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = resize_image_with_max_long_side(image, max_long_side=1024)
        ret = encode(image)
        col1, col2 = st.columns(2)
        if ret is not None:
            attributes = ret['attributes']
            for a in all_checkpoints:
                if a == 'Age':
                    st.sidebar.selectbox(a, all_ages_list, key=a,
                                         index=all_ages_list.index(st.session_state.get(a, attributes[a])))
                elif a == 'Hair_Color':
                    st.sidebar.selectbox(a, all_hair_colors, key=a,
                                         index=all_hair_colors.index(st.session_state.get(a, attributes[a])))
                elif a == 'Hair_Type':
                    st.sidebar.selectbox(a, all_hair_types, key=a,
                                         index=all_hair_types.index(st.session_state.get(a, attributes[a])))
                else:
                    st.sidebar.checkbox(a, value=st.session_state.get(a, attributes[a] > 0.5), key=a)
            show_image(col1, ret['image'] if opt_inverse else ret['aligned_image'], attributes, 'Original Image')
            targets = {}
            for a in all_checkpoints:
                targets[a] = st.session_state[a]
            opts = {'opt_inverse': opt_inverse, 'opt_seg': opt_seg, 'opt_steps': opt_steps}
            print(targets, opts)
            new_image, new_attributes = transform(ret, opts, targets)
            show_image(col2, new_image, new_attributes, 'Output Image')
        else:
            show_image(col1, image, None, 'Original Image')
            with col2:
                st.error('No face detected!')

st.markdown(
    "<br><hr><center>Made by <strong>Zhizhong Huang</strong> ✨",
    unsafe_allow_html=True)

st.markdown('**Arxiv**: https://arxiv.org/abs/2307.07790')
st.markdown('**Github**: https://github.com/Hzzone/AdaTrans')
st.markdown('Please cite our paper:')
st.code(
    '''
    @inproceedings{huang2023adaptive,
      title={Adaptive Nonlinear Latent Transformation for Conditional Face Editing
    },
      author={Huang, Zhizhong and Ma, Siteng and Zhang, Junping and Shan, Hongming},
      booktitle={ICCV},
      year={2023}
    }
    ''',
    language='latex'
)
