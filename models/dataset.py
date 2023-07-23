import csv
import os.path
import os.path as osp
import pathlib

import torch
from torchvision.datasets.folder import pil_loader
import torchvision
import numpy as np
from functools import partial
from torchvision import transforms, datasets


def create_FFHQ_AGING(data_root, label_file, age_group):
    if age_group == 7:
        age_clusters = {'0-2': 0, '3-6': 0, '7-9': 0, '10-14': 1, '15-19': 1,
                        '20-29': 2, '30-39': 3, '40-49': 4, '50-69': 5, '70-120': 6}
    elif age_group == 10:
        age_clusters = {'0-2': 0, '3-6': 1, '7-9': 2, '10-14': 3, '15-19': 4,
                        '20-29': 5, '30-39': 6, '40-49': 7, '50-69': 8, '70-120': 9}
    else:
        raise NotImplementedError

    f = open(label_file, 'r', newline='')
    reader = csv.DictReader(f)
    img_list = []

    missing_counts = 0
    for csv_row in reader:
        age, age_conf = csv_row['age_group'], float(csv_row['age_group_confidence'])
        gender, gender_conf = csv_row['gender'], float(csv_row['gender_confidence'])
        head_pitch, head_roll, head_yaw = float(csv_row['head_pitch']), float(csv_row['head_roll']), float(
            csv_row['head_yaw'])
        left_eye_occluded, right_eye_occluded = float(csv_row['left_eye_occluded']), float(
            csv_row['right_eye_occluded'])
        glasses = csv_row['glasses']

        no_attributes_found = head_pitch == -1 and head_roll == -1 and head_yaw == -1 and \
                              left_eye_occluded == -1 and right_eye_occluded == -1 and glasses == -1

        age_cond = age_conf > 0.6
        gender_cond = gender_conf > 0.66

        head_pose_cond = abs(head_pitch) < 30.0 and abs(head_yaw) < 40.0
        eyes_cond = (left_eye_occluded < 90.0 and right_eye_occluded < 50.0) or (
                left_eye_occluded < 50.0 and right_eye_occluded < 90.0)
        glasses_cond = glasses != 'Dark'

        valid1 = age_cond and gender_cond and no_attributes_found
        valid2 = age_cond and gender_cond and head_pose_cond and eyes_cond and glasses_cond

        if (valid1 or valid2):
            num = int(csv_row['image_number'])
            if num < 69000:
                train = 1
            else:
                train = 0
            img_filename = str(num).zfill(5) + '.png'
            glasses = 0 if glasses == 'None' else 1
            gender = 1 if gender == 'male' else 0
            age = age_clusters[age]
            img_path = osp.join(data_root, img_filename)
            if not osp.exists(img_path):
                missing_counts += 1
                continue
            img_list.append([img_path, glasses, age, gender, train])
    if missing_counts > 0:
        print(f'{missing_counts} images do not exist.')
    return np.array(img_list)


normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)


def create_FACE_Transform(mode, img_size):
    transform = []
    if mode == 'test':
        transform.append(torchvision.transforms.Resize(img_size))
    elif mode == 'train':
        transform += [
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Resize(img_size),
        ]
    transform.append(transforms.ToTensor())
    transform.append(normalize)
    transform = transforms.Compose(transform)
    return transform


class FFHQAge(torch.utils.data.Dataset):
    def __init__(self, data_root, age_group=7, img_size=128, split='train', mode='train', transform=None):
        label_file = 'data/ffhq_aging_labels.csv'
        self.img_list = create_FFHQ_AGING(data_root, label_file, age_group)
        if split == 'train':
            self.img_list = self.img_list[self.img_list[:, -1].astype(int) == 1]
        else:
            self.img_list = self.img_list[self.img_list[:, -1].astype(int) == 0]

        self.transform = transform
        if transform is None:
            self.transform = create_FACE_Transform(mode, img_size)
        self.targets = torch.Tensor(self.img_list[:, 2].astype(int))
        self.age_group = age_group

    def __getitem__(self, idx):
        line = self.img_list[idx]
        img = pil_loader(line[0])
        if self.transform is not None:
            img = self.transform(img)
        group = int(line[2].astype(int))
        # from zzhuang_ops.ops import age2group
        # ordinal_label = age2group(groups=torch.Tensor([group]), age_group=self.age_group, ordinal=True)[0]
        # return img, group, ordinal_label
        return img, group

    def __len__(self):
        return len(self.img_list)


class FFHQ(torch.utils.data.Dataset):
    def __init__(self, data_root, img_size=128, split='train', mode='train', transform=None):
        self.img_list = list(pathlib.Path(data_root).rglob('*.png'))
        self.img_list = sorted(self.img_list)

        self.transform = transform
        if transform is None:
            self.transform = create_FACE_Transform(mode, img_size)

    def __getitem__(self, idx):
        img = pil_loader(self.img_list[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_list)


class CelebA(torch.utils.data.Dataset):
    def __init__(self, data_root, img_size=128, split='train', mode='train', transform=None, **kwargs):
        self.transform = transform
        if transform is None:
            self.transform = create_FACE_Transform(mode, img_size)
        img_list = sorted(list(pathlib.Path(data_root).rglob("*.jpg")))
        label_file = '/home/zzhuang/DATASET/celeba/list_attr_celeba.txt'
        eval_file = '/home/zzhuang/DATASET/celeba/list_eval_partition.txt'
        eval_dict = {}
        for line in open(eval_file).readlines():
            x1, x2 = line.strip().split(' ')
            eval_dict[x1] = int(x2)
        targets = {}
        lines = open(label_file).readlines()
        self.class2idx = lines[1].split()
        for line in lines[2:]:
            line = [l for l in line.strip().split(' ') if l != '']
            assert len(line) == 41
            targets[line[0]] = (np.array(line[1:]).astype(int) > 0.).astype(int)
        self.targets = []
        self.img_list = []
        for fpath in img_list:
            fname = fpath.name
            if split == 'train' and eval_dict[fname] != 0:
                continue
            if split == 'val' and eval_dict[fname] != 1:
                continue
            if split == 'test' and eval_dict[fname] != 2:
                continue
            self.targets.append(targets[fname])
            self.img_list.append(fpath)
        self.targets = torch.from_numpy(np.stack(self.targets)).float()
        # eyeglasses, male, no_beard, smiling, young
        # class_idx = [15, 20, 24, 31, 39]
        class_idx = kwargs.get('class_idx', np.arange(40))
        # class_idx = [15, 20, 31, 39]
        # class_idx = np.arange(40)
        self.targets = self.targets[:, class_idx]
        self.class2idx = np.array(self.class2idx)[class_idx]

    def __getitem__(self, idx):
        img = pil_loader(self.img_list[idx]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        target = self.targets[idx]
        return img, target

    def __len__(self):
        return len(self.img_list)


dataset_dict = {
    'FFHQAge': FFHQAge,
    'FFHQ': FFHQ,
    'CelebA': CelebA,
}
