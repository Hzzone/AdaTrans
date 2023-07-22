import math

import torchvision.transforms
from torch import nn
import torch
from typing import Union
import torch.distributed as dist
import torch.nn.functional as F
import collections.abc as container_abcs
from PIL import Image
import numpy as np


def load_network(state_dict):
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # remove `module.`
        new_state_dict[namekey] = v
    return new_state_dict


def resize(x, s):
    if x.size(2) == s and x.size(3) == s:
        return x
    return F.interpolate(x, size=s, mode='bicubic', align_corners=True, antialias=True)

def convert_to_cuda(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        if data.is_cuda:
            return data
        return data.cuda(non_blocking=True)
    elif isinstance(data, container_abcs.Mapping):
        return {key: convert_to_cuda(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(convert_to_cuda(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, str):
        return [convert_to_cuda(d) for d in data]
    else:
        return data

class dataset_with_indices(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        outs = self.dataset[idx]
        return [outs, idx]

def age2group(age: torch.Tensor = None, groups: torch.Tensor = None, age_group: int = 4, bins: (list, tuple) = None,
              ordinal=False):
    # if isinstance(age, np.ndarray):
    #     groups = np.zeros_like(age)
    # elif isinstance(age, torch.Tensor):
    #     groups = torch.zeros_like(age).to(age.device)
    # elif isinstance(age, (int, float)):
    #     age = np.array([age, ])
    #     groups = np.zeros_like(age)
    # else:
    #     raise NotImplementedError

    if groups is None:
        assert age is not None
        groups = torch.zeros_like(age).to(age.device)

        if bins is not None:
            section = bins
            age_group = len(section) + 1
        else:
            if age_group == 4:
                section = [30, 40, 50]
            elif age_group == 5:
                section = [20, 30, 40, 50]
            elif age_group == 6:
                section = [10, 20, 30, 40, 50]
            elif age_group == 7:
                section = [10, 20, 30, 40, 50, 60]
            elif age_group == 8:
                # 0 - 12, 13 - 18, 19 - 25, 26 - 35, 36 - 45, 46 - 55, 56 - 65, ≥ 66
                section = [12, 18, 25, 35, 45, 55, 65]
            else:
                raise NotImplementedError

        for i, thresh in enumerate(section, 1):
            groups[age > thresh] = i

    # if isinstance(age, (int, float)):
    #     groups = int(groups)

    if ordinal:
        groups = groups.long()
        ordinal_labels = F.one_hot(groups, age_group)
        for i in range(groups.size(0)):
            ordinal_labels[i, :groups[i]] = 1.
        ordinal_labels = ordinal_labels[:, 1:].to(age)
        groups = ordinal_labels

    return groups