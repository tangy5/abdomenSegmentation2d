from __future__ import print_function

import torch.utils.data as data
import os
import random
import glob
from PIL import Image
from utils import preprocess

_FOLDERS_MAP = {
    'image': 'images',
    'label': 'labels',
}

_POSTFIX_MAP = {
    'image': '_leftImg8bit',
    'label': '_gtFine_labelTrainIds',
}

_DATA_FORMAT_MAP = {
    'image': 'png',
    'label': 'png',
}


class Multiorgan(data.Dataset):
  CLASSES = [
      'bk', 'spleen', 'right_kidney', 'left_kidney', 'gallbladder', 'esophagus', 'liver', 'stomach',
      'aorta', 'IVC', 'PSV', 'pancreas', 'r_adrenal_gland', 'l_adrenal_gland'
  ]

  def __init__(self, root, train=True, transform=None, target_transform=None, download=False, crop_size=None, dataset='multiorgan'):
    self.root = root
    self.transform = transform
    self.target_transform = target_transform
    self.train = train
    self.crop_size = crop_size
    self.dataset = dataset
    dataset_split = 'train' if self.train else 'val'
    self.images = self._get_files('image', dataset_split, dataset)
    self.masks = self.images
    # self._get_files('label', dataset_split, dataset)

  def __getitem__(self, index):
    _img = Image.open(self.images[index]).convert('RGB')

    # _target = Image.open(self.masks[index])
    _target = _img
    if self.dataset == 'multiorgan':
      _img, _target = preprocess(_img, _target,
                                flip=True if self.train else False,
                                scale=(0.5, 2.0) if self.train else None,
                                crop=(self.crop_size, self.crop_size) if self.train else (512, 512))
    else:
      _img, _target = preprocess(_img, _target,
                                flip=True if self.train else False,
                                scale=None if self.train else None,
                                crop=(self.crop_size, self.crop_size) if self.train else (1025, 2049))      

    if self.transform is not None:
      _img = self.transform(_img)

    if self.target_transform is not None:
      _target = self.target_transform(_target)
    # print(_img.shape)
    return _img, _target

  def _get_files(self, data, dataset_split, dataset):
    pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
    if dataset == 'multiorgan': 
      pattern = '*.%s' % (_DATA_FORMAT_MAP[data])
    search_files = os.path.join(
        self.root, _FOLDERS_MAP[data], dataset_split, '*', pattern)
    # print(search_files)
    filenames = glob.glob(search_files)
    return sorted(filenames)

  def __len__(self):
    return len(self.images)
