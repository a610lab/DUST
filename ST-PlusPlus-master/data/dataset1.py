import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from data.transform import Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomZoom, RandomCrop, ToTensor
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
from skimage.transform import rotate
from skimage.transform import resize
from torchvision import transforms


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split=None, num=None, transform=None):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        if self.split == 'train' or self.split == 'label':
            with open(self._base_dir + '/train_slices.list', 'r') as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        else:
            if self.split == 'val':
                with open(self._base_dir + '/val.list', 'r') as f:
                    self.sample_list = f.readlines()
                self.sample_list = [item.replace('\n', '')
                                    for item in self.sample_list]
            elif self.split == 'semitrain' or self.split == 'finaltrain':
                with open(self._base_dir + '/reliable.list', 'r') as f:
                    self.sample_list = f.readlines()
                self.sample_list = [item.replace('\n', '')
                                    for item in self.sample_list]

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]
        if num is not None and self.split == "label":
            self.sample_list = self.sample_list[num:]
        if num is not None and self.split == "semitrain":
            self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train" or self.split == 'label' or self.split == 'semitrain' or self.split == 'finaltrain':
            h5f = h5py.File(self._base_dir +
                            "/data/slices/{}.h5".format(case), 'r')
        else:
            if self.split == 'val':
                h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.split == "train" or self.split == 'label' or self.split == 'semitrain' or self.split == 'finaltrain':
            sample = self.transform(sample)
        sample["idx"] = idx
        if self.split == 'label':
            return sample, case

        return sample


class BaseSTDataSets(Dataset):
    def __init__(self, imgs, plabs):
        self.img = [img.cpu().squeeze(0).numpy() for img in imgs]
        self.plab = [lab.cpu().squeeze().numpy() for lab in plabs]
        self.num = len(self.img)
        # self.tr_transform = transforms.Compose([
        #                         RandomGenerator([256, 256])
        #                     ])

    def __getitem__(self, idx):
        image, label = self.img[idx], self.plab[idx]
        sample = {'image': image, 'label': label}
        # sample = self.tr_transform(sample)
        return sample

    def __len__(self):
        return self.num


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(
            image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(
            label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(
            image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {'image': image, 'label': label}
        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


# class BaseDataSets(Dataset):
#     def __init__(self, base_dir=None, split=None, num=None, transform1=None, transform2=None):
#         self._base_dir = base_dir
#         self.sample_list = []
#         self.split = split
#         if self.split == 'train' or self.split == 'label':
#             with open(self._base_dir + '/train_slices.list', 'r') as f1:
#                 self.sample_list = f1.readlines()
#             self.sample_list = [item.replace('\n', '')
#                                 for item in self.sample_list]
#         else:
#             if self.split == 'val':
#                 with open(self._base_dir + '/val.list', 'r') as f:
#                     self.sample_list = f.readlines()
#                 self.sample_list = [item.replace('\n', '')
#                                     for item in self.sample_list]
#             elif self.split == 'semitrain' or self.split == 'finaltrain':
#                 with open(self._base_dir + '/reliable.list', 'r') as f:
#                     self.sample_list = f.readlines()
#                 self.sample_list = [item.replace('\n', '')
#                                     for item in self.sample_list]
#
#         if num is not None and self.split == "train":
#             self.sample_list = self.sample_list[:num]
#         if num is not None and self.split == "label":
#             self.sample_list = self.sample_list[num:]
#         if num is not None and self.split == "semitrain":
#             self.sample_list = self.sample_list[:num]
#         print("total {} samples".format(len(self.sample_list)))
#
#         transform2 = transforms.Compose([
#             transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
#         ])
#
#         if self.split == "train" or self.split == 'label':
#                transform1 = transforms.Compose([
#                    # Resize((500, 500)),
#                    RandomHorizontalFlip(p=0.5),
#                    RandomVerticalFlip(p=0.5),
#                    RandomRotation(90),
#                    # RandomZoom((0.9, 1.1)),
#                    # RandomCrop((464, 464)),
#                    ToTensor(),
#                ])
#         elif self.split == 'semitrain' or self.split == 'finaltrain':
#                 transform1 = transforms.Compose([
#                     # transforms.Resize((500, 500)),
#                     transforms.RandomHorizontalFlip(p=0.5),
#                     transforms.RandomVerticalFlip(p=0.5),
#                     transforms.RandomRotation(90),
#                     # transforms.RandomCrop((464, 464)),
#                     transforms.ToTensor(),
#                 ])
#         elif self.split == 'val':
#                 transform1 = transforms.Compose([
#                    # Resize((512, 512)),
#                    ToTensor(),
#                 ])
#         self.transform1 = transform1
#         self.transform2 = transform2
#
#
#     def __len__(self):
#         return len(self.sample_list)
#
#     def __getitem__(self, idx):
#         case = self.sample_list[idx]
#         if self.split == "train" or self.split == 'label' or self.split == 'semitrain' or self.split == 'finaltrain':
#             image = Image.open(self._base_dir + '/data/train/images/{}.png'.format(case)).convert('RGB')
#             label = Image.open(self._base_dir + '/data/train/masks/{}.png'.format(case)).convert('L')
#         elif self.split == 'val':
#             image = Image.open(self._base_dir + '/data/val/images/{}.png'.format(case)).convert('RGB')
#             label = Image.open(self._base_dir + '/data/val/masks/{}.png'.format(case)).convert('L')
#         sample = {'image': image, 'label': label}
#         sample = self.transform1(sample)
#         sample['image'] = self.transform2(sample['image'])
#         sample["idx"] = idx
#         if self.split == 'label':
#             return sample, case
#
#         return sample
