import os


import torch
import random
import numpy as np
from data.transform import crop, hflip, normalize, resize, blur, cutout
from PIL import Image
from torch.utils.data import Dataset
import h5py
from scipy.ndimage.interpolation import zoom
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
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


# class KvasirDataSets(Dataset):
#     def __init__(self, base_dir=None, split=None, labeled_id_path=None, unlabeled_id_path=None, transform=None):
#         self._base_dir = base_dir
#         self.split = split
#         self.transform = transform
#         if split == 'semi_train':
#             with open(unlabeled_id_path, 'r') as f:
#                 self.ids = f.read().splitlines()
#         else:
#             if split == 'val':
#                 id_path = 'data/Kvasir/val.list'
#             elif split == 'label':
#                 id_path = unlabeled_id_path
#             elif split == 'train':
#                 id_path = labeled_id_path
#
#             with open(id_path, 'r') as f:
#                 self.ids = f.read().splitlines()
#
#     def __getitem__(self, item):
#         id = self.ids[item]
#         img = Image.open(os.path.join(self._base_dir, id.split(' ')[0])).convert('RGB')
#
#         if self.split == 'val' or self.split == 'label':
#             mask = Image.open(os.path.join(self._base_dir, id.split(' ')[1])).convert('L')
#             img, mask = resize(img, mask, base_size=320)
#             img, mask = normalize(img, mask)
#             return img, mask, id
#
#         if self.split == 'train':
#             mask = Image.open(os.path.join(self._base_dir, id.split(' ')[1])).convert('L')
#             img, mask = resize(img, mask, base_size=320)
#             img, mask = crop(img, mask, size=256)
#             img, mask = hflip(img, mask, p=0.5)
#
#         # strong augmentation on unlabeled images
#         if self.split == 'semi_train':
#             mask = Image.open(os.path.join(self._base_dir, id.split(' ')[1])).convert('L')
#             img, mask = resize(img, mask, base_size=320)
#             img, mask = crop(img, mask, size=256)
#             img, mask = hflip(img, mask, p=0.5)
#             if random.random() < 0.8:
#                 img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
#             img = transforms.RandomGrayscale(p=0.2)(img)
#             img = blur(img, p=0.5)
#
#         img, mask = normalize(img, mask)
#
#         return img, mask
#
#     def __len__(self):
#         return len(self.ids)
