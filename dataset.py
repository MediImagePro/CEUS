import random
import torchvision
import os.path
import torch.utils.data as data
import pandas as pd
from PIL import Image
from util import *
import numpy as np
import torchvision.transforms as transforms
from abc import ABC, abstractmethod

from util import extract_bboxes

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


# Function to check if a file is an image based on its extension
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# Function to create a dataset of image file paths from a directory
def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


# Dataset class for aligned images
class AlignedDataset:

    def __init__(self, data_root, phase='train'):
        # Initialization of the dataset
        super(AlignedDataset, self).__init__()
        self.data_root = data_root
        self.phase = phase

        # Image processing parameters
        self.load_size = 286
        self.crop_size = 256
        self.input_nc = 1
        self.output_nc = 3

        self.dir_ABM = os.path.join(self.data_root, self.phase)  # Image directory path
        self.ABM_paths = sorted(make_dataset(self.dir_ABM))  # List of image paths

    # Method to get an item from the dataset
    def __getitem__(self, index):

        ABM_path = self.ABM_paths[index]
        ABM = Image.open(ABM_path).convert('RGB')

        # Splitting the image into three parts
        w, h = ABM.size
        w2 = int(w / 3)
        A = ABM.crop((0, 0, w2, h))  # First part
        B = ABM.crop((w2, 0, 2 * w2, h))  # Second part
        M = ABM.crop((2 * w2, 0, w, h))  # Third part

        # Image transformations
        transform_params = self.get_params(A.size)
        A_transform = self.get_transform(transform_params, grayscale=(self.input_nc == 1))
        B_transform = self.get_transform(transform_params, grayscale=(self.output_nc == 1))
        M_transform = self.get_transform(transform_params, grayscale=True)

        # Applying transformations
        A = A_transform(A)
        B = B_transform(B)
        M = M_transform(M)

        # Processing mask image and extracting bounding boxes
        M_num = M.numpy()
        M_num = np.squeeze(M_num)
        logical_M = np.where(M_num < 0, 0, 1)
        bbox = extract_bboxes(logical_M)  # Extracting bounding boxes

        return {'A': A, 'B': B, 'M': M, 'bbox': bbox}

    # Method to get the length of the dataset
    def __len__(self):

        return len(self.ABM_paths)

    # Function to get image transformation parameters
    def get_params(self, size):
        w, h = size
        new_h = new_w = self.load_size

        x = random.randint(0, np.maximum(0, new_w - self.crop_size))
        y = random.randint(0, np.maximum(0, new_h - self.crop_size))

        flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}

    # Function to compose image transformations
    def get_transform(self, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
        transform_list = []
        if grayscale:
            transform_list.append(transforms.Grayscale(1))

        osize = [self.load_size, self.load_size]
        transform_list.append(transforms.Resize(osize, method))

        if params is None:
            transform_list.append(transforms.RandomCrop(self.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: my_crop(img, params['crop_pos'], self.crop_size)))

        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: my_flip(img, params['flip'])))

        transform_list += [transforms.ToTensor()]
        if convert:
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)


# Function to resize an image to the nearest power of 2
def my_make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img

    my_print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


# Function to scale the width of an image while maintaining aspect ratio
def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)


# Function to crop an image at the specified position and size
def my_crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


# Function to flip an image horizontally if flip is True
def my_flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


# Function to print a warning if the image size is adjusted
def my_print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(my_print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        my_print_size_warning.has_printed = True
