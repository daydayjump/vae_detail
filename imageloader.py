#! /usr/bin/env python
# -*- coding: utf-8 -*-

import torch.utils.data as data
import torch
from torchvision import transforms
from PIL import Image
import numpy as np 
import os
import os.path

'''
make sure the type of file

'''
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

'''
define the class of the imgs

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
'''
def norm_label(label):
    label_norm = []
    # 设置camera,顺序为从正中间顺时针方向，051为0,140为12.
    if label[3] == 51:
        label_norm.append(0)
    elif label[3] == 50:
        label_norm.append(1)
    elif label[3] == 41:
        label_norm.append(2)
    elif label[3] == 190:
        label_norm.append(3)
    elif label[3] == 200:
        label_norm.append(4)
    elif label[3] == 10:
        label_norm.append(5)
    elif label[3] == 240:
        label_norm.append(6)
    elif label[3] == 110:
        label_norm.append(7)
    elif label[3] == 120:
        label_norm.append(8)
    elif label[3] == 90:
        label_norm.append(9)    
    elif label[3] == 80:
        label_norm.append(10)
    elif label[3] == 130:
        label_norm.append(11)
    elif label[3] == 140:
        label_norm.append(12)
    return label_norm


def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    for sessions in sorted(os.listdir(dir)):
        dir_1 = os.path.join(dir, sessions)
        if not os.path.isdir(dir_1):
            continue
        for recodings in sorted(os.listdir(dir_1)):
            dir_2 = os.path.join(dir_1, recodings)
            if not os.path.isdir(dir_2):
                continue
            for cameras in sorted(os.listdir(dir_2)):
                dir_3 = os.path.join(dir_2, cameras)
                if not os.path.isdir(dir_3):
                    continue
                for root, _, fnames in sorted(os.walk(dir_3)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        label = [int(d) for d in
                                 (fname.split('.')[0]).split('_')]
                        label_norm = norm_label(label) 

                        item = (path, label_norm)
                        images.append(item)

    return images


class MyDataset(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
    """

    def __init__(self, root, extensions, loader, transform=None, label_transform=None):
        #classes, class_to_idx = find_classes(root)
        samples = make_dataset(root)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                              "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, label) where label is filename without extension.
        """
        path, label = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.label_transform is not None:
            label = self.label_transform(label)

        return sample, label


    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Label Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.label_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

'''
class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
'''
