"""
Domainbed Datasets
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, ImageFolder
from torchvision.transforms.functional import rotate

# from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
# from wilds.datasets.fmow_dataset import FMoWDataset
from skimage.util import random_noise
ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

DATASETS = [

    # MNIST
    'BaseMNIST',
    'ColoredNoiseMNIST',
    'NoiseColoredMNIST',
    'MNISTColoredNoise',

    'BaseFashion',
    'ColoredNoiseFashion',
    'NoiseColoredFashion',
    'FashionColoredNoise',

    'BaseCIFAR10',
    'ColoredNoiseCIFAR10',
    'NoiseColoredCIFAR10',
    'CIFAR10ColoredNoise',

]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        """
        __getitem__() is a magic method in Python, which when used in a class,
        allows its instances to use the [] (indexer) operators. Say x is an
        instance of this class, then x[i] is roughly equivalent to type(x).__getitem__(x, i).
        """
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes, is_visual, is_clean_label, label_flip_p):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        # ----------- Just consider a small portion

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)
        # --------------------- reduce computation -------------------
        original_images = original_dataset_te.data[:500] if is_visual else original_dataset_tr.data[:3000]

        original_labels = original_dataset_te.targets[:500] if is_visual else original_dataset_tr.targets[:3000]
        # -----------------------------------------------------------

        # # ---------------------  full size      ---------------------
        # original_images = torch.cat((original_dataset_tr.data,
        #                              original_dataset_te.data))
        #
        # original_labels = torch.cat((original_dataset_tr.targets,
        #                              original_dataset_te.targets))
        # # -----------------------------------------------------------
        self.datasets = []
        self.is_clean_label=is_clean_label
        if is_visual:
            for i in range(len(environments)):
                images = original_images#[i::len(environments)]
                labels = original_labels#[i::len(environments)]
                self.datasets.append(dataset_transform(images, labels, environments[i], label_flip_p))
        else:
            shuffle = torch.randperm(len(original_images))
            original_images = original_images[shuffle]
            original_labels = original_labels[shuffle]
            for i in range(len(environments)):
                images = original_images[i::len(environments)]
                labels = original_labels[i::len(environments)]
                self.datasets.append(dataset_transform(images, labels, environments[i], label_flip_p))

        self.input_shape = input_shape
        self.num_classes = num_classes

class BaseMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, is_clean_label, hparams, label_flip_p, is_visual=False, environments = (0.1, 0.2, 0.9)):
        super(BaseMNIST, self).__init__(root,  environments =environments,
                                           dataset_transform = self.color_dataset,
                                           input_shape = (2, 28, 28,), num_classes = 2,
                                        is_visual = is_visual, is_clean_label=is_clean_label, label_flip_p=label_flip_p)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.is_clean_label = is_clean_label
    def color_dataset(self, images, labels, environment, label_flip_p):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        clean_labels = labels
        # Flip label with probability 0.25 if self.is_clean_label is False
        labels = self.torch_xor_(labels, self.torch_bernoulli_(label_flip_p, len(labels)))

        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel

        x = images.float().div_(255.0)
        y = clean_labels.view(-1).long()  if self.is_clean_label else labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class ColoredNoiseMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    """
    the color variable is applied to zero out one channel of both the noise and MNIST
    the noise is environment dependent
    """
    def __init__(self, root, test_envs, is_clean_label, hparams, label_flip_p, is_visual=False, environments = (0.1, 0.2, 0.9)):
        super(ColoredNoiseMNIST, self).__init__(root,  environments =environments,
                                           dataset_transform = self.color_dataset,
                                           input_shape = (2, 28, 28,), num_classes = 2, 
                                           is_visual = is_visual, is_clean_label=is_clean_label, label_flip_p=label_flip_p)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.is_clean_label = is_clean_label

    def color_dataset(self, images, labels, environment, label_flip_p):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        clean_labels = labels
        # Flip label with probability 0.25 if self.is_clean_label is False
        labels = self.torch_xor_(labels, self.torch_bernoulli_(label_flip_p, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))

        images_noise_0 = (1 - environment) * 10 * torch.randn(images.size())
        images_noise = torch.stack([images_noise_0, images_noise_0], dim=1)
        images = torch.stack([images, images], dim=1)

        images_plus_noise = images + images_noise
        # Apply the color to the image by zeroing out the other color channel
        images_plus_noise[torch.tensor(range(len(images_plus_noise))), (
            1 - colors).long(), :, :] *= 0

        x = images_plus_noise.float().div_(255.0)
        y = clean_labels.view(-1).long()  if self.is_clean_label else labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class NoiseColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    """
    the color variable is applied to zero out one channel of only the MNIST,
    but both channels of the noise are independent of the color variable
    the noise is environment dependent
    """
    def __init__(self, root, test_envs, is_clean_label, hparams, label_flip_p, is_visual=False, environments = (0.1, 0.2, 0.9)):
        super(NoiseColoredMNIST, self).__init__(root,  environments =environments,
                                           dataset_transform = self.color_dataset,
                                           input_shape = (2, 28, 28,), num_classes = 2,
                                           is_visual = is_visual, is_clean_label=is_clean_label, label_flip_p=label_flip_p)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.is_clean_label = is_clean_label

    def color_dataset(self, images, labels, environment, label_flip_p):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        clean_labels = labels
        # Flip label with probability 0.25 if self.is_clean_label is False
        labels = self.torch_xor_(labels, self.torch_bernoulli_(label_flip_p, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))

        ##########################################################################
        # the noise is only determined by the environment
        images_noise_0 =  (1-environment)*10* torch.randn(images.size())
        ##########################################################################

        images_noise = torch.stack([images_noise_0 , images_noise_0 ], dim=1)
        images = torch.stack([images, images], dim=1)
        # images_noised = images+torch.randn(images.size()) # continue to add some noise
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = (images+images_noise).float().div_(255.0)
        y = clean_labels.view(-1).long() if self.is_clean_label else labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class MNISTColoredNoise(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    """
    the color variable is applied to zero out one channel of only the noise, 
    but the channels of the MNIST is independent of the color variable
    the noise is environment dependent
    """
    def __init__(self, root, test_envs, is_clean_label, hparams, label_flip_p, is_visual=False, environments = (0.1, 0.2, 0.9)):
        super(MNISTColoredNoise, self).__init__(root,  environments =environments,
                                           dataset_transform = self.color_dataset,
                                           input_shape = (2, 28, 28,), num_classes = 2,
                                           is_visual = is_visual, is_clean_label=is_clean_label, label_flip_p=label_flip_p)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.is_clean_label = is_clean_label

    def color_dataset(self, images, labels, environment, label_flip_p):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        clean_labels = labels
        # Flip label with probability 0.25 if self.is_clean_label is False
        labels = self.torch_xor_(labels, self.torch_bernoulli_(label_flip_p, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))

        images_noise_0 = (1 - environment) * 10 * torch.randn(images.size())
        images_noise = torch.stack([images_noise_0, images_noise_0], dim=1)
        images = torch.stack([images, images], dim=1)
        # images_noise = 9 * torch.randn(images.size()) # environment independent
        # Apply the color to the image_noise by zeroing out the other color channel
        images_noise[torch.tensor(range(len(images_noise))), (
            1 - colors).long(), :, :] *= 0

        x = (images + images_noise).float().div_(255.0)
        y = clean_labels.view(-1).long()  if self.is_clean_label else labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

##########################################################################################################
class MultipleEnvironmentFashion(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes, is_visual, is_clean_label, label_flip_p, is_subgroup=False):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        # ----------- Just consider a small portion

        data_tr = FashionMNIST(root, train=True, download=True)
        data_te = FashionMNIST(root, train=False, download=True)

        if is_visual:
            original_images = data_te.data[:500]
            original_labels = data_te.targets[:500]
        else:
            original_images = data_tr.data[:6000]
            original_labels = data_tr.targets[:6000]

        self.datasets = []
        self.is_clean_label=is_clean_label
        if is_visual:
            for i in range(len(environments)):
                images = original_images#[i::len(environments)]
                labels = original_labels#[i::len(environments)]
                self.datasets.append(dataset_transform(images, labels, environments[i], label_flip_p))
        else:
            shuffle = torch.randperm(len(original_images))
            original_images = original_images[shuffle]
            original_labels = original_labels[shuffle]
            for i in range(len(environments)):
                images = original_images[i::len(environments)]
                labels = original_labels[i::len(environments)]
                self.datasets.append(dataset_transform(images, labels, environments[i], label_flip_p))

        self.input_shape = input_shape
        self.num_classes = num_classes

class BaseFashion(MultipleEnvironmentFashion):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, is_clean_label, hparams, label_flip_p, is_visual=False, environments = (0.1, 0.2, 0.9)):
        super(BaseFashion, self).__init__(root,  environments =environments,
                                           dataset_transform = self.color_dataset,
                                           input_shape = (2, 28, 28,), num_classes = 2,
                                        is_visual = is_visual, is_clean_label=is_clean_label, label_flip_p=label_flip_p)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.is_clean_label = is_clean_label
    def color_dataset(self, images, labels, environment, label_flip_p):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        clean_labels = labels
        # Flip label with probability 0.25 if self.is_clean_label is False
        labels = self.torch_xor_(labels, self.torch_bernoulli_(label_flip_p, len(labels)))

        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel

        x = images.float().div_(255.0)
        y = clean_labels.view(-1).long()  if self.is_clean_label else labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class FashionColoredNoise(MultipleEnvironmentFashion):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    """
    the color variable is applied to zero out one channel of only the noise, 
    but the channels of the MNIST is independent of the color variable
    the noise is environment dependent
    """
    def __init__(self, root, test_envs, is_clean_label, hparams, label_flip_p, is_visual=False, environments = (0.1, 0.2, 0.9)):
        super(FashionColoredNoise, self).__init__(root,  environments =environments,
                                           dataset_transform = self.color_dataset,
                                           input_shape = (2, 28, 28,), num_classes = 2,
                                           is_visual = is_visual, is_clean_label=is_clean_label, label_flip_p=label_flip_p)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.is_clean_label = is_clean_label

    def color_dataset(self, images, labels, environment, label_flip_p):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        clean_labels = labels
        # Flip label with probability 0.25 if self.is_clean_label is False
        labels = self.torch_xor_(labels, self.torch_bernoulli_(label_flip_p, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))

        images_noise_0 = (1 - environment) * 10 * torch.randn(images.size())
        images_noise = torch.stack([images_noise_0, images_noise_0], dim=1)
        images = torch.stack([images, images], dim=1)
        # images_noise = 9 * torch.randn(images.size()) # environment independent
        # Apply the color to the image_noise by zeroing out the other color channel
        images_noise[torch.tensor(range(len(images_noise))), (
            1 - colors).long(), :, :] *= 0

        x = (images + images_noise).float().div_(255.0)
        y = clean_labels.view(-1).long()  if self.is_clean_label else labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class NoiseColoredFashion(MultipleEnvironmentFashion):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    """
    the color variable is applied to zero out one channel of only the MNIST,
    but both channels of the noise are independent of the color variable
    the noise is environment dependent
    """
    def __init__(self, root, test_envs, is_clean_label, hparams, label_flip_p, is_visual=False, environments = (0.1, 0.2, 0.9)):
        super(NoiseColoredFashion, self).__init__(root,  environments =environments,
                                           dataset_transform = self.color_dataset,
                                           input_shape = (2, 28, 28,), num_classes = 2,
                                           is_visual = is_visual, is_clean_label=is_clean_label, label_flip_p=label_flip_p)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.is_clean_label = is_clean_label

    def color_dataset(self, images, labels, environment, label_flip_p):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        clean_labels = labels
        # Flip label with probability 0.25 if self.is_clean_label is False
        labels = self.torch_xor_(labels, self.torch_bernoulli_(label_flip_p, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))

        ##########################################################################
        # the noise is only determined by the environment
        images_noise_0 =  (1-environment)*10* torch.randn(images.size())
        ##########################################################################

        images_noise = torch.stack([images_noise_0 , images_noise_0 ], dim=1)
        images = torch.stack([images, images], dim=1)
        # images_noised = images+torch.randn(images.size()) # continue to add some noise
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = (images+images_noise).float().div_(255.0)
        y = clean_labels.view(-1).long() if self.is_clean_label else labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class ColoredNoiseFashion(MultipleEnvironmentFashion):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    """
    the color variable is applied to zero out one channel of both the noise and MNIST
    the noise is environment dependent
    """
    def __init__(self, root, test_envs, is_clean_label, hparams, label_flip_p, is_visual=False, environments = (0.1, 0.2, 0.9)):
        super(ColoredNoiseFashion, self).__init__(root,  environments =environments,
                                           dataset_transform = self.color_dataset,
                                           input_shape = (2, 28, 28,), num_classes = 2,
                                           is_visual = is_visual, is_clean_label=is_clean_label, label_flip_p=label_flip_p)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.is_clean_label = is_clean_label

    def color_dataset(self, images, labels, environment, label_flip_p):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels <5).float()
        clean_labels = labels
        # Flip label with probability 0.25 if self.is_clean_label is False
        labels = self.torch_xor_(labels, self.torch_bernoulli_(label_flip_p, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))

        images_noise_0 = (1 - environment) * 10 * torch.randn(images.size())
        images_noise = torch.stack([images_noise_0, images_noise_0], dim=1)
        images = torch.stack([images, images], dim=1)

        images_plus_noise = images + images_noise
        # Apply the color to the image by zeroing out the other color channel
        images_plus_noise[torch.tensor(range(len(images_plus_noise))), (
            1 - colors).long(), :, :] *= 0

        x = images_plus_noise.float().div_(255.0)
        y = clean_labels.view(-1).long()  if self.is_clean_label else labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

##########################################################################################################
class MultipleEnvironmentCIFAR10(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes, is_visual, is_clean_label, label_flip_p):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')
        # ----------- Just consider a small portion

        if is_visual:
            data_te = CIFAR10(root, train=False, download=True)
            data = torch.tensor(data_te.data)
            targets = torch.tensor(data_te.targets)

        else:
            data_tr = CIFAR10(root, train=True, download=True)
            data = torch.tensor(data_tr.data)
            targets = torch.tensor(data_tr.targets)

        indices = (targets == 1) | (targets == 8)
        # transform rgb images to grey-scale images
        original_images = torch.mean(data[indices].float(), dim=-1)
        original_labels = targets[indices].float()

        self.datasets = []
        self.is_clean_label=is_clean_label
        if is_visual:
            for i in range(len(environments)):
                images = original_images#[i::len(environments)]
                labels = original_labels#[i::len(environments)]
                self.datasets.append(dataset_transform(images, labels, environments[i], label_flip_p))
        else:
            shuffle = torch.randperm(len(original_images))
            original_images = original_images[shuffle]
            original_labels = original_labels[shuffle]
            for i in range(len(environments)):
                images = original_images[i::len(environments)]
                labels = original_labels[i::len(environments)]
                self.datasets.append(dataset_transform(images, labels, environments[i], label_flip_p))

        self.input_shape = input_shape
        self.num_classes = num_classes

class BaseCIFAR10(MultipleEnvironmentCIFAR10):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, is_clean_label, hparams, label_flip_p, is_visual=False, environments = (0.1, 0.2, 0.9)):
        super(BaseCIFAR10, self).__init__(root,  environments =environments,
                                           dataset_transform = self.color_dataset,
                                           input_shape = (2, 28, 28,), num_classes = 2,
                                        is_visual = is_visual, is_clean_label=is_clean_label, label_flip_p=label_flip_p)

        self.input_shape = (2, 32, 32,)
        self.num_classes = 2
        self.is_clean_label = is_clean_label
    def color_dataset(self, images, labels, environment, label_flip_p):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels<5).float() # labels are 1s or 8s
        clean_labels = labels
        # Flip label with probability 0.25 if self.is_clean_label is False
        labels = self.torch_xor_(labels, self.torch_bernoulli_(label_flip_p, len(labels)))

        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel

        x = images.float().div_(255.0)
        y = clean_labels.view(-1).long()  if self.is_clean_label else labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class CIFAR10ColoredNoise(MultipleEnvironmentCIFAR10):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    """
    the color variable is applied to zero out one channel of only the noise, 
    but the channels of the MNIST is independent of the color variable
    the noise is environment dependent
    """
    def __init__(self, root, test_envs, is_clean_label, hparams, label_flip_p, is_visual=False, environments = (0.1, 0.2, 0.9)):
        super(CIFAR10ColoredNoise, self).__init__(root,  environments =environments,
                                           dataset_transform = self.color_dataset,
                                           input_shape = (2, 28, 28,), num_classes = 2,
                                           is_visual = is_visual, is_clean_label=is_clean_label, label_flip_p=label_flip_p)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.is_clean_label = is_clean_label

    def color_dataset(self, images, labels, environment, label_flip_p):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        clean_labels = labels
        # Flip label with probability 0.25 if self.is_clean_label is False
        labels = self.torch_xor_(labels, self.torch_bernoulli_(label_flip_p, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))

        images_noise_0 = (1 - environment) * 10 * torch.randn(images.size())
        images_noise = torch.stack([images_noise_0, images_noise_0], dim=1)
        images = torch.stack([images, images], dim=1)
        # images_noise = 9 * torch.randn(images.size()) # environment independent
        # Apply the color to the image_noise by zeroing out the other color channel
        images_noise[torch.tensor(range(len(images_noise))), (
            1 - colors).long(), :, :] *= 0

        x = (images + images_noise).float().div_(255.0)
        y = clean_labels.view(-1).long()  if self.is_clean_label else labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class NoiseColoredCIFAR10(MultipleEnvironmentCIFAR10):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    """
    the color variable is applied to zero out one channel of only the MNIST,
    but both channels of the noise are independent of the color variable
    the noise is environment dependent
    """
    def __init__(self, root, test_envs, is_clean_label, hparams, label_flip_p, is_visual=False, environments = (0.1, 0.2, 0.9)):
        super(NoiseColoredCIFAR10, self).__init__(root,  environments =environments,
                                           dataset_transform = self.color_dataset,
                                           input_shape = (2, 28, 28,), num_classes = 2,
                                           is_visual = is_visual, is_clean_label=is_clean_label, label_flip_p=label_flip_p)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.is_clean_label = is_clean_label

    def color_dataset(self, images, labels, environment, label_flip_p):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        clean_labels = labels
        # Flip label with probability 0.25 if self.is_clean_label is False
        labels = self.torch_xor_(labels, self.torch_bernoulli_(label_flip_p, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))

        ##########################################################################
        # the noise is only determined by the environment
        images_noise_0 =  (1-environment)*10* torch.randn(images.size())
        ##########################################################################

        images_noise = torch.stack([images_noise_0 , images_noise_0 ], dim=1)
        images = torch.stack([images, images], dim=1)
        # images_noised = images+torch.randn(images.size()) # continue to add some noise
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = (images+images_noise).float().div_(255.0)
        y = clean_labels.view(-1).long() if self.is_clean_label else labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()

class ColoredNoiseCIFAR10(MultipleEnvironmentCIFAR10):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']
    """
    the color variable is applied to zero out one channel of both the noise and MNIST
    the noise is environment dependent
    """
    def __init__(self, root, test_envs, is_clean_label, hparams, label_flip_p, is_visual=False, environments = (0.1, 0.2, 0.9)):
        super(ColoredNoiseCIFAR10, self).__init__(root,  environments =environments,
                                           dataset_transform = self.color_dataset,
                                           input_shape = (2, 28, 28,), num_classes = 2,
                                           is_visual = is_visual, is_clean_label=is_clean_label, label_flip_p=label_flip_p)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2
        self.is_clean_label = is_clean_label

    def color_dataset(self, images, labels, environment, label_flip_p):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        clean_labels = labels
        # Flip label with probability 0.25 if self.is_clean_label is False
        labels = self.torch_xor_(labels, self.torch_bernoulli_(label_flip_p, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))

        images_noise_0 = (1 - environment) * 10 * torch.randn(images.size())
        images_noise = torch.stack([images_noise_0, images_noise_0], dim=1)
        images = torch.stack([images, images], dim=1)

        images_plus_noise = images + images_noise
        # Apply the color to the image by zeroing out the other color channel
        images_plus_noise[torch.tensor(range(len(images_plus_noise))), (
            1 - colors).long(), :, :] *= 0

        x = images_plus_noise.float().div_(255.0)
        y = clean_labels.view(-1).long()  if self.is_clean_label else labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()