#!/bin/bash

echo '------- Delete Incomplete on MNIST --------'

python -m main_sweep\
       --command delete_incomplete\
       --command_launcher plain\
       --n_trials 3\
       --n_hparams 20\
       --datasets BaseMNIST ColoredNoiseMNIST NoiseColoredMNIST MNISTColoredNoise\
       --is_clean_label\
       --label_flip_p 0\
       --data_dir=./datasets/MNIST/\
       --algorithms DoYoJoAlpha\
       --sub_algorithms ERM\
       --skip_model_save

echo '------- Launch  on MNIST --------'

python -m main_sweep\
       --command launch\
       --command_launcher plain\
       --n_trials 3\
       --n_hparams 20\
       --datasets BaseMNIST ColoredNoiseMNIST NoiseColoredMNIST MNISTColoredNoise\
       --is_clean_label\
       --label_flip_p 0\
       --data_dir=./datasets/MNIST/\
       --algorithms DoYoJoAlpha\
       --sub_algorithms ERM\
       --skip_model_save

echo '-------  Done on MNIST --------'

echo '------- Delete Incomplete on Fashion --------'

python -m main_sweep\
       --command delete_incomplete\
       --command_launcher plain\
       --n_trials 3\
       --n_hparams 20\
       --datasets BaseFashion ColoredNoiseFashion NoiseColoredFashion FashionColoredNoise\
       --is_clean_label\
       --label_flip_p 0\
       --data_dir=./datasets/Fashion/\
       --algorithms ERM\
       --skip_model_save

echo '------- Launch on Fashion --------'

python -m main_sweep\
       --command launch\
       --command_launcher plain\
       --n_trials 3\
       --n_hparams 20\
       --datasets BaseFashion ColoredNoiseFashion NoiseColoredFashion FashionColoredNoise\
       --is_clean_label\
       --label_flip_p 0\
       --data_dir=./datasets/Fashion/\
       --algorithms ERM\
       --skip_model_save
echo '-------  Done On Fashion --------'

echo '------- Delete Incomplete on CIFAR10 --------'

python -m main_sweep\
       --command delete_incomplete\
       --command_launcher plain\
       --n_trials 3\
       --n_hparams 20\
       --datasets BaseCIFAR10 ColoredNoiseCIFAR10 NoiseColoredCIFAR10 CIFAR10ColoredNoise\
       --is_clean_label\
       --label_flip_p 0\
       --data_dir=./datasets/CIFAR10/\
       --algorithms ERM\
       --skip_model_save

echo '------- Launch on CIFAR10 --------'

python -m main_sweep\
       --command launch\
       --command_launcher plain\
       --n_trials 3\
       --n_hparams 20\
       --datasets BaseCIFAR10 ColoredNoiseCIFAR10 NoiseColoredCIFAR10 CIFAR10ColoredNoise\
       --is_clean_label\
       --label_flip_p 0\
       --data_dir=./datasets/CIFAR10/\
       --algorithms ERM\
       --skip_model_save
echo '-------  Done on CIFAR10 --------'