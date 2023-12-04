#!/bin/bash

echo '------- Train VAE-Classifier on ColoredNoiseMNIST --------'

python -m main_train\
       --data_dir=./datasets/MNIST/\
       --algorithm DoYoJoAlpha\
       --sub_algorithm ERM\
       --steps 3000\
       --is_clean_label\
       --label_flip_p 0\
       --dataset ColoredNoiseMNIST\
       --test_envs 2

echo '------- Train VAE-Classifier on ColoredNoiseFashion --------'

python -m main_train\
       --data_dir=./datasets/Fashion/\
       --algorithm DoYoJoAlpha\
       --sub_algorithm ERM\
       --steps 5000\
       --is_clean_label\
       --label_flip_p 0\
       --dataset ColoredNoiseFashion\
       --test_envs 2

echo '------------ complete -------------'