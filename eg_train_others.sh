#!/bin/bash

echo '------- Run others --------'

python -m main_train\
       --data_dir=./datasets/CIFAR10/\
       --algorithm ERM\
       --steps 3000\
       --is_clean_label\
       --label_flip_p 0\
       --dataset ColoredNoiseCIFAR10\
       --test_env 2

echo '------------ complete -------------'