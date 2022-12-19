#!/bin/bash

echo '------- Show In&Out_A on ColoredNoiseMNIST --------'

python -m main_visual_A\
       --alg_state_dict_dir '.\outputs\train_outs\ColoredNoiseMNIST_test_id_2_clean_label_True_label_flip_p=0.0'

echo '------- Show In&Out_A on ColoredNoiseFashion --------'

python -m main_visual_A\
       --alg_state_dict_dir '.\outputs\train_outs\ColoredNoiseFashion_test_id_2_clean_label_True_label_flip_p=0.0'

echo '------- Show In&Out_B on ColoredNoiseMNIST --------'

python -m main_visual_B\
       --unseen_test_envs 0.7\
       --alg_state_dict_dir '.\outputs\train_outs\ColoredNoiseMNIST_test_id_2_clean_label_True_label_flip_p=0.0'

echo '------- Show In&Out_B on ColoredNoiseFashion --------'

python -m main_visual_B\
       --unseen_test_envs 0.7\
       --alg_state_dict_dir '.\outputs\train_outs\ColoredNoiseFashion_test_id_2_clean_label_True_label_flip_p=0.0'

echo '------------ complete -------------'