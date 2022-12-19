"""
Visualization of DoYoJoVAE
"""
import os
import torch

import numpy as np

import random
import matplotlib.pyplot as plt
from sklearn import manifold
import argparse
# from params.train_params import get_args
from algorithms import alg_selector, optimization

from visualutils import data_visual
from datautils import bed_datasets as datasets_now

if __name__ =="__main__":
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser(
        description="Visualization_A")
    parser.add_argument("--alg_state_dict_dir",
                        type=str,
                        default=r'.\outputs\train_outs\ColoredNoiseMNIST_test_id_2_clean_label_True_label_flip_p=0.0')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    algorithm_dict = torch.load(os.path.join(args.alg_state_dict_dir, 'model.pkl'))

    # args = get_args()
    vars(args).update(algorithm_dict['args'])
    # for k, v in sorted(args_saved.items()):
    #     vars(args)[k] = v

    hparams = algorithm_dict['model_hparams']

    # ------------------------------------------------------------------------------------
    # --------------- prepare model

    algorithm_class = alg_selector.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(input_shape=algorithm_dict['model_input_shape'],
                                num_classes=algorithm_dict['model_num_classes'],
                                num_domains=algorithm_dict['model_num_domains'],
                                hparams=hparams,
                                args=args)

    algorithm.load_state_dict(state_dict=algorithm_dict["model_dict"])
    algorithm.to(args.device)

    # ------------------------------------------------------------------------------------
    # --------------- seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    # ------------------------------------------------------------------------------------

    datasets = vars(datasets_now)[args.dataset](args.data_dir, args.test_envs,
                                                   args.is_clean_label, hparams, args.label_flip_p, is_visual=True)
    env_names = datasets.ENVIRONMENTS
    plt.close('all')
    acc_dict = {}
    for test_id in args.test_envs:
        env_name = env_names[test_id]
        fig_save_dir = os.path.join(r'./outputs/InOutComp_A',
                                    args.dataset + '_test_env_' + env_name + '_clean_label_' + str(
                                        args.is_clean_label) + '_label_flip_p=' + str(
                                        args.label_flip_p))
        os.makedirs(fig_save_dir, exist_ok=True)

        batch_data = datasets[test_id][:args.batch_size][0]
        batch_label = datasets[test_id][:args.batch_size][1]
        batch_label = batch_label.to(args.device)
        batch_data = batch_data.to(args.device)
        a_batch = [(batch_data, batch_label)]
        acc_dict[env_name]=optimization.accuracy(network=algorithm, loader=a_batch,
                              weights=None, device=args.device, args=args, pred_by='alpha')

        for train_id in range(len(datasets)):
            if train_id != test_id:
                train_batch_data = datasets[train_id][args.batch_size:][0]
                train_env_name = env_names[train_id]
                # --- test
                fig_save_path_temp = os.path.join(fig_save_dir, 'train_env_' + train_env_name + '.jpeg')
                data_visual.show_sample(train_batch_data.cpu(), dpi=300, subfigsize=(7, 7),
                                        fig_save_path=fig_save_path_temp, fig_format='jpeg')
                plt.close('all')
        #--- test
        fig_save_path_temp = os.path.join(fig_save_dir, 'test_env_' + env_name + '.jpeg')
        data_visual.show_sample(batch_data.cpu(), dpi=300, subfigsize=(7, 7),
                                fig_save_path=fig_save_path_temp, fig_format='jpeg')
        plt.close('all')
        #--- reconstruct
        pred_by_all = algorithm.predict(x=batch_data, pred_by='all')
        fig_save_path_temp = os.path.join(fig_save_dir, 'recon_env_' + env_name +'_acc='+ str(round(acc_dict[env_name], 3)) + '.jpeg')
        data_visual.show_sample(pred_by_all.cpu(), dpi=300, subfigsize=(7, 7),
                    fig_save_path=fig_save_path_temp, fig_format='jpeg')
        plt.close('all')


