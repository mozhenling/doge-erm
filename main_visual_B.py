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

def change_env_name(envs):
    left = ['+' + str(round((1 - e) * 100)) + '%' for e in envs if e < 0.5]
    middel = ['±' + str(round((e) * 100)) + '%' for e in envs if e == 0.5]
    right = ['-' + str(round((e) * 100)) + '%' for e in envs if e > 0.5]
    return left + middel + right

if __name__ =="__main__":
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser(
        description="Visualization_B")
    parser.add_argument("--alg_state_dict_dir",
                        type=str,
                        default=r'.\outputs\train_outs\ColoredNoiseMNIST_test_id_2_clean_label_True_label_flip_p=0.0')
    parser.add_argument('--unseen_test_envs', nargs='+', type=float, default=[0.3, 0.7])
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    ####################################################################################################################
    algorithm_dict = torch.load(os.path.join(args.alg_state_dict_dir, 'model.pkl'))

    ENVIRONMENTS_unseen = change_env_name(args.unseen_test_envs)
    # args_saved = algorithm_dict['args']
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

    # an instance having .__getitem__ method, meaning datasets[index] = self.datasets[index]
    datasets_unseen = vars(datasets_now)[args.dataset](args.data_dir, args.test_envs, args.is_clean_label, hparams,
                                                args.label_flip_p, is_visual=True, environments =args.unseen_test_envs)

    datasets = vars(datasets_now)[args.dataset](args.data_dir, args.test_envs, args.is_clean_label, hparams,
                                                       args.label_flip_p, is_visual=True)
    #--- original traing/test environments

    plt.close('all')
    acc_dict = {}

    env_name_saved = datasets.ENVIRONMENTS[args.test_envs[0]] # there should be only one saved test env.
    fig_save_dir = os.path.join(r'./outputs/InOutComp_B',
                                args.dataset + '_test_env_saved' + env_name_saved + '_clean_label_' + str(
                                    args.is_clean_label) + '_label_flip_p=' + str(
                                    args.label_flip_p))
    os.makedirs(fig_save_dir, exist_ok=True)

    #
    batch_data = datasets[args.test_envs[0]][:args.batch_size][0]
    batch_label = datasets[args.test_envs[0]][:args.batch_size][1]
    batch_label = batch_label.to(args.device)
    batch_data = batch_data.to(args.device)
    a_batch = [(batch_data, batch_label)]
    test_acc=optimization.accuracy(network=algorithm, loader=a_batch,
                          weights=None, device=args.device, args=args, pred_by='alpha')
    # --- test
    fig_save_path_temp = os.path.join(fig_save_dir, 'input_env_saved_' + env_name_saved + '.jpeg')
    data_visual.show_sample(batch_data.cpu(), dpi=300, subfigsize=(7, 7),
                            fig_save_path=fig_save_path_temp, fig_format='jpeg')
    plt.close('all')
    # --- reconstruct
    pred_by_all = algorithm.predict(x=batch_data, pred_by='all')
    fig_save_path_temp = os.path.join(fig_save_dir, 'recon_env_saved_' + env_name_saved + '_acc=' + str(
        round(test_acc * 100, 1)) + '.jpeg')
    data_visual.show_sample(pred_by_all.cpu(), dpi=300, subfigsize=(7, 7),
                            fig_save_path=fig_save_path_temp, fig_format='jpeg')
    plt.close('all')

    print('#'*5, '\n')
    print('test_env_saved: ')
    print('\t', env_name_saved,'\n')
    print('\t', 'acc = ', round(test_acc * 100, 1), '\n')

    for unseen_id, unseen_name in enumerate(ENVIRONMENTS_unseen):

        batch_data = datasets_unseen[unseen_id][:args.batch_size][0]
        batch_label = datasets_unseen[unseen_id][:args.batch_size][1]
        batch_label = batch_label.to(args.device)
        batch_data = batch_data.to(args.device)
        a_batch = [(batch_data, batch_label)]

        acc_dict[unseen_name]=optimization.accuracy(network=algorithm, loader=a_batch,
                              weights=None, device=args.device, args=args, pred_by='alpha')

        print('test_env_unseen: ')
        print('\t', unseen_name)
        print('\t', 'acc = ', round(acc_dict[unseen_name] * 100, 1), '\n')

        #--- inputs
        # --- test
        fig_save_path_temp = os.path.join(fig_save_dir, 'input_env_unseen_' + unseen_name + '.jpeg')
        data_visual.show_sample(batch_data.cpu(), dpi=300, subfigsize=(7, 7),
                                fig_save_path=fig_save_path_temp, fig_format='jpeg')
        plt.close('all')

        #--- reconstruct
        pred_by_all = algorithm.predict(x=batch_data, pred_by='all')
        fig_save_path_temp = os.path.join(fig_save_dir, 'recon_env_unseen_' + unseen_name +'_acc='+ str(round(acc_dict[unseen_name]*100, 1)) + '.jpeg')
        data_visual.show_sample(pred_by_all.cpu(), dpi=300, subfigsize=(7, 7),
                    fig_save_path=fig_save_path_temp, fig_format='jpeg')
        plt.close('all')


