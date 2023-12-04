
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms

def draw_latent_space(data_dict, prefix, dpi=300, figsize=(4, 3), title ='',
                      fig_save_path= None, fig_format = 'jpeg'):
    # https://seaborn.pydata.org/generated/seaborn.scatterplot.html
    plt.figure(figsize=figsize, dpi=dpi) #
    if prefix in ['alpha', 'beta', 'gamma']:
        if prefix == 'alpha':
            marker_str = "o"
        elif prefix == 'beta':
            marker_str = "X"
        elif prefix == 'gamma':
            marker_str = "s"
        else:
            raise ValueError('prefix is not fuound')

        hue_str = 'batch_labels'

        sns.scatterplot(data = data_dict,
                        x = prefix +'_comp1',
                        y = prefix +'_comp2',
                        hue = hue_str,
                        marker=marker_str,
                        # legend = 'full',

                        palette='deep')
    else:
        hue_str = 'all_batch_labels'
        style_str = 'all_latent_labels' # 'all_latent_labels'
        sns.scatterplot(data = data_dict,
                        x = prefix +'_comp1',
                        y = prefix +'_comp2',
                        hue = hue_str,
                        style= style_str,
                        legend = False,

                        palette='deep')

    plt.title(title)
    # for lh in g._legend.legendHandles:
    #     lh.set_alpha(1)
    plt.xlabel(prefix +'_comp1')
    plt.ylabel(prefix +'_comp2')
    plt.tight_layout()
    if fig_save_path is not None:
        plt.savefig(fig_save_path, format=fig_format)
    plt.show()

def draw_recon(true_x,recon_x, sample_size=5, dpi=300, subfigsize=(7, 7), title ='True_above and Recon_below',
               shape = (28, 28, 3), fig_save_path= None, fig_format = 'jpeg'):


    # plt.figure(figsize=(6, 3))
    fig, axs = plt.subplots(2, sample_size, figsize=subfigsize, dpi=dpi) #,
    axs = axs.ravel()
    n = 0
    j = 0
    # (M, N, 3): an image with RGB values (0-1 float or 0-255 int) <----------------------------------- Check reshape in the model
    true_x_3 = torch.cat((true_x, torch.zeros(true_x.size()[0], 1, shape[0], shape[1])), dim=1)
    recon_x_3 = torch.cat( (recon_x, torch.zeros(recon_x.size()[0], 1, shape[0], shape[1])),  dim=1)

    for i in range(n, n + sample_size):
        # (3, 28, 28) to (28, 28, 3)
        true_x_img = transforms.ToPILImage()(true_x_3[i].squeeze())
        recon_x_img = transforms.ToPILImage()(recon_x_3[i].squeeze())

        axs[j].imshow(true_x_img)
        axs[j].axes.get_xaxis().set_visible(False)
        axs[j].axes.get_yaxis().set_visible(False)

        axs[j + sample_size].imshow(recon_x_img)
        axs[j + sample_size].axes.get_xaxis().set_visible(False)
        axs[j+ sample_size].axes.get_yaxis().set_visible(False)

        j += 1

    if fig_save_path is not None:
        plt.savefig(fig_save_path, format=fig_format)
    plt.show()

