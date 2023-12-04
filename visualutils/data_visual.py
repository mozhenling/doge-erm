
import torch
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from datautils import bed_datasets as datasets_now, bed_dataloaders as dataloader
from params.seedutils import seed_everything
def show_sample(data, label = None, nrows=5, ncols=5,  dpi=300, subfigsize=(7, 7),
                         fig_save_path= None, fig_format = 'jpeg'):


    fig, axs = plt.subplots(nrows, ncols, figsize=subfigsize, dpi=dpi) #,
    # axs = axs.ravel()
    shape = data.size()
    # (M, N, 3): an image with RGB values (0-1 float or 0-255 int) <----------------------------------- Check reshape in the model
    data_3 = torch.cat((data, torch.zeros(data.size()[0], 1, shape[2], shape[3])), dim=1)
    count = 0
    # title = []
    for i in range(nrows):
        for j in range(ncols):
            # (3, 28, 28) to (28, 28, 3)
            true_x_img = transforms.ToPILImage()(data_3[count].squeeze())
            axs[i][j].axis('off')
            axs[i][j].imshow(true_x_img)
            if label is not None:
                axs[i][j].set_title(str(label[count].numpy()))
            count +=1
    if fig_save_path is not None:
        plt.savefig(fig_save_path, format=fig_format)
    plt.tight_layout()
    plt.show()

if __name__ =="__main__":
    ##################################
    # ------------------------------------------------------------------------------------
    # --------------- prepare data
    seed_everything(2022)
    # ------------------------------------------------------------------------------------
    ##################################
    dataset = 'BaseCIFAR10'
    hparams = None
    data_dir = '..\datasets\CIFAR10'
    test_envs = None
    is_clean_label = True
    label_flip_p = 0.
    device = 'cpu'
    ##################################
    fig_save_dir = os.path.join(r'../outputs/data_samples', dataset+'_label_flip_p='+str(label_flip_p))
    os.makedirs(fig_save_dir, exist_ok=True)
    # -- use training domain data
    batch_size = 64 * 6
    # an instance having .__getitem__ method, meaning datasets[index] = self.datasets[index]
    datasets = vars(datasets_now)[dataset](data_dir, test_envs,
                                                   is_clean_label, hparams, label_flip_p, is_visual=True)
    env_names = datasets.ENVIRONMENTS
    plt.close('all')
    for env_name, env_data in zip(env_names, datasets):
        batch_data = env_data[:batch_size][0]
        # batch_label = env_data[:batch_size][1]
        # batch_label = batch_label.to(device)
        batch_data = batch_data.to(device)
        fig_save_path = os.path.join(fig_save_dir, 'env_'+env_name+'.jpeg')
        show_sample(batch_data, dpi=300, subfigsize=(7, 7),
                     fig_save_path=fig_save_path, fig_format='jpeg')
        plt.close('all')