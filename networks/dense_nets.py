import torch
import torch.nn as nn

class Dense_VAE_CMNIST(nn.Module):
    """
    VAE dense networks for CMNIST dataset
    """
    ####################################################################################################################
    # ----------------------------------- Initialization
    ####################################################################################################################
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(Dense_VAE_CMNIST, self).__init__()
        # ---------------------------------------------------------------
        # self.params_dict = {}
        # input_shape =1568 for CMNIST data with two color channels
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        # dimensions of the latent features
        self.latent_all_dim = self.input_dim // 4
        # dimensions of the base features
        self.base_dim = self.input_dim // 4
        # 4 * unobcfd_onehot_dim
        self.unobcfd_dim = None
        self.latent_ratio = {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}
        # num of classification labels
        self.num_classes = num_classes
        # num of environments
        self.num_envs = num_domains
        # unobserved confounder
        self.unobcfd_onehot_dim = num_domains + num_classes
        # reconstruction domain id, may be changed later
        self.recon_domain_id = 0

        self.alpha_dim = int(self.latent_all_dim * self.latent_ratio['alpha'])
        self.beta_dim = int(self.latent_all_dim * self.latent_ratio['beta'])
        self.gamma_dim = self.latent_all_dim - self.alpha_dim - self.beta_dim
        # -------------------- subnetworks
        self.net_dict = {}
        # -- encodes x to base features
        self.net_dict['net_x_to_base']         = self.net_x_to_base
        # -- encodes base features to latent codes
        self.net_dict['net_base_to_latent']    = self.net_base_to_latent
        # -- encodes y labels to unobserved confounder
        self.net_dict['net_onehot_to_unobcfd']    = self.net_onehot_to_unobcfd
        # -- encodes unobserved confounder to the latent prior
        self.net_dict['net_unobcfd_to_all_prior']  = self.net_unobcfd_to_all_prior
        # -- decodes latent re-parameterized variables to x
        self.net_dict['net_latent_all_to_x']   = self.net_latent_all_to_x
        # -- decodes re-parameterized alpha to y
        self.net_dict['net_sublatent_to_y']    = self.net_sublatent_to_y
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
    ####################################################################################################################
    # ----------------------------------- Encoder
    ####################################################################################################################

    def net_x_to_base(self, x_dim=None, base_dim=None):
        # input_shape =1568 for CMNIST data with two color channels
        x_dim = x_dim if x_dim is not None else self.input_dim

        # dimensions of the base features
        # self.base_dim = self.input_dim // 4
        base_dim = base_dim if base_dim is not None else self.base_dim
        return nn.Sequential(
            Flatten(),
            self.Fc_bn_ReLU(x_dim, x_dim//2),
            self.Fc_bn_ReLU(x_dim//2, base_dim),
        )

    def net_base_to_latent(self, latent_dim, base_dim=None):
        # dimensions of the base features
        # self.base_dim = self.input_dim // 4
        base_dim = base_dim if base_dim is not None else self.base_dim

        # dimensions of the latent features
        # self.latent_all_dim = self.input_dim // 4
        latent_dim = latent_dim if latent_dim is not None else self.latent_all_dim

        return nn.Sequential(
            nn.Linear(base_dim, latent_dim)
        )

    ####################################################################################################################
    # ----------------------------------- Latent Prior
    ####################################################################################################################

    def net_onehot_to_unobcfd(self, unobcfd_onehot_dim=None, unobcfd_dim = None):
        unobcfd_onehot_dim = unobcfd_onehot_dim if unobcfd_onehot_dim is not None else self.unobcfd_onehot_dim
        # -- encodes unobserved confounder (from all environment)
        unobcfd_dim = unobcfd_dim if unobcfd_dim is not None else int(unobcfd_onehot_dim * 4)
        self.unobcfd_dim = unobcfd_dim
        return nn.Sequential(
            self.Fc_bn_ReLU(unobcfd_onehot_dim, int(unobcfd_onehot_dim * 2)),
            self.Fc_bn_ReLU(int(unobcfd_onehot_dim * 2), unobcfd_dim)
        )

    def net_unobcfd_to_all_prior(self,  unobcfd_dim = None, latent_dim=None):
        # -- encoder unobserved confounder to latent mean and logvar
        unobcfd_dim = unobcfd_dim if unobcfd_dim is not None else self.unobcfd_dim
        # dimensions of the latent features
        # self.latent_all_dim = self.input_dim // 4
        latent_dim = latent_dim if latent_dim is not None else self.latent_all_dim
        return nn.Sequential(
            nn.Linear(unobcfd_dim, latent_dim))

    ####################################################################################################################
    # ----------------------------------- Decoder & Predictor
    ####################################################################################################################

    def net_latent_all_to_x(self, latent_all_dim=None,  x_dim=None):
        latent_all_dim = latent_all_dim if latent_all_dim is not None else self.latent_all_dim
        # input_shape =1568 for CMNIST data with two color channels
        x_dim = x_dim if x_dim is not None else self.input_dim
        return nn.Sequential(
            self.Fc_bn_ReLU(latent_all_dim, x_dim//4),
            self.Fc_bn_ReLU(x_dim//4, x_dim//2),
            nn.Linear(x_dim//2, x_dim),
            UnFlatten(type='cmnist'),
            nn.Sigmoid()
        )

    def net_sublatent_to_y(self, sublatent_dim, num_classes=None):
        num_classes = num_classes if num_classes is not None else self.num_classes
        return nn.Sequential(
            self.Fc_bn_ReLU(sublatent_dim, sublatent_dim // 2),
            self.Fc_bn_ReLU(sublatent_dim // 2, sublatent_dim // 4),
            nn.Linear(sublatent_dim // 4, num_classes)
        )
    def Fc_bn_ReLU(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU())

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.device)
        return eps.mul(std).add_(mu)

def Featurizer(input_shape,num_classes, num_domains, hparams, args):
    return Dense_Featurizer_CMNIST(input_shape,num_classes, num_domains, hparams, args)

class Dense_Featurizer_CMNIST(nn.Module):
    def __init__(self, input_shape,num_classes, num_domains,  hparams, args):
        super(Dense_Featurizer_CMNIST, self).__init__()
        self.model = torch.nn.ModuleDict()
        self.init_model = Dense_VAE_CMNIST(input_shape, num_classes, num_domains, hparams, args)
        self.input_dim = input_shape[0] * input_shape[1] * input_shape[2]
        self.base_dim = self.init_model.base_dim
        self.alpha_dim = int(self.init_model.latent_all_dim * self.init_model.latent_ratio['alpha'])
        self.n_outputs = self.alpha_dim
        self.model['net_x_to_base'] = self.init_model.net_dict['net_x_to_base']()
        self.model['net_mean_alpha'] = self.init_model.net_dict['net_base_to_latent'](latent_dim = self.alpha_dim)
        self.model['net_logvar_alpha'] = self.init_model.net_dict['net_base_to_latent'](latent_dim = self.alpha_dim)

    def forward(self, x):
        x_base = self.model['net_x_to_base'](x)
        latent_alpha = self.model['net_mean_alpha'](x_base)

        return latent_alpha

    # def subalg_env_logits(self, env_batch_x, doyojo):
    #     # -- x to x_base
    #     x_base = self.model['net_x_to_base'](env_batch_x)
    #     doyojo_x_base = doyojo.model['net_x_to_base'](env_batch_x)
    #     x_base_avg = (x_base + doyojo_x_base.clone().detach()) / 2
    #     # -- x_base to latent alpha
    #     mean_alpha=self.model['net_mean_alpha'](x_base_avg)
    #     logvar_alpha=self.model['net_logvar_alpha'](x_base_avg)
    #     # -- reparam.
    #     latent_alpha=doyojo.reparametrize(mean_alpha, logvar_alpha)
    #
    #     # -- deep copy:
    #     doyojo.x_base_list.append((doyojo_x_base + x_base.clone().detach()) / 2)
    #     doyojo.mean_alpha_list.append(mean_alpha.clone().detach())
    #     doyojo.logvar_alpha_list.append(logvar_alpha.clone().detach())
    #     doyojo.latent_alpha_list.append(latent_alpha.clone().detach())
    #
    #     return self.model['net_alpha_to_y'](latent_alpha)




def Classifier(in_features, input_shape, num_classes, num_domains, hparams, args):
    return Dense_Classifier_CMNIST(in_features, input_shape,num_classes, num_domains, hparams, args)

class Dense_Classifier_CMNIST(nn.Module):
    def __init__(self,in_features, input_shape,num_classes, num_domains, hparams, args):
        super(Dense_Classifier_CMNIST, self).__init__()
        init_model = Dense_VAE_CMNIST(input_shape,num_classes, num_domains,hparams, args)
        self.model = torch.nn.ModuleDict()
        self.model['net_alpha_to_y'] = init_model.net_dict['net_sublatent_to_y'](sublatent_dim = in_features)
    def forward(self, x):
        return self.model['net_alpha_to_y'](x)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, type = '3d'):
        super(UnFlatten, self).__init__()
        self.type = type
    def forward(self, input):
        if self.type == 'cmnist':
            return input.view(input.size(0), 2, 28, 28)
        elif self.type == '2d':
            return input.view(input.size(0), input.size(1), 1, 1)
