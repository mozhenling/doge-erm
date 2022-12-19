

import torch
import copy
import numpy as np
from algorithms.classes.Algorithm import Algorithm
from algorithms.optimization import get_optimizer, get_scheduler
from networks.net_selector import get_nets
from losses.vae_losses import  vae_bce_kld_losses
from losses.auto_weights import AutoWeightedLoss
from algorithms.subalg_selector import get_subalgorithm_class
########################################################################################################################
# -------------------------------------- Do Your Job based VAE
########################################################################################################################
class DoYoJoAlpha(Algorithm):
    """
    -base encoder for all
    -latent variables:
        -alpha: causal variable, predicting across domains
    -classifier for alpha
    -shared decoder for all
    """
    ####################################################################################################################
    # ----------------------------------- Initialization
    ####################################################################################################################
    def __init__(self,
                 input_shape =(2, 28, 28),   # input channel of raw data, greyscale = 1
                 num_classes =2,     # the number of labels
                 num_domains =2,     # number of environments
                 Init_Model =None,    # the sub-networks of VAE for specific data
                 args=None,          # other arguments
                 hparams = None  # hyper-params. to be tuned
                 ):
        super(DoYoJoAlpha, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        self.register_buffer('update_count', torch.tensor([0])) # See IRM for the purpose
        # --------------------------------------------------------------------------------------------------------------
        self.is_auto_loss =  hparams['is_auto_loss']
        if Init_Model is None :
            init_model = get_nets(input_shape, num_classes, num_domains, hparams, args)
        else:
            init_model = Init_Model(input_shape, num_classes, num_domains, hparams, args)
        # -------------------- uncertain params.
        try:
            self.latent_ratio = args.self.latent_ratio  # if not define, use the default
            self.latent_all_dim = args.latent_all_dim
            self.base_dim = args.base_dim
        except:
            self.latent_ratio = init_model.latent_ratio
            self.latent_all_dim = init_model.latent_all_dim
            self.base_dim = init_model.base_dim

        # --------------------------------------------------------------------------------------------------------------
        # -------------------- other dimensions/variables
        self.input_dim = init_model.input_dim
        self.num_classes = init_model.num_classes
        self.num_envs = init_model.num_envs
        self.unobcfd_onehot_dim = init_model.unobcfd_onehot_dim  # unobserved confounder
        self.recon_domain_id = init_model.recon_domain_id  # reconstruction domain id, may be changed later
        # -------------------- dimensions of latent variables
        self.alpha_dim = int(self.latent_all_dim * self.latent_ratio['alpha'])
        # -----------------------------------------------------------
        self.vae_losses = vae_bce_kld_losses
        # -----------------------------------------------------------
        # -- set sub-algorithm
        self.subalg_name = hparams['subalg_name']
        self.subalg_args = copy.deepcopy(self.args)
        self.subalg_args.algorithm = self.subalg_name
        self.subalg_hparams = copy.deepcopy(self.hparams)
        self.subalg = get_subalgorithm_class(self.subalg_name)(input_shape, num_classes, num_domains,
                                                               self.subalg_hparams, self.subalg_args)
        # --------------------------------------------------------------------------------------------------------------
        # ------------- subnetworks
        # for registering the modules
        self.model = torch.nn.ModuleDict()
        # -- encodes x to base features
        self.x_base = None
        self.model['net_x_to_base']  = init_model.net_dict['net_x_to_base']()
        # -- encodes base features to latent code alpha
        self.mean_alpha   = None
        self.logvar_alpha = None
        self.latent_alpha = None
        self.model['net_mean_alpha']    = init_model.net_dict['net_base_to_latent'](latent_dim = self.alpha_dim)
        self.model['net_logvar_alpha']  = init_model.net_dict['net_base_to_latent'](latent_dim = self.alpha_dim)

        # -- concatenation of all latent codes
        self.latent_all        = None
        self.mean_latent_all   = None
        self.logvar_latent_all = None

        # -- decodes latent re-parameterized variables to x
        self.model['net_latent_all_to_x']  = init_model.net_dict['net_latent_all_to_x'](latent_all_dim = self.alpha_dim )
        # -- decodes re-parameterized alpha to y
        alpha_dim_temp = self.alpha_dim * 2 if self.subalg_name in ['MTL'] else self.alpha_dim
        self.model['net_alpha_to_y']  = init_model.net_dict['net_sublatent_to_y'](sublatent_dim = alpha_dim_temp)
        # --------------------------------------------------------------------------------------------------------------
        # -- alpha related
        self.x_base_list = []
        self.mean_alpha_list = []
        self.logvar_alpha_list = []
        self.latent_alpha_list = []
        self.alpha_net_keys = ['net_x_to_base', 'net_mean_alpha', 'net_logvar_alpha', 'net_alpha_to_y']

        # --------------------------------------------------------------------------------------------------------------
        # -- set optimization
        if self.is_auto_loss:
            self.autoweightedloss = AutoWeightedLoss(auto_loss_type=hparams['auto_loss_type'],
                                                     awl_anneal_count_thresh = hparams['awl_anneal_count_thresh'])
        # -----------------------------------------------------------
        # or self.parameters()
        self.optimizer = get_optimizer(params=[{'params':net.parameters()} for net in self.model.values()],
                                       hparams=self.hparams, args=self.args)
        self.scheduler = get_scheduler(optimizer=self.optimizer, args=self.args)

        
    def update(self, minibatches, unlabeled=None):
        # -- minibatches (train_env_data_loader1, train_env_data_loader2, ...), one batch (data, label) after another
        # -- opt : optimizer, sch: learning schedule
        self.x_base_list = []
        self.mean_alpha_list = []
        self.logvar_alpha_list = []
        self.latent_alpha_list = []
        # -- select one for generation of data
        self.recon_domain_id = int(np.random.choice(self.num_envs, 1))  # generate one env. randomly

        # -- process alpha by the domain generalization algorithm
        # including the above lists
        dict_all_losses = self.subalg.update( minibatches, doyojo=self)
        self.update_latent_alpha()

        # -- concatenate all latent variables
        self.mean_latent_all   = self.mean_alpha_list[self.recon_domain_id]
        self.logvar_latent_all = self.logvar_alpha_list[self.recon_domain_id]
        # #####################################################################################【change】
        # -- the two are the same
        self.latent_all = self.latent_alpha_list[self.recon_domain_id]
        # #####################################################################################
        # -- latent_all (after mean, logvar, reparam) to x
        x_recon = self.model['net_latent_all_to_x'](self.latent_all)

        # -- loss components of the ELBO
        dict_all_losses['recon'], dict_all_losses['kld'] = self.vae_losses( #vae_bce_kld_with_prior_loss
            recon_x      = x_recon,
            x            = minibatches[self.recon_domain_id][0],
            mu           = self.mean_latent_all,
            logvar       = self.logvar_latent_all
         )

        # assemble the losses with hyper-params.
        if self.is_auto_loss:
            final_loss = self.autoweightedloss.get_auto_loss(dict_all_losses)
        else:
            # try: # debug
            final_loss = torch.mean(torch.stack([self.hparams[key+'_weight'] * dict_all_losses[key]
                                                            for key in dict_all_losses.keys()], dim=0) )
            # except:
            #     pass
        # -- update the model
        self.optimizer.zero_grad()
        final_loss.backward()
        self.optimizer.step()
        # -- update again by subalg
        if self.subalg_name in ['ANDMask', 'SANDMask','Fish', 'MLDG']:
            self.optimizer.zero_grad()
            self.subalg.update_alpha_nets(doyojo=self)
            self.optimizer.step()

        # -- if learning schedule is not none, adjust learning rate based on sch
        if self.args.scheduler:
            self.scheduler.step()
        self.update_count += 1

        return_dict = {}
        for i, (key, value) in enumerate(dict_all_losses.items()):
            return_dict['s'+str(i)+'_'+key] = value.item()
        return_dict['s'+str(len(dict_all_losses)) + '_' + 'wloss'] = final_loss.item()
        return return_dict

    def predict(self, x, pred_by = 'alpha', do_latent_dict = None):
        self.x_base = self.model['net_x_to_base'](x)
        if pred_by in ['alpha','beta', 'gamma']:
            pred = self.get_y_by_alpha(self.x_base)
        elif pred_by in ['all']:
            pred = self.get_x_by_latent_all(self.x_base, do_latent_dict)
        else:
            raise ValueError('The function name is NOT found')

        return pred

    ####################################################################################################################
    # ----------------------------------- Predictions after training
    ####################################################################################################################
    def get_y_by_alpha(self, x_base):
        self.mean_alpha   = self.model['net_mean_alpha'](x_base)
        self.logvar_alpha = self.model['net_logvar_alpha'](x_base)
        self.latent_alpha = self.reparametrize(self.mean_alpha, self.logvar_alpha)
        if self.subalg_name in ['MTL']:
            latent_alpha = torch.cat((self.latent_alpha, self.latent_alpha.mean(0).view(1, -1).repeat(len(self.latent_alpha), 1)), 1)
        else:
            latent_alpha = self.latent_alpha
        logits_by_alpha   = self.model['net_alpha_to_y'](latent_alpha)

        return logits_by_alpha

    def get_x_by_latent_all(self, x_base, do_latent_dict =None):
        """
        :param x_base: x->base
        :param do_latent: a dictionary of soft intervention, i.e., some random noise,
        :return: reconstructed x
        """
        # -- update alpha
        _ = self.get_y_by_alpha(x_base)

        # -- updata latent all
        self.mean_latent_all = self.mean_alpha
        self.logvar_latent_all = self.logvar_alpha

        # -- soft interventions
        latent_alpha = self.latent_alpha.clone().detach()
        if do_latent_dict is not None:
            for key, value in do_latent_dict.items():
                if key in ['alpha']:
                    latent_alpha = value
                else:
                    raise NotImplementedError
        # #####################################################################################
        self.latent_all = latent_alpha
        # #####################################################################################

        return self.model['net_latent_all_to_x'](self.latent_all)

    ####################################################################################################################
    # ---------------   functions of featurizer and classifer used in sub-algorithms
    ####################################################################################################################
    def subalg_net_state_dict(self, key_list=None):
        key_list_now = self.alpha_net_keys if key_list is None else key_list
        state_dict = {}
        for net_key in key_list_now:
            state_dict[net_key] = self.model[net_key].state_dict()
        return state_dict

    def subalg_net_load_state_dict(self, state_dict, key_list=None):
        key_list_now = self.alpha_net_keys if key_list is None else key_list
        for net_key in key_list_now:
            self.model[net_key].load_state_dict(state_dict[net_key])

    def subalg_net_parameters(self, key_list=None):
        key_list_now = self.alpha_net_keys if key_list is None else key_list
        for net_key in key_list_now:
            for name, param in self.model[net_key].named_parameters():
                yield param

    def subalg_env_logits(self, env_batch_x):
        return self.subalg_env_classifier_outs(self.subalg_env_featurizer_outs(env_batch_x))

    def subalg_env_featurizer_outs(self, env_batch_x):
        # -- x to x_base
        self.x_base_list.append(self.model['net_x_to_base'](env_batch_x))
        # -- x_base to latent alpha
        self.mean_alpha_list.append(self.model['net_mean_alpha'](self.x_base_list[-1]))
        self.logvar_alpha_list.append(self.model['net_logvar_alpha'](self.x_base_list[-1]))
        # -- reparam.
        self.latent_alpha_list.append(self.reparametrize(self.mean_alpha_list[-1], self.logvar_alpha_list[-1]))
        return self.latent_alpha_list[-1]

    def subalg_env_classifier_outs(self, env_latent_alpha):
        # -- latent_alpha to y loss
        return self.model['net_alpha_to_y'](env_latent_alpha)

    def subalg_all_logits(self, all_x):
        return self.subalg_all_classifier_outs(self.subalg_all_featurizer_outs(all_x))

    def subalg_all_featurizer_outs(self, all_x):
        all_x_base = self.model['net_x_to_base'](all_x)
        all_mean_alpha = self.model['net_mean_alpha'](all_x_base)
        all_logvar_alpha = self.model['net_logvar_alpha'](all_x_base)
        all_latent_alpha = self.reparametrize(all_mean_alpha, all_logvar_alpha)

        # -- store each env value separately in the list
        self.x_base_list = [all_x_base[i * self.hparams['batch_size']:(i + 1) * self.hparams['batch_size']]
                            for i in range(self.num_envs)]
        self.mean_alpha_list = [all_mean_alpha[i * self.hparams['batch_size']:(i + 1) * self.hparams['batch_size']]
                                for i in range(self.num_envs)]
        self.logvar_alpha_list = [all_logvar_alpha[i * self.hparams['batch_size']:(i + 1) * self.hparams['batch_size']]
                                  for i in range(self.num_envs)]
        self.latent_alpha_list = [all_latent_alpha[i * self.hparams['batch_size']:(i + 1) * self.hparams['batch_size']]
                                  for i in range(self.num_envs)]
        return all_latent_alpha

    def subalg_all_classifier_outs(self, all_z):
        return self.model['net_alpha_to_y'](all_z)

    ####################################################################################################################
    # ---------------   other functions
    ####################################################################################################################

    def update_latent_alpha(self):
        # -- process latent alpha by averaging values over all envs.
        self.latent_alpha = torch.mean(torch.stack(self.latent_alpha_list, dim=0), dim=0)  # dim 0 mean
        self.mean_alpha = torch.mean(torch.stack(self.mean_alpha_list, dim=0), dim=0)  # dim 0 mean
        self.logvar_alpha = torch.mean(torch.stack(self.logvar_alpha_list, dim=0), dim=0)  # dim 0 mean

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.args.device)
        return eps.mul(std).add_(mu)
