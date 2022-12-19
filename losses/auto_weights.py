"""
Automatically weight different losses
See ref. for more:
https://github.com/median-research-group/LibMTL
https://github.com/AvivNavon/nash-mtl
"""
import torch
import torch.nn as nn

class AutoWeightedLoss():
    def __init__(self, auto_loss_type, awl_anneal_count_thresh=10,
                 awl_anneal_keys= ('recon','kld', 'erm_alpha', 'erm_beta', 'erm_gamma')):
        self.iter_count = 0  # number of updates / iteration
        self.anneal_count = 0
        self.anneal_count_thresh = awl_anneal_count_thresh
        self.anneal_keys = awl_anneal_keys # which losses run first
        self.auto_loss_type = auto_loss_type if auto_loss_type is not None else 'coef_var_loss'

    def get_auto_loss(self, loss_dict):
        self.anneal_count += 1
        if self.anneal_count <=self.anneal_count_thresh:
            return torch.stack([loss_dict[key] for key in self.anneal_keys], dim=0).mean() # global mean=batch mean

        if self.auto_loss_type in ['cv','cvl','coef_var_loss']:
            if self.iter_count <1:
                return self.coef_var_loss_init(loss_dict)
            else:
                return self.coef_var_loss( loss_dict)
        else:
            raise NotImplementedError

    def coef_var_loss_init(self, loss_dict):
        self.loss_num = len(loss_dict)

        if self.auto_loss_type in ['cv', 'cvl', 'coef_var_loss']:
            # ---- initialization for coef_var_loss
            self.losses_mean_last = loss_dict.values()
            self.losses_ratio_mean_last  =  [torch.tensor([1.]).cuda() for _ in range(self.loss_num)]
            self.losses_ratio_std_last = [torch.tensor([0.]).cuda() for _ in range(self.loss_num)]
            self.iter_count += 1
        return  torch.stack(list(loss_dict.values()), dim=0).mean()

    def coef_var_loss(self, loss_dict):
        """
        [1]R. Groenendijk, S. Karaoglu, T. Gevers, and T. Mensink, “Multi-Loss Weighting with Coefficient of Variations,”
         arXiv:2009.01717 [cs], Nov. 2020, Accessed: Feb. 28, 2022. [Online]. Available: http://arxiv.org/abs/2009.01717
        For step t, now means t, last means t-1

        losses_now: list of components of the objective function
        losses_mean_last: mean list of components of the objective function

        """
        # -- get values at last time
        losses_now = loss_dict.values()
        losses_mean_last = self.losses_mean_last # single value
        losses_ratio_mean_last = self.losses_ratio_mean_last # single value
        losses_ratio_std_last = self.losses_ratio_std_last # single value
        step = self.iter_count
        # --

        small_num = 1e-16
        # ------------- Welford’s algorithm
        # -- running mean of loss
        losses_ratio_now = [loss_now / (loss_mean_last.clone().detach() + small_num)
                            for loss_now, loss_mean_last in zip(losses_now, losses_mean_last)]

        # -- update values
        with torch.no_grad():
            losses_mean_now = [(1 - 1 / step) * loss_mean_last + loss_now / step
                               for loss_now, loss_mean_last in zip(losses_now, losses_mean_last)]

            losses_ratio_mean_now = [(1 - 1 / step) * loss_ratio_mean_last + loss_ratio_now / step
                                     for loss_ratio_mean_last, loss_ratio_now in zip(
                    losses_ratio_mean_last, losses_ratio_now)]

            losses_ratio_std_now = [torch.sqrt((1 - 1 / step) * loss_ratio_std_last ** 2 +
                                               (loss_ratio_now - loss_ratio_mean_last) * (
                                                           loss_ratio_now - loss_ratio_mean_now) / step)
                                    for loss_ratio_std_last, loss_ratio_now, loss_ratio_mean_last, loss_ratio_mean_now
                                    in zip(
                    losses_ratio_std_last, losses_ratio_now, losses_ratio_mean_last, losses_ratio_mean_now
                )]
            # -- coefficient of variation
            cofv_now = [loss_ratio_std_now / (loss_ratio_mean_now + small_num)
                        for loss_ratio_std_now, loss_ratio_mean_now in zip(
                    losses_ratio_std_now, losses_ratio_mean_now
                )]
            # normalization constant
            c_normal = sum(cofv_now) # sum of the iterable

        # -- final objective function
        obj_final = sum([w.clone().detach() * loss_ratio / (c_normal.clone().detach() + small_num)
                         for w, loss_ratio in zip(cofv_now, losses_ratio_now)]) / len(cofv_now)
        # -- update values for next time
        # -- get values at last time
        self.losses_mean_last = losses_mean_now
        self.losses_ratio_mean_last = losses_ratio_mean_now
        self.losses_ratio_std_last = losses_ratio_std_now
        self.iter_count +=1
        # --
        return obj_final

class Uncertainty_Positive(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    Ref. https://github.com/Mikoto10032/AutomaticWeightedLoss
    """
    def __init__(self, num=2):
        super(Uncertainty_Positive, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

