
import torch
from torch import autograd
import torch.nn.functional as F
#################################
"""
#-------- classification related losses
also including domain generalization and causal enable losses
"""
def cls_loss_with_penalty(p_name, logits, labels):
    """
    :param args: configuration parameters
    :param logits: model output
    :param labels: classification label
    :return: environment dictionary, loss names
    """
    #-- store the losses
    loss_dict = {}
    # batch_size = logits.size()[0] # no need
    #-----------------------------------------------------
    #-- ERM: accuracy penalty must be appended
    loss_dict['erm'] = cross_entropy(logits, labels)
    # -----------------------------------------------------
    if p_name in ['sd_penalty']:
        loss_dict['sd_penalty'] = sd_penalty(logits)
    # -----------------------------------------------------
    if p_name in ['irm_penalty']:
        loss_dict['irm_penalty'] = irm_penalty(logits, labels)
    # -----------------------------------------------------
    return  loss_dict

#################################
#-- ERM
def cross_entropy(logits, targets):
    return F.cross_entropy(logits, targets.squeeze())

#################################
#-- Invariant Risk Minimization (IRM) penalty
def irm_penalty(logits, target):
    scale = torch.tensor(1.).to(target.device).requires_grad_()
    loss = cross_entropy(logits * scale, target)
    grad = autograd.grad(outputs=loss, inputs=[scale], create_graph=True)[0]
    return torch.sum(grad ** 2) # usually too small. So, it uses sum while others use mean

#################################
#-- Spectral Decoupling (SD) penalty
def sd_penalty(logits):
    return (logits **2).mean()


