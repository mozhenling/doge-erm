import torch
import torch.nn.functional as F

#################################
"""
#-------- VAE losses
MSE or BCE ? + KLD  
In short: Maximizing likelihood of model whose prediction 
are normal distribution (multinomial distribution) is 
equivalent to minimizing MSE (BCE)
"""
#################################

def vae_mse_kld_losses(recon_x, x, mu, logvar):
    """
    mean squared error loss + KL divergence loss
    :param recon_x: reconstructed x
    :param x: original x
    :param mu: mean
    :param logvar: log variance
    :return:
        mean squared error loss + KL divergence loss
        mean squared error loss
        KL divergence loss
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    """
    sse_loss = torch.nn.MSELoss(reduction='sum')
    MSE_loss = sse_loss(recon_x, x) / x.size()[0] # batch mean

    KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size()[0]  # batch mean

    return  MSE_loss, KLD_loss

def vae_mse_kld_with_prior_loss(recon_x, x, mu, logvar, mu_prior, logvar_prior):
    """
    mean squared error loss + KL divergence loss
    :param recon_x: reconstructed x
    :param x: original x
    :param mu: mean
    :param logvar: log variance
    :mu_prior: prior mean
    :logvar_prior: prior log variance
    :return:
        mean squared error loss + KL divergence loss
        mean squared error loss
        KL divergence loss
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    """
    sse_loss = torch.nn.MSELoss(reduction='sum')
    MSE_loss = sse_loss(recon_x, x) / x.size()[0]

    KLD_element = torch.log(logvar_prior.exp() ** 0.5 / logvar.exp() ** 0.5) + \
                  0.5 * ((mu - mu_prior).pow(2) + logvar.exp()) / logvar_prior.exp() - 0.5

    KLD = KLD_element.mul_(1).sum() / x.size()[0]

    return MSE_loss, KLD

def vae_bce_kld_losses(recon_x, x, mu, logvar):
    """
    binary cross entropy loss + KL divergence loss
    :param recon_x: reconstructed x
    :param x: original x
    :param mu: mean
    :param logvar: log variance
    :args: resizing parameters, etc.
    :return:
        binary cross entropy loss+ KL divergence loss
        mean squared error loss
        KL divergence loss

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    """
    BCE_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size()[0]    # batch mean
    KLD_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / x.size()[0]  # batch mean

    return BCE_loss, KLD_loss


def vae_bce_kld_with_prior_loss(recon_x, x, mu, logvar, mu_prior, logvar_prior):
    """
    mean squared error loss + KL divergence loss
    :param recon_x: reconstructed x
    :param x: original x
    :param mu: mean
    :param logvar: log variance
    :mu_prior: prior mean
    :logvar_prior: prior log variance
    :return:
        mean squared error loss + KL divergence loss
        mean squared error loss
        KL divergence loss
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    """
    BCE_loss = F.binary_cross_entropy(recon_x, x, reduction='sum') / x.size()[0] # average over batch

    KLD_element = torch.log(logvar_prior.exp() ** 0.5 / logvar.exp() ** 0.5) + \
                  0.5 * ((mu - mu_prior).pow(2) + logvar.exp()) / logvar_prior.exp() - 0.5

    KLD = KLD_element.mul_(1).sum() / x.size()[0] # batch mean

    return BCE_loss, KLD

