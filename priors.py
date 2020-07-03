import math

import numpy as np
import torch
import torch.nn as nn


def log_normal_diag(z, mean, log_var,  dim=None):
    log_normal = -0.5 * (math.log(2.0 * math.pi) + log_var + (torch.pow(z - mean, 2) /
                         (torch.exp(log_var) + 1e-5)))
    return log_normal.sum(dim=dim)


class Prior:
    # TODO: manage devices and latent_size
    def log_p_z(self, z) -> torch.Tensor:
        pass

    def generate_z(self, batch_size) -> torch.Tensor:
        pass


class SimpleGaussian(Prior):
    def __init__(self, latent_size) -> None:
        super().__init__()
        self.latent_size = latent_size

    def log_p_z(self, z) -> torch.Tensor:
        return - 0.5 * (math.log(2 * math.pi) + z.pow(2)).sum(dim=1)

    def generate_z(self, batch_size) -> torch.Tensor:
        z = torch.randn([batch_size, self.latent_size])

        return z


class MoG(Prior):
    def __init__(self, num_comp, latent_size):
        super().__init__()
        self.num_comp = num_comp
        self.latent_size = latent_size
        self.mog_mu = nn.Parameter(torch.FloatTensor(1, self.num_comp, self.latent_size)).cuda()
        self.mog_logvar = nn.Parameter(torch.FloatTensor(1, self.num_comp, self.latent_size)).cuda()
        self.init_comp()

    def log_p_z(self, z):
        # MB x 1 x hid
        # mu and logsigma 1 x C х hid
        z_expand = z.unsqueeze(1)

        # MB x C:
        log_comps = log_normal_diag(z_expand, self.mog_mu, self.mog_logvar, dim=2)

        log_comps -= math.log(self.num_comp)
        log_prior = torch.logsumexp(log_comps, 1)  # MB x 1

        return log_prior

    def generate_z(self, batch_size):
        mixture_idx = np.random.choice(self.mog_mu.shape[1], size=batch_size, replace=True)
        z = torch.randn([batch_size, self.latent_size]).cuda()
        z *= self.mog_logvar.data[0, mixture_idx].exp()
        z += self.mog_mu[0, mixture_idx]

        return z

    def init_comp(self):
        self.mog_mu.data.normal_(0, 0.5)
        self.mog_logvar.data.fill_(-2)


class Vamp(Prior):
    def __init__(self, n_components, latent_size, input_size, encoder):
        super().__init__()
        self.n_components = n_components
        self.latent_size = latent_size
        self.input_size = input_size
        self.encoder = encoder

        # word embeddings values
        # self.pseudoinputs_mean = torch.zeros(self.n_components)
        # self.pseudoinputs_std = torch.ones(self.n_components)
        self.pseudoinputs_mean = 0.
        self.pseudoinputs_std = 1.
        min_inp = -10

        self.means = nn.Sequential(nn.Linear(self.n_components, self.input_size.prod().item(), bias=False),
                                   nn.Hardtanh(min_val=min_inp, max_val=1.0)).cuda()

        self.means[0].weight.data.normal_(self.pseudoinputs_mean, self.pseudoinputs_std)

        # create an idle input for calling pseudo-inputs
        self.idle_input = nn.Parameter(torch.eye(self.n_components, self.n_components), requires_grad=False).cuda()
        # self.idle_input.data = torch.eye(self.n_components, self.n_components)

    def log_p_z(self, z):
        # z: MB x hid
        # C x inp_dim
        x = self.means(self.idle_input).reshape(torch.Size([self.n_components]) + torch.Size(self.input_size))

        z_p_mean, z_p_logvar = self.encoder(x)  # C x hid
        z_expand = z.unsqueeze(1)  # MB x 1 x hid
        means = z_p_mean.unsqueeze(0)  # 1 x C x hid
        logvars = z_p_logvar.unsqueeze(0)  # 1 x C x hid

        # MB x C
        log_comps = log_normal_diag(z_expand, means, logvars, dim=2) - math.log(self.n_components)
        log_prior = torch.logsumexp(log_comps, dim=1)  # MB x 1

        return log_prior

    def generate_z(self, batch_size=25):
        idx = np.random.choice(self.n_components, size=batch_size, replace=True)
        # batch_size x inp_dim
        means = self.means(self.idle_input[idx]).reshape(torch.Size([batch_size]) + torch.Size(self.input_size))
        z_sample_gen_mean, z_sample_gen_logvar = self.encoder(means)  # batch_size x hid
        z = torch.randn([batch_size, self.latent_size]).cuda()
        z *= z_sample_gen_logvar.data.exp()
        z += z_sample_gen_mean.data

        return z
