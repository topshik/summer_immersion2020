import math

import numpy as np
import torch
import torch.nn as nn


def log_normal_diag(z, mean, log_var,  dim=None):
    log_normal = -0.5 * (math.log(2.0 * math.pi) + log_var + (torch.pow(z - mean, 2) /
                         (torch.exp(log_var) + 1e-5)))
    return log_normal.sum(dim=dim)


class Prior:
    def log_p_z(self, z) -> torch.Tensor:
        pass

    def generate_z(self, batch_size, latent_size) -> torch.Tensor:
        pass


class SimpleGaussian(Prior):
    def __init__(self, batch_size, latent_size) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.latent_size = latent_size

    def log_p_z(self, z) -> torch.Tensor:
        return - 0.5 * (math.log(2 * math.pi) + z.pow(2)).sum(dim=1)

    def generate_z(self, batch_size, latent_size) -> torch.Tensor:
        z = torch.randn([batch_size, latent_size])

        return z


class MoG(Prior):
    def __init__(self, z1_size, input_type, ll, arc, **kwargs):
        super().__init__()
        # super().__init__(z1_size, input_type, ll, arc)
        # self.num_comp = kwargs['number_components']
        # self.incremental = kwargs['incremental']
        # self.mog_mu = nn.Parameter(torch.FloatTensor(1, self.num_comp, self.hid_dim))
        # self.mog_logvar = nn.Parameter(torch.FloatTensor(1, self.num_comp, self.hid_dim))
        # self.init_comp()

        self.mog_mu = nn.Parameter(torch.FloatTensor(1, self.num_comp, self.hid_dim))
        self.mog_logvar = nn.Parameter(torch.FloatTensor(1, self.num_comp, self.hid_dim))

    def log_p_z(self, z):
        z_expand = z.unsqueeze(1)  # MB x 1 x hid
        # mu and logsigma 1 x C х hid

        # MB x C:
        log_comps = log_normal_diag(z_expand, self.mog_mu, self.mog_logvar, dim=2)
        num_tsk = len(self.learned_mu)

        log_comps -= math.log(self.num_comp * (1 + num_tsk))
        log_prior = torch.logsumexp(log_comps, 1)  # MB x 1
        return log_prior

    def generate_x(self, N=25):
        num_tsk = len(self.learned_mu)
        mixture_idx = np.random.choice(self.mog_mu.shape[1] * (1 + num_tsk), size=N,
                                       replace=True)
        z_sample_rand = self.reparameterize(self.mog_mu[0, mixture_idx],
                                            self.mog_logvar.data[0, mixture_idx])

        samples_rand, _ = self.p_x(z_sample_rand)
        return samples_rand


    def init_comp(self):
        self.mog_mu.data.normal_(0, 0.5)
        self.mog_logvar.data.fill_(-2)

