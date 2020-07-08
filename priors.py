import math
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def log_normal_diag(z: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor, dim: int = None) -> torch.Tensor:
    """
    Get log likelyhood of multivariate normal distribution with diagonal covariance matrix
    :param z: latent representation, tensor of shape [batch_size x latent_size]
    :param mean: latent distribution mean parameter, tensor of shape [batch_size x latent_size]
    :param log_var: latent distribution log_variance parameter (diagonal of covariance matrix,
                    tensor of shape [batch_size x latent_size]
    :param dim: dimension for reduction, usually -1 is needed
    :return: tensor of shape [batch_size]
    """
    log_normal = -0.5 * (math.log(2.0 * math.pi) + log_var + (torch.pow(z - mean, 2) /
                         (torch.exp(log_var) + 1e-5)))

    return log_normal.sum(dim=dim)


class Prior:
    # TODO: manage devices and latent_size
    def __init__(self, device: torch.device, latent_size: int) -> None:
        """
        Prior disturb
        :param device: computing device
        :param latent_size: latent representation size
        """
        self.device = device
        self.latent_size = latent_size

    def log_p_z(self, z: torch.Tensor) -> torch.Tensor:
        """
        Return log likelyhood
        :param z: tensor of shape [batch_size x latent_size]
        :return: tensor of shape [batch_size]
        """
        pass

    def generate_z(self, batch_size: int) -> torch.Tensor:
        """
        Generate latent representation
        :param batch_size: batch size
        :return: tensor of shape [batch_size x latent_size]
        """
        pass


class SimpleGaussian(Prior):
    def __init__(self, device: torch.device, latent_size: int) -> None:
        super().__init__(device, latent_size)

    def log_p_z(self, z: torch.Tensor) -> torch.Tensor:
        return - 0.5 * (math.log(2 * math.pi) + z.pow(2)).sum(dim=1)

    def generate_z(self, batch_size: int) -> torch.Tensor:
        z = torch.randn([batch_size, self.latent_size])

        return z


class MoG(Prior):
    def __init__(self, device: torch.device, num_comp: int, latent_size: int) -> None:
        super().__init__(device, latent_size)
        self.num_comp = num_comp
        self.mog_mu = nn.Parameter(torch.FloatTensor(1, self.num_comp, self.latent_size)).to(self.device)
        self.mog_log_var = nn.Parameter(torch.FloatTensor(1, self.num_comp, self.latent_size)).to(self.device)

        # init components
        self.mog_mu.data.normal_(0, 0.5)
        self.mog_log_var.data.fill_(-2)

    def log_p_z(self, z: torch.Tensor) -> torch.Tensor:
        z_expand = z.unsqueeze(1)
        log_comps = log_normal_diag(z_expand, self.mog_mu, self.mog_log_var, dim=2)  # batch_size x components_num
        log_comps -= math.log(self.num_comp)
        log_prior = torch.logsumexp(log_comps, 1)

        return log_prior

    def generate_z(self, batch_size: int) -> torch.Tensor:
        mixture_idx = np.random.choice(self.mog_mu.shape[1], size=batch_size, replace=True)
        z = torch.randn([batch_size, self.latent_size]).to(self.device)
        z *= self.mog_log_var.data[0, mixture_idx].exp()
        z += self.mog_mu[0, mixture_idx]

        return z


class Vamp(Prior):
    def __init__(self, device: torch.device, n_components: int, latent_size: int, input_size: torch.Tensor,
                 encoder: Callable[[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor]]) -> None:
        """
        Variational mixture of posteriors prior
        :param n_components: number of learnable components
        :param latent_size: model latent size
        :param input_size: [max_sequence_length, embedding_size] for generating pseudoinputs
        :param encoder: callable object, which maps inputs to parameters of the variational posterior
        """
        super().__init__(device, latent_size)
        self.n_components = n_components
        self.input_size = input_size
        self.encoder = encoder

        # word embeddings values
        self.pseudoinputs_mean = 0.
        self.pseudoinputs_std = 1.
        max_input_value = 10

        # learnable input parameters
        self.means = nn.Sequential(nn.Linear(self.n_components, self.input_size.prod().item(), bias=False),
                                   nn.Hardtanh(min_val=-max_input_value, max_val=max_input_value)).to(self.device)
        self.means[0].weight.data.normal_(self.pseudoinputs_mean, self.pseudoinputs_std)

        self.idle_input = nn.Parameter(torch.eye(self.n_components, self.n_components),
                                       requires_grad=False).to(self.device)

    def log_p_z(self, z: torch.Tensor) -> torch.Tensor:
        # components_num x input_size
        x = self.means(self.idle_input).reshape(torch.Size([self.n_components]) + torch.Size(self.input_size))
        z_p_mean, z_p_log_var = self.encoder(x)  # components_num x latent_size
        z_expand = z.unsqueeze(1)               # batch_size x 1 x latent_size
        means = z_p_mean.unsqueeze(0)           # 1 x components_num x latent_size
        log_vars = z_p_log_var.unsqueeze(0)       # 1 x components_num x latent_size

        # batch_size x components_num
        log_comps = log_normal_diag(z_expand, means, log_vars, dim=2) - math.log(self.n_components)
        log_prior = torch.logsumexp(log_comps, dim=1)

        return log_prior

    def generate_z(self, batch_size: int = 25, idx: int = None) -> torch.Tensor:
        if idx is None:
            idx = np.random.choice(self.n_components, size=batch_size, replace=True)
        # batch_size x inp_dim
        means = self.means(self.idle_input[idx]).reshape(torch.Size([batch_size]) + torch.Size(self.input_size))
        z_sample_gen_mean, z_sample_gen_log_var = self.encoder(means)  # batch_size x latent_size
        z = torch.randn([batch_size, self.latent_size]).to(self.device)
        z *= z_sample_gen_log_var.data.exp()
        z += z_sample_gen_mean.data

        return z
