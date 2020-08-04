import os
from typing import Callable, List, Tuple

import fastcluster
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import scipy
import torch

import priors


def idx2word(idx: torch.tensor, i2w: Callable, pad_idx: int, eos_idx: int) -> torch.tensor:
    sent_str = [str()] * len(idx)
    for i, sent in enumerate(idx):
        for word_id in sent:
            if word_id == pad_idx:
                break
            sent_str[i] += i2w[str(word_id.item())] + " "
            if word_id == eos_idx:
                break
        sent_str[i] = sent_str[i].strip()

    return sent_str


def interpolate(start: np.array, end: np.array, steps: int) -> np.array:
    interpolation = np.zeros((start.shape[0], steps + 2))
    for dim, (s, e) in enumerate(zip(start, end)):
        interpolation[dim] = np.linspace(s, e, steps+2)

    return interpolation.T


def seriation(dendrogram: np.array, n_points: int, cur_index: int) -> List[int]:
    """
    Compute the order implied by a hierarchical tree (dendrogram)
    :param dendrogram: hierarchical tree
    :param n_points: number of points given to the clustering process
    :param cur_index: position in the tree for the recursive traversal
    :return: order implied by the hierarchical tree
    """
    if cur_index < n_points:
        return [cur_index]
    else:
        left = int(dendrogram[cur_index - n_points, 0])
        right = int(dendrogram[cur_index - n_points, 1])
        return seriation(dendrogram, n_points, left) + seriation(dendrogram, n_points, right)


def compute_serial_matrix(dist_mat: np.array, method: str = "ward") -> Tuple[np.array, List[int], np.array]:
    """
    Transform a distance matrix into a sorted distance matrix according to the order implied by
    the hierarchical tree (dendrogram)
    :param dist_mat: pairwise distance matrix for studied objects
    :param method: `["ward","single","average","complete"]`
    :return:
        - input dist_mat, but with re-ordered rows and columns according to the hierarchical tree
        - order implied by the hierarhical tree
        - hierarhical tree (dendrogram)
    """
    n_points = len(dist_mat)
    flat_dist_mat = scipy.spatial.distance.squareform(dist_mat)
    res_linkage = fastcluster.linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, n_points, n_points + n_points - 2)
    seriated_dist = np.zeros((n_points, n_points))
    a, b = np.triu_indices(n_points, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]

    return seriated_dist, res_order, res_linkage


class ValLossEarlyStopping(EarlyStopping):
    def __init__(self, version, *args, **kwargs) -> None:
        """
        Wrapper for default callback that logs NLL when training is stopped
        """
        super().__init__(*args, **kwargs)
        self.version = version

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._run_early_stopping_check(trainer, pl_module)
        if getattr(trainer, "should_stop") or pl_module.current_epoch == pl_module.config.train.max_epochs - 1:
            if isinstance(pl_module.prior, priors.MoG):
                means = pl_module.prior.mog_mu.cpu().detach().numpy()[0]
                dist_mat = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(means))
                reordered_dist_mat = compute_serial_matrix(dist_mat)[0]
                figure, axes = plt.subplots(figsize=(10, 10))
                cbar_ax = figure.add_axes([.9, 0.15, 0.05, 0.7])
                figure.colorbar(axes.pcolormesh(reordered_dist_mat), cbar_ax)
                plt.xlim([0, np.max(reordered_dist_mat)])
                plt.ylim([0, np.max(reordered_dist_mat)])
                plt.savefig(os.path.join(pl_module.config.hydra_base_dir, f"version_{self.version}_pdist_matrix.png"))

            # dumps metrics for current launch
            with open(f"{pl_module.config.hydra_base_dir}/metrics.csv", "a") as output:
                output.write(",".join([pl_module.config.prior.type,
                                       str(pl_module.config.train.max_epochs),
                                       str(pl_module.config.kl.zero_epochs),
                                       str(pl_module.config.kl.anneal_function),
                                       f"{pl_module.config.kl.weight:.4f}",
                                       f"{pl_module.val_avg_kl:.4f}",
                                       f"{pl_module.val_avg_nll:.4f}",
                                       f"{pl_module.val_avg_elbo:.4f}",
                                       f"{pl_module.calculate_likelihood().item():.4f}", "\n"]))
            print(f"\nFind logs in file: {pl_module.config.hydra_base_dir}/metrics.csv")
