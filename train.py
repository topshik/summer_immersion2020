import os

import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl

import plmodel


@hydra.main(config_path="train-config.yaml", strict=False)
def train(config: DictConfig) -> None:
    config.hydra_base_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())

    with open(f"{config.hydra_base_dir}/metrics.csv", "w") as output:
        output.write("epochs number,zero KL epochs number,KL mode,KL beta value,KL,NLL,ELBO,"
                     "NLL (importance sampling)\n")

    start_version = max(map(int, [name.split('_')[1] for name in os.listdir("lightning_logs")]))
    print(f"Starting from version: {start_version + 1}")
    print(f"Writing logs to file {config.hydra_base_dir}/metrics.csv")
    eps = 1e-4

    for beta in np.linspace(0 + eps, 20, 8):
        for mode in ["const", "linear"]:
            config.kl_weight = beta
            config.anneal_function = mode
            model = plmodel.PLSentenceVAE(config)
            trainer = pl.Trainer(max_epochs=config.max_epochs, gpus=1, auto_select_gpus=True)
            trainer.fit(model)

    with open(f"{config.hydra_base_dir}/metrics.csv", "w") as output:
        output.write(f"Started from version: {start_version + 1}")


if __name__ == "__main__":
    train()
