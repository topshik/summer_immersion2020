import os

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

import plmodel


@hydra.main(config_path="train-config.yaml", strict=False)
def train(config: DictConfig) -> None:
    config.hydra_base_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    model = plmodel.PLSentenceVAE(config)

    with open(f"{config.hydra_base_dir}/metrics.csv", "w") as output:
        output.write("epochs number,zero KL epochs number,KL mode,KL beta value,KL,NLL,ELBO,"
                     "NLL (importance sampling)\n")

    trainer = pl.Trainer(max_epochs=config.max_epochs, gpus=1, auto_select_gpus=True)
    trainer.fit(model)


if __name__ == "__main__":
    train()
