import os

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

import plmodel


@hydra.main(config_path="train_config.yaml")
def train(params: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    model = plmodel.PLSentenceVAE(**params)
    trainer = pl.Trainer(max_epochs=params["max_epochs"], gpus=1, auto_select_gpus=True)
    trainer.fit(model)


if __name__ == "__main__":
    train()
