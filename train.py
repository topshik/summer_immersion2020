import os

import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl

import plmodel
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


@hydra.main(config_path="train-config.yaml", strict=False)
def train(config: DictConfig) -> None:
    config.hydra_base_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())

    with open(f"{config.hydra_base_dir}/metrics.csv", "w") as output:
        output.write("version,epochs number,zero KL epochs number,KL mode,KL beta value,KL,NLL,ELBO,"
                     "NLL (importance sampling)\n")

    start_version = max(map(int, [name.split('_')[1] for name in os.listdir("lightning_logs")])) + 1
    current_version = start_version
    print(f"Starting from version: {start_version}")
    print(f"Writing logs to file {config.hydra_base_dir}/metrics.csv")

    beta_space = np.linspace(1e-2, 5, 10)
    modes_space = ["const", "linear", "logistic"]

    for beta in beta_space:
        for mode in modes_space:
            config.kl.weight = beta
            config.kl.anneal_function = mode

            with open(f"{config.hydra_base_dir}/metrics.csv", "a") as output:
                output.write(f"{current_version},")

            model = plmodel.PLSentenceVAE(config)
            checkpoint_callback = ModelCheckpoint(
                save_top_k=(config.chkpnt.top_k if config.chkpnt.top_k != -1 else config.train.max_epochs),
                verbose=True,
                monitor="val_loss",
                mode="min",
                prefix=f"version_{current_version}_"
            )
            trainer = pl.Trainer(max_epochs=config.train.max_epochs,
                                 gpus=1,
                                 auto_select_gpus=True,
                                 early_stop_callback=plmodel.ValLossEarlyStopping(patience=3, min_delta=0.3),
                                 checkpoint_callback=checkpoint_callback)
            trainer.fit(model)

            current_version += 1

    with open(f"{config.hydra_base_dir}/metrics.csv", "a") as output:
        output.write(f"Started from version: {start_version}")

    print(f"Check metrics in file {config.hydra_base_dir}/metrics.csv")


if __name__ == "__main__":
    train()
