import os
import shutil

import hydra
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl

import plmodel
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import utils
from validate import validate


@hydra.main(config_path="train-config.yaml", strict=False)
def train(config: DictConfig) -> None:
    config.hydra_base_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())

    # save config near the results
    shutil.copy2("train-config.yaml", os.path.join(config.hydra_base_dir, "train-config.yaml"))

    with open(f"{config.hydra_base_dir}/metrics.csv", "w") as output:
        output.write("version,prior,epochs number,zero KL epochs number,KL mode,KL beta value,KL,NLL,ELBO,"
                     "NLL (importance sampling)\n")

    if os.path.exists("lightning_logs"):
        start_version = max(map(int, [name.split('_')[1] for name in os.listdir("lightning_logs")])) + 1
    else:
        start_version = 0

    current_version = start_version
    print(f"Starting from version: {start_version}")
    print(f"Writing logs to file {config.hydra_base_dir}/metrics.csv")

    beta_space = np.linspace(0.5, 10, 5)
    modes_space = ["logistic"]
    priors = ["MoG", "Vamp"]

    for beta in beta_space:
        for mode in modes_space:
            for prior in priors:
                config.kl.weight = beta
                config.kl.anneal_function = mode
                config.prior.type = prior

                with open(f"{config.hydra_base_dir}/metrics.csv", "a") as output:
                    output.write(f"{current_version},")

                model = plmodel.PLSentenceVAE(config)
                checkpoint_callback = ModelCheckpoint(
                    filepath="/".join([config.hydra_base_dir, f"{current_version}_" + "{epoch}-{val_loss:.2f}"]),
                    save_top_k=(config.chkpnt.top_k if config.chkpnt.top_k != -1 else config.train.max_epochs),
                    verbose=True,
                    monitor="val_loss",
                    mode="min",
                )
                trainer = pl.Trainer(max_epochs=1,
                                     gpus=1,
                                     auto_select_gpus=True,
                                     early_stop_callback=utils.ValLossEarlyStopping(
                                         version=current_version,
                                         patience=1,
                                         min_delta=0.1),
                                     checkpoint_callback=checkpoint_callback)
                trainer.fit(model)

                with open(f"{config.hydra_base_dir}/{current_version}_samples.txt", "w") as output:
                    validate(model, output)

                current_version += 1

    with open(f"{config.hydra_base_dir}/metrics.csv", "a") as output:
        output.write(f"Started from version: {start_version}")

    print(f"Check metrics in file {config.hydra_base_dir}/metrics.csv")


if __name__ == "__main__":
    train()
