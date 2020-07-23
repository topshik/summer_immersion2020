from typing import Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils

import priors
from ptb import PTB


class PLSentenceVAE(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()

        # datasets related attributes
        self.ptb_train = PTB(data_dir=config.dataset.data_directory,
                             split="train",
                             create_data=config.dataset.create_data,
                             max_sequence_length=config.dataset.max_sequence_length)
        self.ptb_val = PTB(data_dir=config.dataset.data_directory,
                           split="valid",
                           create_data=config.dataset.create_data,
                           max_sequence_length=config.dataset.max_sequence_length)
        self.sos_idx = self.ptb_train.sos_idx
        self.eos_idx = self.ptb_train.eos_idx
        self.pad_idx = self.ptb_train.pad_idx
        self.unk_idx = self.ptb_train.unk_idx
        self.vocab_size = self.ptb_train.vocab_size

        # kl annealing
        if config.kl.x0 is None:
            config.kl.x0 = len(self.ptb_train) * (config.train.max_epochs -
                                                  config.kl.zero_epochs) // config.train.batch_size

        config.kl.zero_steps = len(self.ptb_train) * config.kl.zero_epochs // config.train.batch_size

        # dataloaders
        self.len_train_loader, self.len_val_loader = 0, 0

        # logs
        self.val_avg_elbo = 0
        self.val_avg_nll = 0
        self.val_avg_kl = 0

        # model architecture
        if config.prior.type == "SimpleGaussian":
            self.prior = priors.SimpleGaussian(torch.device("cuda"), config.model.latent_size)
        elif config.prior.type == "MoG":
            self.prior = priors.MoG(torch.device("cuda"), config.prior.n_components, config.model.latent_size)
        elif config.prior.type == "Vamp":
            self.prior = priors.Vamp(torch.device("cuda"), config.prior.n_components, config.model.latent_size,
                                     input_size=torch.tensor([config.dataset.max_sequence_length,
                                                              config.model.embedding_size]),
                                     encoder=self.q_z)
        else:
            raise ValueError()

        self.embedding = nn.Embedding(self.vocab_size, config.model.embedding_size)
        self.embedding_dropout = nn.Dropout(p=config.model.embedding_dropout)

        if config.model.rnn_type == "rnn":
            rnn = nn.RNN
        elif config.model.rnn_type == "gru":
            rnn = nn.GRU
        elif config.model.rnn_type == "lstm":
            rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(config.model.embedding_size, config.model.hidden_size,
                               num_layers=config.model.num_layers,
                               bidirectional=config.model.bidirectional,
                               batch_first=True)

        self.hidden_factor = (2 if config.model.bidirectional else 1) * config.model.num_layers
        self.hidden2mean = nn.Linear(config.model.hidden_size * self.hidden_factor, config.model.latent_size)
        self.hidden2log_var = nn.Linear(config.model.hidden_size * self.hidden_factor, config.model.latent_size)
        self.latent2hidden = nn.Linear(config.model.latent_size, config.model.hidden_size * self.hidden_factor)

        self.decoder_rnn = rnn(config.model.embedding_size, config.model.hidden_size,
                               num_layers=config.model.num_layers,
                               bidirectional=config.model.bidirectional,
                               batch_first=True)

        self.outputs2vocab = nn.Linear(config.model.hidden_size * (2 if config.model.bidirectional else 1),
                                       self.vocab_size)

        self.config = config

    def q_z(self, input_embedding: torch.Tensor, sorted_lengths: torch.Tensor = None) -> \
            Tuple[torch.Tensor, torch.Tensor]:
        """
        Encoder forward pass
        :param input_embedding: batch of embedded input sequences,
                                tensor of shape [batch_size, max_sequence_length, embedding_size]
        :param sorted_lengths: lengths of the sequences sorted in descending order,
                               tensor of shape [batch_size]
        :return: mean and log_var of the variational posterior,
                 tensors of shapes [batch_size x latent_size]
        """
        if sorted_lengths is None:
            sorted_lengths = torch.ones(input_embedding.shape[0]) * self.max_sequence_length

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        _, hidden = self.encoder_rnn(packed_input)

        if self.config.model.rnn_type == "lstm":
            hidden = hidden[0]

        if self.config.model.bidirectional or self.config.model.num_layers > 1:
            # flatten hidden state
            # possibly incorrect, maybe need to permute
            hidden = hidden.view(self.config.train.batch_size, self.config.model.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # reparametrization
        mean = self.hidden2mean(hidden)
        log_var = self.hidden2log_var(hidden)

        return mean, log_var

    def forward(self, input_sequence: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                                                   torch.Tensor, torch.Tensor]:
        """
        Perform forward pass for the model
        :param input_sequence: preprocessed sentences (pad, sos, eos added),
        :param length: lengths of the sentences, tensor of shape [batch_size]
        :return: tuple of four torch.Tensors:
            1st - prediction logits
            2nd - mean of variational posterior
            3rd - log variance of variational posterior
            4th - sample from variational posterior
        """
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # encoder
        input_embedding = self.embedding(input_sequence)
        mean, log_var = self.q_z(input_embedding, sorted_lengths)
        std = torch.exp(0.5 * log_var)

        z = torch.randn([self.config.train.batch_size, self.config.model.latent_size]).to(self.device)
        z = z * std + mean

        # decoder
        hidden = self.latent2hidden(z)

        if self.config.model.bidirectional or self.config.model.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, self.config.train.batch_size, self.config.model.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if self.config.model.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size()).to(self.device)
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.config.model.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)

        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        # FIXME lstm rnn type breaks here
        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _, reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b, s, _ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)
        # disable <unk> tokens
        # logp.data[:, :, self.unk_idx] -= 1e12

        return logp, mean, log_var, z

    def inference(self, batch_size: int = 4, z: torch.Tensor = None) -> torch.Tensor:
        """
        Create samples either pure or for interpolation
        :param batch_size: number of the samples to be produced
        :param z: sample from variational posterior to be used
        :return: generated sequence
        """
        if z is None:
            z = self.prior.generate_z(batch_size=batch_size).to(self.device)
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.config.model.bidirectional or self.config.model.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.config.model.hidden_size)

        hidden = hidden.unsqueeze(0)

        sequence_idx = torch.arange(0, batch_size, device=self.device).long()
        sequence_running = torch.arange(0, batch_size, device=self.device).long()
        sequence_mask = torch.ones(batch_size, device=self.device).bool()

        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, device=self.device).long()
        # generated_sequenced
        generations = torch.empty((batch_size, self.config.dataset.max_sequence_length),
                                  device=self.device).fill_(self.pad_idx).long()

        input_sequence = torch.empty(batch_size, device=self.device).fill_(self.sos_idx).long()
        for t in range(self.config.dataset.max_sequence_length):
            input_sequence.unsqueeze_(1)
            input_embedding = self.embedding(input_sequence)
            output, hidden = self.decoder_rnn(input_embedding, hidden)
            logits = self.outputs2vocab(output)
            # disables <unk> tokens
            logits.data[:, :, self.unk_idx] -= 1e12

            # sample
            _, input_sequence = torch.topk(logits, 1, dim=-1)
            input_sequence = input_sequence.squeeze(-1)
            input_sequence = input_sequence.squeeze(-1)

            # save new tokens to generations
            running_latest = generations[sequence_running]
            running_latest[:, t] = input_sequence.data
            generations[sequence_running] = running_latest

            sequence_mask[sequence_running] = (input_sequence != self.eos_idx)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]
                running_seqs = torch.arange(0, len(running_seqs), device=self.device).long()
            else:
                break

        return generations

    def train_dataloader(self) -> DataLoader:
        """
        Create train dataloader from train dataset
        :return: pytorch dataloader
        """
        train_loader = DataLoader(dataset=self.ptb_train, batch_size=self.config.train.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=torch.cuda.is_available(), drop_last=True)
        self.len_train_loader = len(train_loader)

        return train_loader

    def val_dataloader(self) -> DataLoader:
        """
        Create validation dataloader from validation dataset
        :return: pytorch dataloader
        """
        val_loader = DataLoader(dataset=self.ptb_val, batch_size=self.config.train.batch_size, num_workers=4,
                                pin_memory=torch.cuda.is_available(), drop_last=True)
        self.len_val_loader = len(val_loader)

        return val_loader

    def kl_anneal_function(self) -> float:
        """
        Compute current KL divergence coefficient
        :return: coefficient
        """
        if self.current_epoch < self.config.kl.zero_epochs:
            return 0.

        if self.config.kl.anneal_function == "logistic":
            return float(1 / (1 + np.exp(-self.config.kl.k * (self.global_step - self.config.kl.x0))))
        elif self.config.kl.anneal_function == "linear":
            return self.config.kl.weight * min(1., (self.global_step - self.config.kl.zero_steps) / self.config.kl.x0)
        elif self.config.kl.anneal_function == "const":
            return self.config.kl.weight

    def loss_fn(self, z, logp: torch.Tensor, target: torch.Tensor, length: torch.Tensor, mean: torch.Tensor,
                log_var: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute loss
        :param z: latent representation
        :param logp: prediction logits tensor of shape [batch_size, max_sentence_length, vocab_size]
        :param target: target tensor of shape [batch_size, max_sentence_length]
        :param length: sentences lengths tensor of shape [batch_size]
        :param mean: mean of the variational posterior of shape [batch_size, latent_size]
        :param log_var: log variance of the variational posterior of shape [batch_size, latent_size]
        :return: tuple of reconstruction loss, KL loss and KL weight
        """
        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)

        logp = logp.view(-1, logp.size(2))

        nll = torch.nn.NLLLoss(ignore_index=self.pad_idx, reduction="sum")
        nll_loss = nll(logp, target) / self.config.train.batch_size
        kl_loss = self.kl_loss_mc(z, mean, log_var)
        kl_weight = self.kl_anneal_function()

        return nll_loss, kl_loss, kl_weight

    def kl_loss_mc(self, z: torch.Tensor, mean, log_var) -> torch.Tensor:
        """
        Monte Carlo estimator for KL divergence loss part
        :param z: latent representation, tensor of shape [batch_size x latent_size]
        :param mean: mean of the variational posterior, tensor of shape [batch_size, latent_size]
        :param log_var: log variance of the variational posterior, tensor of shape [batch_size, latent_size]
        :return: batch averaged KL loss
        """
        log_q_z = priors.log_normal_diag(z, mean, log_var, dim=1)
        log_p_z = self.prior.log_p_z(z)

        return torch.mean(log_q_z - log_p_z)

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Perform training step
        :param batch: dict with keys (input, target, length); contains preprocessed (pad, sos, eos added) input tensor,
        target tensor and lengths of the sentences
        :param batch_idx: number of batch in current epoch
        :return: dict with training logs
        """
        # Forward pass
        logp, mean, log_var, z = self.forward(batch["input"], batch["length"])

        # loss calculation
        nll_loss, kl_loss, kl_weight = self.loss_fn(z, logp, batch["target"], batch["length"], mean, log_var)
        loss = nll_loss + kl_weight * kl_loss

        return {"loss": loss, "NLL loss": nll_loss.data, "KL loss": kl_loss.data, "KL weight": kl_weight}

    def training_epoch_end(self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]) \
            -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Log losses to tensorboard
        :param outputs: list of validation step outputs for all batches
        :return: dict of averaged logs
        """
        avg_elbo = torch.stack([x["loss"] for x in outputs]).mean()
        avg_nll = torch.stack([x["NLL loss"] for x in outputs]).mean()
        avg_kl = torch.stack([x["KL loss"] for x in outputs]).mean()
        tensorboard_logs = {"ELBO (train)": avg_elbo.data, "NLL loss (train)": avg_nll.data,
                            "KL loss (train)": avg_kl.data, "KL weight": outputs[0]["KL weight"]}

        return {"train_loss": avg_elbo, "log": tensorboard_logs}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Perform validation step
        :param batch: dict with keys (input, target, length); contains preprocessed (pad, sos, eos added) input tensor,
        target tensor and lengths of the sentences
        :param batch_idx: number of batch in current epoch
        :return: dict with training logs
        """
        # Forward pass
        logp, mean, log_var, z = self.forward(batch["input"], batch["length"])

        # loss calculation
        nll_loss, kl_loss, kl_weight = self.loss_fn(z, logp, batch["target"], batch["length"], mean, log_var)
        if self.current_epoch < self.config.kl.zero_epochs:
            kl_weight = 0
        loss = nll_loss + kl_weight * kl_loss

        return {"loss": loss.data, "NLL loss": nll_loss.data, "KL loss": kl_loss.data, "KL weight": kl_weight}

    def validation_epoch_end(self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]) \
            -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Save model, logs losses at the end of validation epoch
        :param outputs: list of validation step outputs for all batches
        :return: dict of averaged logs
        """
        avg_elbo = torch.stack([x["loss"] for x in outputs]).mean()
        avg_nll = torch.stack([x["NLL loss"] for x in outputs]).mean()
        avg_kl = torch.stack([x["KL loss"] for x in outputs]).mean()
        tensorboard_logs = {"ELBO (val)": avg_elbo.data, "NLL loss (val)": avg_nll.data, "KL loss (val)": avg_kl.data}

        self.val_avg_elbo = avg_elbo
        self.val_avg_nll = avg_nll
        self.val_avg_kl = avg_kl

        return {"val_loss": avg_elbo, "log": tensorboard_logs}

    def configure_optimizers(self) -> Optimizer:
        """
        Configure optimizer for training
        :return: optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def calculate_elbo_batch(self, samples_input, samples_length, samples_target):
        """
        Calculate ELBO for the batch
        :return: ELBO value
        """
        samples_input, samples_target = samples_input.to(self.device), samples_target.to(self.device)
        logp, mean, log_var, z = self.forward(samples_input, samples_length)
        nll_loss, kl_loss, kl_weight = self.loss_fn(z, logp, samples_target, samples_length, mean, log_var)
        elbo = nll_loss + kl_loss

        return elbo

    def calculate_likelihood(self, n_importance_samples=300) -> torch.Tensor:
        """
        Calculate NLL via importance sampling
        :param n_importance_samples: number of samples in Important sampling (per observation)
        :return: NLL averaged via importance sampling on validation set
        """
        n_samples = len(self.ptb_val)
        likelihood_test = []

        if n_importance_samples <= self.config.train.batch_size:
            n_iterations = 1
        else:
            n_iterations = n_importance_samples // self.config.train.batch_size
            n_importance_samples = self.config.train.batch_size

        for i in range(n_samples):
            point_input = torch.from_numpy(self.ptb_val[i]["input"]).unsqueeze(0)
            point_length = torch.tensor(self.ptb_val[i]["length"])
            point_target = torch.from_numpy(self.ptb_val[i]["target"]).unsqueeze(0)

            point_input_expanded = point_input.expand(n_importance_samples, point_input.size(1))
            point_length_expanded = point_length.expand(n_importance_samples)
            point_target_expanded = point_target.expand(n_importance_samples, point_target.size(1))
            elbo_for_sample = torch.zeros(n_iterations).to(self.device)
            for j in range(n_iterations):
                elbo_for_sample[j] = (-self.calculate_elbo_batch(point_input_expanded,
                                                                 point_length_expanded,
                                                                 point_target_expanded).data)

            likelihood_x = torch.logsumexp(elbo_for_sample, dim=0)
            likelihood_test.append(likelihood_x - np.log(len(elbo_for_sample)))

        likelihood_test = torch.tensor(likelihood_test)

        return -likelihood_test.mean()


class ValLossEarlyStopping(EarlyStopping):
    def __init__(self, *args, **kwargs) -> None:
        """
        Wrapper for default callback that logs NLL when training is stopped
        """
        super().__init__(*args, **kwargs)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._run_early_stopping_check(trainer, pl_module)
        if getattr(trainer, "should_stop") or pl_module.current_epoch == pl_module.config.train.max_epochs - 1:
            # dumps metrics for current launch
            with open(f"{pl_module.config.hydra_base_dir}/metrics.csv", "a") as output:
                output.write(",".join([str(pl_module.config.train.max_epochs),
                                       str(pl_module.config.kl.zero_epochs),
                                       str(pl_module.config.anneal_function),
                                       f"{pl_module.config.kl.weight:.4f}",
                                       f"{pl_module.val_avg_kl:.4f}",
                                       f"{pl_module.val_avg_nll:.4f}",
                                       f"{pl_module.val_avg_elbo:.4f}",
                                       f"{pl_module.calculate_likelihood().item():.4f}", "\n"]))
            print(f"\nFind logs in file: {pl_module.config.hydra_base_dir}/metrics.csv")
