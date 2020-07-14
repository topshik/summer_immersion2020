import os
from typing import Dict, List, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils

import priors
from ptb import PTB


class PLSentenceVAE(pl.LightningModule):
    def __init__(self, batch_size: int = 256, max_sequence_length: int = 50, data_directory: str = 'data',
                 create_data: bool = False, embedding_size: int = 300, hidden_size: int = 256, latent_size: int = 256,
                 word_dropout: float = 0, embedding_dropout: float = 0.5, rnn_type: str = 'gru', num_layers: int = 1,
                 bidirectional: bool = False, prior: str = 'SimpleGaussian', n_components: int = 200,
                 anneal_function: str = 'logistic', kl_weight: float = 1.0, k: int = 0.0025, x0: int = None,
                 max_epochs: int = 20, kl_zero_epochs: int = 0) -> None:
        super().__init__()

        self.step = 0
        self.current_epoch += 1
        self.max_epochs = max_epochs

        # datasets and their params
        self.max_sequence_length = max_sequence_length
        self.ptb_train = PTB(data_dir=data_directory, split='train', create_data=create_data,
                             max_sequence_length=self.max_sequence_length)
        self.ptb_val = PTB(data_dir=data_directory, split='valid', create_data=create_data,
                           max_sequence_length=self.max_sequence_length)
        self.sos_idx = self.ptb_train.sos_idx
        self.eos_idx = self.ptb_train.eos_idx
        self.pad_idx = self.ptb_train.pad_idx
        self.unk_idx = self.ptb_train.unk_idx
        self.vocab_size = self.ptb_train.vocab_size

        # kl annealing params
        self.anneal_function = anneal_function
        self.kl_weight = kl_weight
        self.k = k
        self.kl_zero_epochs = kl_zero_epochs
        if x0 is None:
            self.x0 = len(self.ptb_train) * (self.max_epochs - self.kl_zero_epochs)
        else:
            self.x0 = x0

        # dataloaders
        self.batch_size = batch_size
        self.len_train_loader, self.len_val_loader = 0, 0

        # net params and architecture
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        if prior == 'SimpleGaussian':
            self.prior = priors.SimpleGaussian(torch.device('cuda'), self.latent_size)
        elif prior == 'MoG':
            self.prior = priors.MoG(torch.device('cuda'), n_components, self.latent_size)
        elif prior == 'Vamp':
            self.n_components = n_components
            self.prior = priors.Vamp(torch.device('cuda'), n_components, self.latent_size,
                                     input_size=torch.tensor([max_sequence_length, embedding_size]),
                                     encoder=self.q_z)
        else:
            raise ValueError()

        self.embedding = nn.Embedding(self.vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        elif rnn_type == 'lstm':
            rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers
        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2log_var = nn.Linear(hidden_size * self.hidden_factor, latent_size)

        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)

        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), self.vocab_size)

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

        if self.rnn_type == 'lstm':
            hidden = hidden[0]

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            # possibly incorrect, maybe need to permute
            hidden = hidden.view(self.batch_size, self.hidden_size * self.hidden_factor)
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

        z = torch.randn([self.batch_size, self.latent_size]).to(self.device)
        z = z * std + mean

        # decoder
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, self.batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size()).to(self.device)
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
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

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        sequence_idx = torch.arange(0, batch_size, device=self.device).long()
        sequence_running = torch.arange(0, batch_size, device=self.device).long()
        sequence_mask = torch.ones(batch_size, device=self.device).bool()

        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, device=self.device).long()
        # generated_sequenced
        generations = torch.empty((batch_size, self.max_sequence_length), device=self.device).fill_(self.pad_idx).long()

        input_sequence = torch.empty(batch_size, device=self.device).fill_(self.sos_idx).long()
        for t in range(self.max_sequence_length):
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
        train_loader = DataLoader(dataset=self.ptb_train, batch_size=self.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=torch.cuda.is_available(), drop_last=True)
        self.len_train_loader = len(train_loader)
        return train_loader

    def val_dataloader(self) -> DataLoader:
        """
        Create train dataloader from train dataset
        :return: pytorch dataloader
        """
        val_loader = DataLoader(dataset=self.ptb_val, batch_size=self.batch_size, num_workers=4,
                                pin_memory=torch.cuda.is_available(), drop_last=True)
        self.len_val_loader = len(val_loader)

        return val_loader

    def kl_anneal_function(self) -> float:
        """
        Compute current KL divergence coefficient
        :return: coefficient
        """
        if self.anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-self.k * (self.step - self.x0))))
        elif self.anneal_function == 'linear':
            return self.k * min(1., self.step / self.x0)
        elif self.anneal_function == 'zero':
            return 0
        elif self.anneal_function == 'const':
            return self.kl_weight

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

        nll = torch.nn.NLLLoss(ignore_index=self.pad_idx, reduction='sum')
        nll_loss = nll(logp, target) / self.batch_size
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
        logp, mean, log_var, z = self.forward(batch['input'], batch['length'])

        # loss calculation
        nll_loss, kl_loss, kl_weight = self.loss_fn(z, logp, batch['target'], batch['length'], mean, log_var)
        if self.current_epoch < self.kl_zero_epochs:
            kl_weight = 0
        loss = nll_loss + kl_weight * kl_loss
        self.step += 1

        return {'loss': loss, 'NLL loss': nll_loss.data, 'KL loss': kl_loss.data, 'KL weight': kl_weight}

    def training_epoch_end(self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]) \
            -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Log losses to tensorboard
        :param outputs: list of validation step outputs for all batches
        :return: dict of averaged logs
        """
        avg_elbo = torch.stack([x['loss'] for x in outputs]).mean()
        avg_nll = torch.stack([x['NLL loss'] for x in outputs]).mean()
        avg_kl = torch.stack([x['KL loss'] for x in outputs]).mean()
        tensorboard_logs = {'ELBO (train)': avg_elbo.data, 'NLL loss (train)': avg_nll.data,
                            'KL loss (train)': avg_kl.data, 'KL weight': outputs[0]['KL weight']}

        return {'train_loss': avg_elbo, 'log': tensorboard_logs}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Perform validation step
        :param batch: dict with keys (input, target, length); contains preprocessed (pad, sos, eos added) input tensor,
        target tensor and lengths of the sentences
        :param batch_idx: number of batch in current epoch
        :return: dict with training logs
        """
        # Forward pass
        logp, mean, log_var, z = self.forward(batch['input'], batch['length'])

        # loss calculation
        nll_loss, kl_loss, kl_weight = self.loss_fn(z, logp, batch['target'], batch['length'], mean, log_var)
        if self.current_epoch < self.kl_zero_epochs:
            kl_weight = 0
        loss = nll_loss + kl_weight * kl_loss

        return {'loss': loss.data, 'NLL loss': nll_loss.data, 'KL loss': kl_loss.data, 'KL weight': kl_weight}

    def validation_epoch_end(self, outputs: Union[List[Dict[str, torch.Tensor]], List[List[Dict[str, torch.Tensor]]]]) \
            -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Save model, logs losses at the end of validation epoch
        :param outputs: list of validation step outputs for all batches
        :return: dict of averaged logs
        """
        # TODO lightning saving interface
        save_model_path = 'checkpoints'
        checkpoint_path = os.path.join(save_model_path, "E%i.pth" % self.current_epoch)
        torch.save(self.state_dict(), checkpoint_path)

        avg_elbo = torch.stack([x['loss'] for x in outputs]).mean()
        avg_nll = torch.stack([x['NLL loss'] for x in outputs]).mean()
        avg_kl = torch.stack([x['KL loss'] for x in outputs]).mean()
        tensorboard_logs = {'ELBO (val)': avg_elbo.data, 'NLL loss (val)': avg_nll.data, 'KL loss (val)': avg_kl.data}

        return {'val_loss': avg_elbo, 'log': tensorboard_logs}

    def configure_optimizers(self) -> Optimizer:
        """
        Configure optimizer for training
        :return: optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
