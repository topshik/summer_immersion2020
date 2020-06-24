from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils

from ptb import PTB


def save_sample(save_to: torch.Tensor, sample: torch.Tensor, running_seqs: torch.Tensor, t: int):
    # select only still running
    running_latest = save_to[running_seqs]
    # update token at position t
    running_latest[:, t] = sample.data
    # save back
    save_to[running_seqs] = running_latest

    return save_to


class PLSentenceVAE(pl.LightningModule):
    def __init__(self, vocab_size: int = 9877, k: int = 0.0025, x0: int = 2500, embedding_size: int = 300,
                 max_sequence_length: int = 60, rnn_type: str = 'rnn', latent_size: int = 16, hidden_size: int = 256,
                 word_dropout: float = 0, embedding_dropout: float = 0.5, num_layers: int = 1,
                 bidirectional: bool = False, print_every: int = 50) -> None:
        super().__init__()
        # kl annealing params
        self.step = 0
        self.k = k
        self.x0 = x0

        # datasets and their params, are set in prepare_data()
        self.batch_size = 32
        self.len_train_loader = 0
        self.ptb_train, self.ptb_val = None, None

        self.new_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.print_every = print_every

        # tokens info
        self.max_sequence_length = max_sequence_length
        self.sos_idx = 0
        self.eos_idx = 0
        self.pad_idx = 0
        self.unk_idx = 0
        self.vocab_size = vocab_size

        # net params and architecture
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional,
                               batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        self.hidden2mean = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.hidden2logv = nn.Linear(hidden_size * self.hidden_factor, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)

    def forward(self, input_sequence: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                                                   torch.Tensor, torch.Tensor]:
        """
        Perform forward pass for the model
        :param input_sequence: preprocessed sentences (pad, sos, eos added)
        :param length: lengths of the sentences
        :return: tuple of four torch.Tensors:
            1st - prediction logits
            2nd - mean of variational posterior
            3rd - log variance of variational posterior
            4th - sample from variational posterior
        """
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # encoder
        input_embedding = self.embedding(input_sequence)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            # possibly incorrect, maybe need to permute
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # reparametrization
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = torch.randn([batch_size, self.latent_size]).to(self.new_device)
        z = z * std + mean

        # DECODER
        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob = prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)
        input_embedding = self.embedding_dropout(input_embedding)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

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

        return logp, mean, logv, z

    def inference(self, n: int = 4, z: torch.Tensor = None) -> torch.Tensor:
        """
        Create samples either pure or for interpolation
        :param n: number of the samples to be produced
        :param z: sample from variational posterior to be used
        :return: generated sequence
        """
        if z is None:
            batch_size = n
            z = torch.randn([batch_size, self.latent_size]).to(self.new_device)
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, device=self.new_device).long()  # all idx of batch
        # all idx of batch which are still generating
        sequence_running = torch.arange(0, batch_size, device=self.new_device).long()
        sequence_mask = torch.ones(batch_size, device=self.new_device).bool()
        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size, device=self.new_device).long()
        # generated_sequenced
        generations = torch.empty((batch_size, self.max_sequence_length), device=self.new_device).fill_(self.pad_idx).long()

        t = 0
        input_sequence = torch.empty(batch_size, device=self.new_device).fill_(self.sos_idx).long()
        while t < self.max_sequence_length and len(running_seqs) > 0:
            input_sequence = input_sequence.unsqueeze(1)
            input_embedding = self.embedding(input_sequence)
            output, hidden = self.decoder_rnn(input_embedding, hidden)
            logits = self.outputs2vocab(output)
            # sample
            _, input_sequence = torch.topk(logits, 1, dim=-1)
            input_sequence = input_sequence.squeeze()
            # save next input
            generations = save_sample(generations, input_sequence, sequence_running, t)
            # update global running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx)
            sequence_running = sequence_idx.masked_select(sequence_mask)
            print('kek', sequence_running)
            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)
            print('lol', running_seqs)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                hidden = hidden[:, running_seqs]
                running_seqs = torch.arange(0, len(running_seqs), device=self.new_device).long()
            t += 1

        return generations

    def prepare_data(self) -> None:
        """
        Load datasets
        """
        # defaults from args, TODO: remove to parameters
        data_dir = 'data'
        create_data = False
        splits = ['train', 'valid']
        max_sequence_length = 60
        min_occ = 1

        datasets = {}
        for split in splits:
            datasets[split] = PTB(data_dir=data_dir, split=split, create_data=create_data,
                                  max_sequence_length=max_sequence_length, min_occ=min_occ)
        self.ptb_train, self.ptb_val = datasets['train'], datasets['valid']
        self.sos_idx = self.ptb_train.sos_idx
        self.eos_idx = self.ptb_train.eos_idx
        self.pad_idx = self.ptb_train.pad_idx
        self.unk_idx = self.ptb_train.unk_idx

    def train_dataloader(self) -> DataLoader:
        """
        Create train dataloader from train dataset
        :return: pytorh dataloader
        """
        train_loader = DataLoader(dataset=self.ptb_train, batch_size=self.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=torch.cuda.is_available())
        self.len_train_loader = len(train_loader)
        return train_loader

    def val_dataloader(self) -> DataLoader:
        """
        Create train dataloader from train dataset
        :return: pytorh dataloader
        """
        val_loader = DataLoader(dataset=self.ptb_val, batch_size=self.batch_size, num_workers=4,
                                pin_memory=torch.cuda.is_available())
        return val_loader

    def kl_anneal_function(self, anneal_function: str) -> float:
        """
        Compute current KL divergence coefficient
        :param anneal_function:
        :return: coefficient
        """
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-self.k * (self.step - self.x0))))
        elif anneal_function == 'linear':
            return min(1., self.step / self.x0)

    def loss_fn(self, logp: torch.Tensor, target: torch.Tensor, length: torch.Tensor, mean: torch.Tensor,
                logv: torch.Tensor, anneal_function: str) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute loss
        :param logp: prediction logits tensor of shape [batch_size, batch_max_sentence_length, vocab_size]
        :param target: target tensor of shape [batch_size, max_sentence_length]
        :param length: sentences lengths tensor of shape [batch_size]
        :param mean: mean of the variational posterior of shape [batch_size]
        :param logv: log variance of the variational posterior of shape [batch_size]
        :param anneal_function: type on the annealing: either linear or logistic
        :return: tuple of reconstruction loss, KL loss and KL weight
        """
        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        nll = torch.nn.NLLLoss(ignore_index=self.pad_idx, reduction='sum')
        nll_loss = nll(logp, target)
        kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        kl_weight = self.kl_anneal_function(anneal_function)

        return nll_loss, kl_loss, kl_weight

    def training_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Perform trainig step
        :param batch: dict with keys (input, target, length); contains preprocessed (pad, sos, eos added) input tensor,
        target tensor and lengths of the sentences
        :param batch_idx: number of batch in current epoch
        :return: dict with training logs
        """
        batch_size = batch['input'].size(0)
        for key in batch:
            batch[key] = batch[key].to(self.new_device)

        # Forward pass
        logp, mean, logv, z = self.forward(batch['input'], batch['length'])

        # loss calculation
        nll_loss, kl_loss, kl_weight = self.loss_fn(logp, batch['target'], batch['length'], mean, logv, 'logistic')
        loss = (nll_loss + kl_weight * kl_loss) / batch_size
        self.step += 1

        if batch_idx % self.print_every == 0 or batch_idx + 1 == self.len_train_loader:
            print("TRAIN Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                  % (batch_idx, self.len_train_loader - 1, loss.item(), nll_loss.item() / batch_size,
                     kl_loss.item() / batch_size, kl_weight))

        logs = {'train_loss': loss}
        # TODO: add full logs to lightning
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self) -> Optimizer:
        """
        Configure optimizer for training
        :return: optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = PLSentenceVAE().cuda()
trainer = pl.Trainer(max_epochs=8, auto_select_gpus=True)
trainer.fit(model)
