import json
import os
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils

import priors
from ptb import PTB
from utils import idx2word, interpolate


class PLSentenceVAE(pl.LightningModule):
    def __init__(self, prior: str = 'SimpleGaussian', batch_size: int = 256, vocab_size: int = 9877,
                 k: int = 0.0025, x0: int = 2500, embedding_size: int = 300, max_sequence_length: int = 50,
                 rnn_type: str = 'gru', latent_size: int = 96, hidden_size: int = 256, word_dropout: float = 0,
                 embedding_dropout: float = 0.5, num_layers: int = 1, bidirectional: bool = False,
                 print_every: int = 50) -> None:
        super().__init__()
        # kl annealing params
        self.step = 0
        self.epoch = 1
        self.k = k
        self.x0 = x0

        # datasets and their params, are set in prepare_data()
        self.batch_size = batch_size
        self.len_train_loader, self.len_val_loader = 0, 0
        self.ptb_train, self.ptb_val = None, None

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

        if prior == 'SimpleGaussian':
            self.prior = priors.SimpleGaussian(self.batch_size, self.latent_size)
        else:
            raise ValueError()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
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
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # encoder
        input_embedding = self.embedding(input_sequence)
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
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = torch.randn([self.batch_size, self.latent_size]).to(self.device)
        z = z * std + mean

        # DECODER
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

        return logp, mean, logv, z

    def inference(self, batch_size: int = 4, z: torch.Tensor = None) -> torch.Tensor:
        """
        Create samples either pure or for interpolation
        :param batch_size: number of the samples to be produced
        :param z: sample from variational posterior to be used
        :return: generated sequence
        """
        if z is None:
            z = self.prior.generate_z(batch_size=batch_size, latent_size=self.latent_size).to(self.device)
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

    def prepare_data(self) -> None:
        """
        Load datasets
        """
        # defaults from args, TODO: remove to parameters
        data_dir = 'data'
        create_data = False
        splits = ['train', 'valid']
        min_occ = 1

        datasets = {}
        for split in splits:
            datasets[split] = PTB(data_dir=data_dir, split=split, create_data=create_data,
                                  max_sequence_length=self.max_sequence_length, min_occ=min_occ)
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

    def loss_fn(self, z, logp: torch.Tensor, target: torch.Tensor, length: torch.Tensor, mean: torch.Tensor,
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
        # kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        kl_loss = self.kl_loss_mc(z, mean, logv)
        kl_weight = self.kl_anneal_function(anneal_function)

        return nll_loss, kl_loss, kl_weight

    def kl_loss_mc(self, z: torch.Tensor, mean, logv):
        log_q_z = priors.log_normal_diag(z, mean, logv, dim=1)
        log_p_z = self.prior.log_p_z(z)

        print('logs: ', log_q_z.shape, log_p_z.shape)

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
        logp, mean, logv, z = self.forward(batch['input'], batch['length'])

        # loss calculation
        nll_loss, kl_loss, kl_weight = self.loss_fn(z, logp, batch['target'], batch['length'], mean, logv, 'logistic')
        loss = (nll_loss + kl_weight * kl_loss) / self.batch_size
        self.step += 1

        return {'loss': loss, 'NLL loss': nll_loss.data, 'KL loss': kl_loss.data, 'KL weight': kl_weight}

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        """
        Perform validation step
        :param batch: dict with keys (input, target, length); contains preprocessed (pad, sos, eos added) input tensor,
        target tensor and lengths of the sentences
        :param batch_idx: number of batch in current epoch
        :return: dict with training logs
        """
        # Forward pass
        logp, mean, logv, z = self.forward(batch['input'], batch['length'])

        # loss calculation
        nll_loss, kl_loss, kl_weight = self.loss_fn(z, logp, batch['target'], batch['length'], mean, logv, 'logistic')
        loss = (nll_loss + kl_weight * kl_loss) / self.batch_size

        return {'ELBO': loss.data, 'NLL loss': nll_loss.data, 'KL loss': kl_loss.data, 'KL weight': kl_weight}

    def validation_epoch_end(self, outputs):
        """
        Saves model, logs losses at the end of validation epoch
        :param outputs: list of validation step outputs for all batches
        :return: dict of averaged logs
        """
        # TODO lightning saving interface
        save_model_path = 'checkpoints'
        checkpoint_path = os.path.join(save_model_path, "E%i.pth" % self.epoch)
        self.epoch += 1
        torch.save(model.state_dict(), checkpoint_path)
        print("\nModel saved at %s" % checkpoint_path)

        avg_elbo = torch.stack([x['ELBO'] for x in outputs]).mean()
        avg_nll = torch.stack([x['NLL loss'] for x in outputs]).mean()
        avg_kl = torch.stack([x['KL loss'] for x in outputs]).mean()
        tensorboard_logs = {'ELBO': avg_elbo, 'NLL loss': avg_nll, 'KL loss': avg_kl}
        return {'val_loss': avg_elbo, 'NLL loss': avg_nll, 'KL loss': avg_kl, 'log': tensorboard_logs}

    def configure_optimizers(self) -> Optimizer:
        """
        Configure optimizer for training
        :return: optimizer
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# training
model = PLSentenceVAE()
trainer = pl.Trainer(max_epochs=10, gpus=1, auto_select_gpus=True)
trainer.fit(model)
print('Training ended\n')

# inference
# path = 'checkpoints/E11.pth'
# model = PLSentenceVAE()
# model.prepare_data()
# model.load_state_dict(torch.load(path))
# model.cuda()


model.eval()
with open('data/ptb.vocab.json', 'r') as file:
    vocab = json.load(file)

w2i, i2w = vocab['w2i'], vocab['i2w']

n_samples = 10
samples = model.inference(batch_size=n_samples)
print('----------SAMPLES----------')
print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

z1 = torch.randn([model.latent_size]).numpy()
z2 = torch.randn([model.latent_size]).numpy()
z_in = torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float().cuda()
samples = model.inference(z=z_in)
print('-------INTERPOLATION-------')
print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
