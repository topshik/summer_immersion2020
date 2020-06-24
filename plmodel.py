from collections import OrderedDict, defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
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
    def __init__(self, vocab_size: int, embedding_size: int, sos_idx: int, eos_idx: int, pad_idx: int, unk_idx: int,
                 max_sequence_length: int, rnn_type: str = 'rnn', latent_size: int = 256, hidden_size: int = 256,
                 word_dropout: float = 0, embedding_dropout: float = 0.5, num_layers: int = 1,
                 bidirectional: bool = False):
        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        self.step = 0  # needed for KL annealing
        self.ptb_train, self.ptb_val = None, None
        self.batch_size = 32
        self.len_train_loader = 0

        avail = torch.cuda.is_available()
        self.new_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size

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

    def forward(self, input_sequence, length):

        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)
        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            # possibly incorrect, maybe need to permute
            hidden = hidden.view(batch_size, self.hidden_size * self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
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

    def inference(self, n=4, z=None):
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

        return generations, z

    def prepare_data(self) -> None:
        # defaults from args
        data_dir = 'data'
        create_data = False
        splits = ['train', 'valid']
        max_sequence_length = 60
        min_occ = 1

        datasets = OrderedDict()
        for split in splits:
            datasets[split] = PTB(
                data_dir=data_dir,
                split=split,
                create_data=create_data,
                max_sequence_length=max_sequence_length,
                min_occ=min_occ
            )

        self.ptb_train, self.ptb_val = datasets['train'], datasets['valid']

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(dataset=self.ptb_train, batch_size=self.batch_size, shuffle=True,
                                  num_workers=4, pin_memory=torch.cuda.is_available())
        self.len_train_loader = len(train_loader)
        return train_loader

    def kl_anneal_function(self, anneal_function, k, x0):
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-k * (self.step - x0))))
        elif anneal_function == 'linear':
            return min(1, self.step / x0)

    def loss_fn(self, logp, target, length, mean, logv, anneal_function, k, x0):
        NLL = torch.nn.NLLLoss(ignore_index=self.pad_idx, reduction='sum')
        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)
        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = self.kl_anneal_function(anneal_function, k, x0)

        return NLL_loss, KL_loss, KL_weight

    def training_step(self, batch, batch_idx):
        # defaults from args
        k1 = 0.0025
        x0 = 2500
        print_every = 50
        tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        tracker = defaultdict(tensor)

        batch_size = batch['input'].size(0)
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = v.to(self.new_device)

        # Forward pass
        logp, mean, logv, z = self.forward(batch['input'], batch['length'])

        # loss calculation
        NLL_loss, KL_loss, KL_weight = self.loss_fn(logp, batch['target'], batch['length'], mean, logv, 'logistic',
                                                    k1, x0)
        loss = (NLL_loss + KL_weight * KL_loss) / batch_size
        self.step += 1

        # bookkeepeing
        tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data.view(1, -1)), dim=0)

        # if batch_idx % print_every == 0 or batch_idx + 1 == len(data_loader):
        #     print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
        #           % (split.upper(), batch_idx, len(data_loader) - 1, loss.item(), NLL_loss.item() / batch_size,
        #              KL_loss.item() / batch_size, KL_weight))

        if batch_idx % print_every == 0 or batch_idx + 1 == self.len_train_loader:
            print("TRAIN Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                  % (batch_idx, self.len_train_loader - 1, loss.item(), NLL_loss.item() / batch_size,
                     KL_loss.item() / batch_size, KL_weight))

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


splits = ['train', 'valid']
datasets = OrderedDict()
for split in splits:
    datasets[split] = PTB(
        data_dir='data',
        split=split,
        create_data=False,
        max_sequence_length=60,
        min_occ=1
    )

model = PLSentenceVAE(
    vocab_size=datasets['train'].vocab_size,
    sos_idx=datasets['train'].sos_idx,
    eos_idx=datasets['train'].eos_idx,
    pad_idx=datasets['train'].pad_idx,
    unk_idx=datasets['train'].unk_idx,
    max_sequence_length=60,
    embedding_size=300,
    rnn_type='rnn',
    hidden_size=256,
    word_dropout=0,
    embedding_dropout=0.5,
    latent_size=16,
    num_layers=1,
    bidirectional=False
    )

model.cuda()
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model)
