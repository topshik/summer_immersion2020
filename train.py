import json

import pytorch_lightning as pl
import torch

import plmodel
from utils import idx2word, interpolate


def train(prior: str, n_components: int):
    model = plmodel.PLSentenceVAE(prior=prior, n_components=n_components, batch_size=256, hidden_size=191,
                                  embedding_size=353, latent_size=13, word_dropout=0.62)
    trainer = pl.Trainer(max_epochs=13, gpus=1, auto_select_gpus=True)
    trainer.fit(model)


def validate(model_path: str, prior: str, n_components: int):
    model = plmodel.PLSentenceVAE(prior=prior, n_components=n_components, batch_size=1, hidden_size=191,
                                  embedding_size=353, latent_size=13, word_dropout=0.62)
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()
    # TODO: create separate function
    with open('data/ptb.vocab.json', 'r') as file:
        vocab = json.load(file)
    w2i, i2w = vocab['w2i'], vocab['i2w']

    n_reconstructions = 15

    original_sentences_input = []
    original_sentences_length = []
    original_sentences_target = []
    for i in range(1, 1 + n_reconstructions):
        original_sentences_input.append(model.ptb_train[i]['input'])
        original_sentences_length.append(model.ptb_train[i]['length'])
        original_sentences_target.append(model.ptb_train[i]['target'])

    original_sentences_input = torch.tensor(original_sentences_input).type(torch.long)
    original_sentences_length = torch.tensor(original_sentences_length).type(torch.long)
    original_sentences_target = torch.tensor(original_sentences_target).type(torch.long)

    logits, _, _, _ = model.forward(original_sentences_input.cuda(), original_sentences_length.cuda())
    samples = torch.max(logits, dim=-1)[1]

    print('\n----------RECONSTRUCTIONS----------')
    originals_strings = idx2word(original_sentences_target, i2w=i2w, pad_idx=w2i['<pad>'], eos_idx=w2i['<eos>'])
    reconstructed_strings = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'], eos_idx=w2i['<eos>'])

    for orig, recon in zip(originals_strings, reconstructed_strings):
        print('Original: ', orig, '\nReconstructed: ', recon, end='\n\n')

    # print('\\----------PSEUDO INPUTS----------')
    # idx_list = np.random.choice(np.arange(500), size=20)
    # for idx in idx_list:
    #     z_in = model.prior.generate_z(1, idx)
    #     samples = model.inference(z=z_in)
    #     print(idx, ': ', *idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'], eos_idx=w2i['<eos>']), end='\n')

    n_samples = 10
    samples = model.inference(batch_size=n_samples)
    print('\n----------SAMPLES----------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'], eos_idx=w2i['<eos>']), sep='\n')

    z1 = torch.randn([model.latent_size]).numpy()
    z2 = torch.randn([model.latent_size]).numpy()
    z_in = torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float().cuda()
    samples = model.inference(z=z_in)
    print('\n-------INTERPOLATION-------')
    print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'], eos_idx=w2i['<eos>']), sep='\n')