import json

import numpy as np
import torch

import plmodel
import priors
from utils import idx2word, interpolate


def validate(model: plmodel.PLSentenceVAE, output_file, n_samples: int = 30) -> None:
    model.eval()

    with open("data/ptb.vocab.json", "r") as file:
        vocab = json.load(file)
    w2i, i2w = vocab["w2i"], vocab["i2w"]

    # reconstructions
    original_sentences_input = []
    original_sentences_length = []
    original_sentences_target = []
    for i in range(1, 1 + n_samples):
        original_sentences_input.append(model.ptb_train[i]["input"])
        original_sentences_length.append(model.ptb_train[i]["length"])
        original_sentences_target.append(model.ptb_train[i]["target"])

    original_sentences_input = torch.tensor(original_sentences_input).type(torch.long)
    original_sentences_length = torch.tensor(original_sentences_length).type(torch.long)
    original_sentences_target = torch.tensor(original_sentences_target).type(torch.long)

    logits, _, _, _ = model.forward(original_sentences_input, original_sentences_length)
    samples = torch.max(logits, dim=-1)[1]

    output_file.write("\n\n----------RECONSTRUCTIONS----------\n")
    originals_strings = idx2word(original_sentences_target, i2w=i2w, pad_idx=w2i["<pad>"], eos_idx=w2i["<eos>"])
    reconstructed_strings = idx2word(samples, i2w=i2w, pad_idx=w2i["<pad>"], eos_idx=w2i["<eos>"])

    for orig, recon in zip(originals_strings, reconstructed_strings):
        output_file.write("Original: " + orig + "\nReconstructed: " + recon + "\n\n")

    # mixture component"
    if isinstance(model.prior, priors.MoG):
        output_file.write("\n\n----------PSEUDO INPUTS----------\n")
        idx_list = np.random.choice(np.arange(100), size=40)
        for idx in idx_list:
            z_in = model.prior.generate_z(1, idx)
            samples = model.inference(z=z_in)
            for elem in idx2word(samples, i2w=i2w, pad_idx=w2i["<pad>"], eos_idx=w2i["<eos>"]):
                output_file.write(str(idx) + ": " + elem + "\n")

    samples = model.inference(batch_size=n_samples)
    output_file.write("\n\n----------SAMPLES----------\n")
    for elem in idx2word(samples, i2w=i2w, pad_idx=w2i["<pad>"], eos_idx=w2i["<eos>"]):
        output_file.write(elem + "\n")

    z1 = torch.randn([model.config.model.latent_size]).numpy()
    z2 = torch.randn([model.config.model.latent_size]).numpy()
    z_in = torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float()
    samples = model.inference(z=z_in)
    output_file.write("\n\n-------INTERPOLATION-------\n")
    for elem in idx2word(samples, i2w=i2w, pad_idx=w2i["<pad>"], eos_idx=w2i["<eos>"]):
        output_file.write(elem + "\n")
