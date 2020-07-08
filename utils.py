import numpy as np


def idx2word(idx, i2w, pad_idx, eos_idx):
    sent_str = [str()] * len(idx)
    for i, sent in enumerate(idx):
        for word_id in sent:
            if word_id == pad_idx:
                break
            sent_str[i] += i2w[str(word_id.item())] + " "
            if word_id == eos_idx:
                break
        sent_str[i] = sent_str[i].strip()
    return sent_str


def interpolate(start, end, steps):
    interpolation = np.zeros((start.shape[0], steps + 2))
    for dim, (s, e) in enumerate(zip(start, end)):
        interpolation[dim] = np.linspace(s, e, steps+2)

    return interpolation.T
