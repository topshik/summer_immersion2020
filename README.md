# Text generation with VAE

Plan:
1. Understant VAE approach: how to derive ELBO, how to obtain "optimal" prior (like they do in Vamp and Boo)
2. Why do we need VAE in NLP? Which tasks people solve with it, on which datasets
3. Launch experiments with simple RNN-based VAE (example - [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349)) for a baseline. Reference repo: https://github.com/timbmg/Sentence-VAE
4. Refactor architecture so that it is possible to change prior to sample from
5. Add MoG prior
6. Add VampPrior
