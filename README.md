# Text generation with VAE

Plan:
1. Understant VAE approach: how to derive ELBO, how to obtain "optimal" prior (like they do in Vamp and Boo)
2. Why do we need VAE in NLP? Which tasks people solve with it, on which datasets
3. Launch experiments with simple RNN-based VAE (example - [Generating Sentences from a Continuous Space](https://arxiv.org/abs/1511.06349)) for a baseline. Reference repo: https://github.com/timbmg/Sentence-VAE
4. Refactor architecture so that it is possible to change prior to sample from
5. Add MoG prior
6. Add VampPrior

# Usage
## Docker setup
#### If you don't want to run app in docker, you can proceed to packages section
Dockerfile describes Nvidia Ubuntu 18.04 docker image with python3.8 and some basic packages installation.

### Optional:
You may want to build and run docker container in a tmux terminal, to detach from it and connect later:
```angular2
sudo apt update && apt upgrade
apt install -y tmux
tmux new -s <some_name>
``` 
### Build and run
Run following commands to build docker image and then run a container:
```
. docker-build.sh
. docker-run.sh
```

If you used tmux, you can now detach with `ctrl+b d` and create further connections to the container via
```angular2
docker exec -it <container_hash> /bin/bash
```

## Environment and packages
If you used docker build, then you can enable created python environment from inside of the container with
```angular2
. ../.env/bin/activate
```

Otherwise, you need to create new virtual environment with your favourite tool and install dependencies, e.g.
```angular2
virtualenv .env --python=python3.8
. .env/bin/activate
pip install -r requirements.txt
```

## Use the model
All training parameters are stored in the `train-config.yaml` file, so you can edit it according to your needs and run
```angular2
python train.py
```

Checkpoint for each epoch will be saved in the `checkpoints` directory. 
You can also watch logs with tensorboard:
```angular2
tensorboard --logdir lightning_logs
```
or
```angular2
tensorboard --logdir lightning_logs --bind_all
```
if you run tensorboard from inside of the docker container.