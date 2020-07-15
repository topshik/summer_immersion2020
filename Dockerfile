FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

## Base packages for ubuntu

# clean the libs list
RUN apt clean \
 && apt update -qq \
 && apt install software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt install -y --no-install-recommends \
    git \
    wget \
    bzip2 \
    vim \
    nano \
    g++ \
    make \
    cmake \
    build-essential \
    software-properties-common \
    apt-transport-https \
    sudo \
    gosu \
    libgl1-mesa-glx \
    graphviz \
    tmux \
    screen \
    htop \
    p7zip-full \
    python3.8 \
    virtualenv


# Create a non-root user and switch to it.
RUN adduser --disabled-password --gecos '' --shell /bin/bash user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory.
ENV HOME=/home/user
RUN chmod 777 /home/user

## Create a Python 3.8 environment.
RUN virtualenv .env --python=python3.8 \
 && . .env/bin/activate \
 && pip install requirements.txt
