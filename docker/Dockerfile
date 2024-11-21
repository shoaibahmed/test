FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04
LABEL org.opencontainers.image.authors="ssiddiqui@nvidia.com"
ENV LC_ALL=C
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /setup

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    clang \
    gcc \
    g++ \
    ninja-build \
    libc++-dev \
    libtbb-dev \
    autoconf \
    libtool \
    cmake \
    curl \
    git \
    zsh \
    tmux \
    vim \
    protobuf-compiler \
    python-is-python3 \
    python3-dev \
    python3-pip \
    sudo \
    vim-tiny \
    wget \
    libgeos-dev \
    htop && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN pip install \
    ninja \
    wget \
    matplotlib \
    natsort

RUN pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

RUN pip install \
    einops \
    opt_einsum \
    wandb

# Base HF packages
RUN pip install \
    transformers \
    datasets \
    accelerate \
    tokenizers \
    tiktoken

RUN pip install \
    peft \
    bitsandbytes

RUN git clone https://github.com/facebookresearch/xformers /setup/xformers/ && \
    cd /setup/xformers/ && \
    git submodule update --init --recursive && \
    python setup.py install && \
    rm -rf /setup/xformers/

RUN git clone https://github.com/Dao-AILab/flash-attention /setup/flash-attention/ && \
    cd /setup/flash-attention/ && \
    python setup.py install && \
    rm -rf /setup/flash-attention/

RUN pip install \
    git+https://github.com/shoaibahmed/lm-evaluation-harness

RUN pip install \
    torcheval
