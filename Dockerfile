FROM ubuntu:latest
MAINTAINER Cyrille Rossant "cyrille.rossant at gmail dot com"

RUN apt-get update && apt-get install -y \
    wget \
    git \
    curl \
    fontconfig \
    mesa-utils \
    binutils \
    libfontconfig1 \
    libsm6 \
    libXrender1 \
    libfreetype6 \
    libglib2.0-0 \
    xvfb

# Install miniconda and phy.
RUN wget -qO- http://phy.cortexlab.net/install/latest.sh | bash

# Install test dependencies.
RUN $HOME/miniconda/bin/conda install --yes \
    pytest \
    flake8 \
    requests && \
    $HOME/miniconda/bin/pip install --upgrade \
    pip \
    responses

ENV PATH=$HOME/miniconda/bin:$PATH
ENV DISPLAY=:99.0
WORKDIR $HOME
