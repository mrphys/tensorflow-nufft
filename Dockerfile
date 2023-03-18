#syntax=docker/dockerfile:1.1.5-experimental
ARG PY_VERSION=3.8
ARG TF_VERSION=2.11
FROM tensorflow/build:$TF_VERSION-python$PY_VERSION as base

ENV TF_NEED_CUDA="1"
ARG PY_VERSION
ARG TF_VERSION

# TODO: Remove this if tensorflow/build container removes their keras-nightly install
# https://github.com/tensorflow/build/issues/78
RUN python -m pip uninstall -y keras-nightly

RUN python -m pip install --default-timeout=1000 tensorflow==$TF_VERSION

# Dev container.
FROM base as dev

# Create non-root user.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    # Add user to sudoers.
    apt-get update && \
    apt-get install -y sudo && \
    echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    # Change default shell to bash.
    usermod --shell /bin/bash $USERNAME
