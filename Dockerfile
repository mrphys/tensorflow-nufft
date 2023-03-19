#syntax=docker/dockerfile:1.1.5-experimental
ARG PY_VERSION=3.8
ARG TF_VERSION=2.11
FROM tensorflow/build:$TF_VERSION-python$PY_VERSION as base_install

ENV TF_NEED_CUDA="1"
ARG PY_VERSION
ARG TF_VERSION

# TODO: Remove this if tensorflow/build container removes their keras-nightly install
# https://github.com/tensorflow/build/issues/78
RUN python -m pip uninstall -y keras-nightly

RUN python -m pip install --default-timeout=1000 tensorflow==$TF_VERSION

COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY ./ /tensorflow-nufft
WORKDIR /tensorflow-nufft


# ------------------------------------------------------------------------------
# Make wheel.
# ------------------------------------------------------------------------------
FROM base_install as make_wheel

RUN python configure.py

RUN bash tools/testing/build_and_run_tests.sh

RUN bazel build \
        --noshow_progress \
        --noshow_loading_progress \
        --verbose_failures \
        --test_output=errors \
        --crosstool_top=@ubuntu20.04-gcc9_manylinux2014-cuda11.2-cudnn8.1-tensorrt7.2_config_cuda//crosstool:toolchain \
        build_pip_pkg && \
    # Package Whl
    bazel-bin/build_pip_pkg artifacts

RUN bash tools/releases/tf_auditwheel_patch.sh
RUN python -m auditwheel repair --plat manylinux2014_x86_64 artifacts/*.whl
RUN ls -al wheelhouse/


# ------------------------------------------------------------------------------
# Dev container.
# ------------------------------------------------------------------------------
FROM base_install as dev_container

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


# ------------------------------------------------------------------------------
# Test wheel in fresh environment.
# ------------------------------------------------------------------------------
FROM python:$PY_VERSION as test_wheel_in_fresh_environment

ARG TF_VERSION

RUN python -m pip install --default-timeout=1000 tensorflow==$TF_VERSION

COPY --from=make_wheel /tensorflow-nufft/wheelhouse/ /tensorflow-nufft/wheelhouse/
RUN pip install /tensorflow-nufft/wheelhouse/*.whl

RUN python -c "import tensorflow_nufft as tfft"


# ------------------------------------------------------------------------------
# Build output.
# ------------------------------------------------------------------------------
FROM scratch as output

COPY --from=test_wheel_in_fresh_environment /tensorflow-nufft/wheelhouse/ .
