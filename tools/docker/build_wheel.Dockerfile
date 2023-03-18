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
FROM base_install as tfa_gpu_tests
CMD ["bash", "tools/testing/build_and_run_tests.sh"]

# ------------------------------------------------------------------------------
FROM base_install as make_wheel

RUN python configure.py

# Test
RUN bash tools/testing/build_and_run_tests.sh

# Build
RUN bazel build \
        --noshow_progress \
        --noshow_loading_progress \
        --verbose_failures \
        --test_output=errors \
        --crosstool_top=@ubuntu20.04-gcc9_manylinux2014-cuda11.2-cudnn8.1-tensorrt7.2_config_cuda//crosstool:toolchain \
        build_pip_pkg && \
    # Package Whl
    bazel-bin/build_pip_pkg artifacts $NIGHTLY_FLAG

RUN bash tools/releases/tf_auditwheel_patch.sh
RUN python -m auditwheel repair --plat manylinux2014_x86_64 artifacts/*.whl
RUN ls -al wheelhouse/

# -------------------------------------------------------------------

FROM python:$PY_VERSION as test_wheel_in_fresh_environment

ARG TF_VERSION
ARG SKIP_CUSTOM_OP_TESTS

RUN python -m pip install --default-timeout=1000 tensorflow==$TF_VERSION

COPY --from=make_wheel /tensorflow-nufft/wheelhouse/ /tensorflow-nufft/wheelhouse/
RUN pip install /tensorflow-nufft/wheelhouse/*.whl

RUN if [[ -z "$SKIP_CUSTOM_OP_TESTS" ]] ; then python -c "import tensorflow_nufft as tfft" ; else python -c "import tensorflow_nufft as tfft" ; fi

# -------------------------------------------------------------------
FROM scratch as output

COPY --from=test_wheel_in_fresh_environment /tensorflow-nufft/wheelhouse/ .