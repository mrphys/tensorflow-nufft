FROM tensorflow/tensorflow:custom-op-gpu-ubuntu16

# We need the FFTW3 library.
RUN apt-get update && \
    apt-get install -y libfftw3-dev

# To create modern manylinux wheels, we need a newer version of auditwheel than
# provided in the default image.
RUN pip install 'auditwheel >= 4.0.0'

# Add the TensorFlow framework library to the auditwheel whitelist. As a result
# auditwheel will be happy with the fact that this package links against
# TensorFlow.
RUN TF_SHARED_LIBRARY_NAME=libtensorflow_framework.so.2 && \
    AUDITWHEEL_POLICY_JSON=$(python -c 'import site; print(site.getsitepackages()[0])')/auditwheel/policy/policy.json && \
    sed -i "s/libresolv.so.2\"/libresolv.so.2\", \"$TF_SHARED_LIBRARY_NAME\"/g" $AUDITWHEEL_POLICY_JSON

# Ubuntu 16.04 has patchelf 0.9, which has a number of bugs. Install version
# 0.12 from source.
RUN cd /opt && \
    git clone https://github.com/NixOS/patchelf.git --branch 0.12 && \
    cd patchelf && \
    ./bootstrap.sh && \
    ./configure && \
    make && \
    make check && \
    make install
