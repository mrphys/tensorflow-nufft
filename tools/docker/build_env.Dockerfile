FROM tensorflow/tensorflow:custom-op-gpu-ubuntu16

RUN apt-get update && \
    apt-get install -y libfftw3-dev
