set -e -x

if [ "$TF_NEED_CUDA" == "1" ]; then
  CUDA_FLAG="--crosstool_top=@ubuntu20.04-gcc9_manylinux2014-cuda11.2-cudnn8.1-tensorrt7.2_config_cuda//crosstool:toolchain"
fi

bazel build $CUDA_FLAG //tensorflow_nufft/...
cp ./bazel-bin/tensorflow_nufft/python/ops/_nufft_ops.so ./tensorflow_nufft/python/ops/
cp ./bazel-bin/tensorflow_nufft/proto/nufft_options_pb2.py ./tensorflow_nufft/proto/
