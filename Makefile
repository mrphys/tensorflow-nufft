CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

# Root directory.
ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Dependencies.
FINUFFT_DIR := third_party/finufft
FINUFFT_LIB = $(FINUFFT_DIR)/lib-static/libfinufft.a

NUFFT_SRCS = $(wildcard tensorflow_nufft/cc/kernels/*.cc) $(wildcard tensorflow_nufft/cc/ops/*.cc)
TIME_TWO_SRCS = tensorflow_time_two/cc/kernels/time_two_kernels.cc $(wildcard tensorflow_time_two/cc/kernels/*.h) $(wildcard tensorflow_time_two/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11 
LDFLAGS = -shared ${TF_LFLAGS}

# Additional include directories.
CFLAGS += -I$(ROOT_DIR) # This Makefile's directory.
CFLAGS += -I$(ROOT_DIR)/$(FINUFFT_DIR)/include # For FINUFFT's relative includes
# CFLAGS += -Ithird_party/finufft/include

# Additional dynamic linking.
LDFLAGS += -lfftw3 -lfftw3_omp -lfftw3f -lfftw3f_omp

# Additional static linking.
LDFLAGS += $(FINUFFT_LIB)


TARGET_LIB = tensorflow_nufft/python/ops/_nufft_ops.so
# TIME_TWO_GPU_ONLY_TARGET_LIB = tensorflow_time_two/python/ops/_time_two_ops.cu.o
# TIME_TWO_TARGET_LIB = tensorflow_time_two/python/ops/_time_two_ops.so


# nufft op for CPU
ops: $(TARGET_LIB) 

$(TARGET_LIB): $(NUFFT_SRCS) $(FINUFFT_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

$(FINUFFT_LIB):
	$(MAKE) lib -C $(FINUFFT_DIR)

test: tensorflow_nufft/python/ops/nufft_ops_test.py tensorflow_nufft/python/ops/nufft_ops.py $(TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_nufft/python/ops/nufft_ops_test.py

pip_pkg: $(TARGET_LIB)
	./build_pip_pkg.sh make artifacts

clean:
	$(MAKE) clean -C $(FINUFFT_DIR)
	rm -f $(TARGET_LIB)

.PHONY: clean

# # time_two op for GPU
# time_two_gpu_only: $(TIME_TWO_GPU_ONLY_TARGET_LIB)

# $(TIME_TWO_GPU_ONLY_TARGET_LIB): tensorflow_time_two/cc/kernels/time_two_kernels.cu.cc
# 	$(NVCC) -std=c++11 -c -o $@ $^  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

# time_two_op: $(TIME_TWO_TARGET_LIB)
# $(TIME_TWO_TARGET_LIB): $(TIME_TWO_SRCS) $(TIME_TWO_GPU_ONLY_TARGET_LIB)
# 	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart

# time_two_test: tensorflow_time_two/python/ops/time_two_ops_test.py tensorflow_time_two/python/ops/time_two_ops.py $(TIME_TWO_TARGET_LIB)
# 	$(PYTHON_BIN_PATH) tensorflow_time_two/python/ops/time_two_ops_test.py

# clean:
# 	rm -f $(TARGET_LIB) $(TIME_TWO_GPU_ONLY_TARGET_LIB) $(TIME_TWO_TARGET_LIB)
