CXX := g++
NVCC := nvcc
PY_VERSION ?= 3.8
PYTHON = python$(PY_VERSION)

ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
FINUFFT_INCLUDE = /dt7/usr/include/finufft/include
CUFINUFFT_INCLUDE = /dt7/usr/include/cufinufft/include

KERNELS_DIR = tensorflow_nufft/cc/kernels
OPS_DIR = tensorflow_nufft/cc/ops

CUSOURCES = $(wildcard $(KERNELS_DIR)/*.cu.cc)
CUOBJECTS = $(patsubst %.cu.cc, %.cu.o, $(CUSOURCES))
CXXSOURCES = $(filter-out $(CUSOURCES), $(wildcard $(KERNELS_DIR)/*.cc)) $(wildcard $(OPS_DIR)/*.cc)

TARGET_LIB = tensorflow_nufft/python/ops/_nufft_ops.so
TARGET_DLINK = tensorflow_nufft/cc/kernels/nufft_kernels.dlink.o

TF_CFLAGS := $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LDFLAGS := $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CUDA_INCLUDE = /usr/local/cuda/targets/x86_64-linux/include
CUDA_LIBDIR = /usr/local/cuda/targets/x86_64-linux/lib

CUDA ?= 1
CFLAGS = -O3 -march=x86-64 -mtune=generic

-include make.inc

CFLAGS += -fPIC -std=c++11

ifeq ($(CUDA), 1)
TF_CFLAGS += -DGOOGLE_CUDA=1
endif

CXXFLAGS = $(CFLAGS) $(TF_CFLAGS)
CXXFLAGS += -I$(FINUFFT_INCLUDE)
ifeq ($(CUDA), 1)
CXXFLAGS += -I$(CUDA_INCLUDE)
endif

CUFLAGS = -Xcompiler "$(CFLAGS)" $(TF_CFLAGS) -DNDEBUG --expt-relaxed-constexpr
CUFLAGS += -I$(CUFINUFFT_INCLUDE)

LDFLAGS = $(TF_LDFLAGS)
LDFLAGS += -lfinufft
ifeq ($(CUDA), 1)
LDFLAGS += -lcufinufft
endif
LDFLAGS += -lfftw3 -lfftw3_omp -lfftw3f -lfftw3f_omp
LDFLAGS += -lgomp
ifeq ($(CUDA), 1)
LDFLAGS += -L$(CUDA_LIBDIR)
LDFLAGS += -lcudart -lnvToolsExt
endif

all: lib wheel

lib: $(TARGET_LIB)

%.cu.o: %.cu.cc
	$(NVCC) -ccbin $(CXX) -arch=sm_61 -dc -x cu $(CUFLAGS) -o $@ -c $<

$(TARGET_DLINK): $(CUOBJECTS)
	$(NVCC) -ccbin $(CXX) -arch=sm_61 -dlink $(CUFLAGS) -o $@ $^

$(TARGET_LIB): $(CXXSOURCES) $(CUOBJECTS) $(TARGET_DLINK)
	$(CXX) -shared $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

wheel:
	./tools/build/build_pip_pkg.sh make --python $(PYTHON) artifacts

test:
	$(PYTHON) tensorflow_nufft/python/ops/nufft_ops_test.py

benchmark: $(wildcard tensorflow_nufft/python/ops/*.py) $(TARGET_LIB)
	$(PYTHON) tensorflow_nufft/python/ops/nufft_ops_test.py --benchmarks=.*

lint: $(wildcard tensorflow_nufft/python/ops/*.py)
	pylint --rcfile=pylintrc tensorflow_nufft/python

docs: $(TARGET)
	$(MAKE) -C tools/docs html

clean:
	rm -f $(TARGET_LIB)
	rm -f $(TARGET_DLINK)
	rm -f $(CUOBJECTS)
	rm -rf artifacts/

.PHONY: all lib wheel test benchmark lint docs clean
