CXX := /dt9/usr/bin/g++
NVCC := nvcc
PY_VERSION ?= 3.8
PYTHON = python$(PY_VERSION)

ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

KERNELS_DIR = tensorflow_nufft/cc/kernels
OPS_DIR = tensorflow_nufft/cc/ops
PROTO_DIR = tensorflow_nufft/proto
PYOPS_DIR = tensorflow_nufft/python/ops

# Protocol buffer source files (*.proto files).
PROTO_SOURCES = $(wildcard $(PROTO_DIR)/*.proto)
# protoc generated files (*.pb.h and *.pb.cc files).
PROTO_OBJECTS = $(patsubst $(PROTO_DIR)/%.proto, $(PROTO_DIR)/%.pb.cc, $(PROTO_SOURCES))
PROTO_HEADERS = $(patsubst $(PROTO_DIR)/%.proto, $(PROTO_DIR)/%.pb.h, $(PROTO_SOURCES))
# protoc generated files (*_pb2.py files).
PROTO_MODULES = $(patsubst $(PROTO_DIR)/%.proto, $(PROTO_DIR)/%_pb2.py, $(PROTO_SOURCES))

CUSOURCES = $(wildcard $(KERNELS_DIR)/*.cu.cc)
CUOBJECTS = $(patsubst %.cu.cc, %.cu.o, $(CUSOURCES))
CXXSOURCES = $(filter-out $(CUSOURCES), $(wildcard $(KERNELS_DIR)/*.cc) $(wildcard $(OPS_DIR)/*.cc))
CXXHEADERS = $(wildcard $(KERNELS_DIR)/*.h) $(wildcard $(OPS_DIR)/*.h)

TARGET_LIB = tensorflow_nufft/python/ops/_nufft_ops.so
TARGET_DLINK = tensorflow_nufft/cc/kernels/nufft_kernels.dlink.o

TF_CFLAGS := $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LDFLAGS := $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CUDA_INCLUDE = /usr/local/cuda/targets/x86_64-linux/include
CUDA_LIBDIR = /usr/local/cuda/targets/x86_64-linux/lib

CUDA ?= 1
OMP ?= 1
CFLAGS = -O3 -march=x86-64 -mtune=generic -funroll-loops -fcx-limited-range

-include make.inc

CFLAGS += -fPIC

ifeq ($(CUDA), 1)
TF_CFLAGS += -DGOOGLE_CUDA=1
endif

ifeq ($(OMP), 1)
CFLAGS += -fopenmp
endif

CXXFLAGS = -std=c++14 $(CFLAGS) $(TF_CFLAGS)
CXXFLAGS += -I$(ROOT_DIR)
ifeq ($(CUDA), 1)
CXXFLAGS += -I$(CUDA_INCLUDE)
endif

# As of TensorFlow 2.7, a deprecated-declarations is triggered by TensorFlow's
# header files, which we can't do anything about. Therefore, disable these
# warnings.
CXXFLAGS += -Wno-deprecated-declarations

# ==============================================================================
# NVCC options
# ==============================================================================

NVARCH_FLAGS ?= \
	-gencode=arch=compute_35,code=sm_35 \
	-gencode=arch=compute_50,code=sm_50 \
	-gencode=arch=compute_52,code=sm_52 \
	-gencode=arch=compute_60,code=sm_60 \
	-gencode=arch=compute_61,code=sm_61 \
	-gencode=arch=compute_70,code=sm_70 \
	-gencode=arch=compute_75,code=sm_75 \
	-gencode=arch=compute_80,code=sm_80 \
	-gencode=arch=compute_86,code=sm_86 \
	-gencode=arch=compute_86,code=compute_86

CUDAFE = --diag_suppress=174 --diag_suppress=177 --diag_suppress=611 --diag_suppress=20012 --diag_suppress=1886 --display_error_number

CUFLAGS = $(NVARCH_FLAGS) -Xcompiler "$(CFLAGS)" $(TF_CFLAGS) -DNDEBUG --expt-relaxed-constexpr
CUFLAGS += -I$(ROOT_DIR)
CUFLAGS += -Xcudafe "$(CUDAFE)" -Wno-deprecated-gpu-targets


# ==============================================================================
# Linker options
# ==============================================================================

LDFLAGS = -lfftw3 -lfftw3f

ifeq ($(OMP), 1)
LDFLAGS += -lgomp
LDFLAGS += -lfftw3_omp -lfftw3f_omp
endif

ifeq ($(CUDA), 1)
LDFLAGS += -L$(CUDA_LIBDIR)
LDFLAGS += -lcudart_static
endif

LDFLAGS += $(TF_LDFLAGS)

# ==============================================================================
# TensorFlow NUFFT
# ==============================================================================

all: lib wheel

lib: proto $(TARGET_LIB)

%.cu.o: %.cu.cc
	$(NVCC) -ccbin $(CXX) -dc -x cu $(CUFLAGS) -t 0 -o $@ -c $<

$(TARGET_DLINK): $(CUOBJECTS)
	$(NVCC) -ccbin $(CXX) -dlink $(CUFLAGS) -t 0 -o $@ $^

$(TARGET_LIB): $(CXXSOURCES) $(PROTO_OBJECTS) $(CUOBJECTS) $(TARGET_DLINK)
	$(CXX) -shared $(CXXFLAGS) -o $@ $^ $(LDFLAGS)


# ==============================================================================
# Miscellaneous
# ==============================================================================

proto:
	protoc -I$(PROTO_DIR) --python_out=$(PROTO_DIR) --cpp_out=$(PROTO_DIR) $(PROTO_SOURCES)

wheel:
	./tools/build/build_pip_pkg.sh make --python $(PYTHON) artifacts

test:
	$(PYTHON) -m unittest discover -v -p *_test.py

benchmark: $(wildcard tensorflow_nufft/python/ops/*.py) $(TARGET_LIB)
	$(PYTHON) tensorflow_nufft/python/ops/nufft_ops_test.py --benchmarks=.*

lint: $(wildcard tensorflow_nufft/python/ops/*.py)
	pylint --rcfile=pylintrc tensorflow_nufft/python

cpplint:
	python2.7 tools/lint/cpplint.py $(CXXSOURCES) $(CUSOURCES) $(CXXHEADERS)

docs: $(TARGET)
	rm -rf docs/_* docs/api_docs/tfft/
	$(MAKE) -C docs dirhtml PY_VERSION=$(PY_VERSION)

# Cleans compiled objects.
clean:
	rm -f $(TARGET_LIB)
	rm -f $(TARGET_DLINK)
	rm -f $(CUOBJECTS)
	rm -f $(PROTO_OBJECTS) $(PROTO_HEADERS) $(PROTO_MODULES)
	rm -rf artifacts/

.PHONY: all lib proto wheel test benchmark lint docs clean allclean
