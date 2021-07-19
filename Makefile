CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

# Root directory.
ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# Dependencies.
FINUFFT_DIR_CPU := third_party/finufft
FINUFFT_LIB_CPU = $(FINUFFT_DIR_CPU)/lib-static/libfinufft.a
FINUFFT_DIR_GPU := third_party/cufinufft
FINUFFT_LIB_GPU = $(FINUFFT_DIR_GPU)/lib-static/libcufinufft.a

NUFFT_SRCS_GPU = $(wildcard tensorflow_nufft/cc/kernels/*.cu.cc)
NUFFT_OBJS_GPU = $(patsubst %.cu.cc, %.cu.o, $(NUFFT_SRCS_GPU))
NUFFT_SRCS = $(filter-out $(NUFFT_SRCS_GPU), $(wildcard tensorflow_nufft/cc/kernels/*.cc)) $(wildcard tensorflow_nufft/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
TF_INCLUDE := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')

CUDA_INCLUDE = /usr/local/cuda/targets/x86_64-linux/include
CUDA_LIBDIR = /usr/local/cuda/targets/x86_64-linux/lib

# GCC-specific compilation flags.
CCFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
CCFLAGS += -I$(ROOT_DIR)/$(FINUFFT_DIR_CPU)/include
CCFLAGS += -I$(ROOT_DIR)/$(FINUFFT_DIR_GPU)/include
CCFLAGS += -DGOOGLE_CUDA=1
CCFLAGS += -I$(CUDA_INCLUDE)
CCFLAGS += -L$(CUDA_LIBDIR)

# NVCC-specific compilation flags.
CUFLAGS = $(TF_CFLAGS) -std=c++11 -DGOOGLE_CUDA=1 -x cu -Xcompiler "-fPIC" -DNDEBUG --expt-relaxed-constexpr
CUFLAGS += -I$(ROOT_DIR)/$(FINUFFT_DIR_GPU)/include

# Include this Makefile's directory.
CCFLAGS += -I$(ROOT_DIR)
CUFLAGS += -I$(ROOT_DIR)

# Linker flags.
LDFLAGS = -shared ${TF_LFLAGS}

# Additional dynamic linking.
LDFLAGS += -lfftw3 -lfftw3_omp -lfftw3f -lfftw3f_omp
LDFLAGS += -lcudadevrt -lcudart -lnvToolsExt

TARGET_LIB = tensorflow_nufft/python/ops/_nufft_ops.so
TARGET_LIB_GPU = tensorflow_nufft/python/ops/_nufft_ops.cu.o

all: op

op: $(TARGET_LIB)

.PHONY: test
test: $(wildcard tensorflow_nufft/python/ops/*.py) $(TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_nufft/python/ops/nufft_ops_test.py

.PHONY: lint
lint: $(wildcard tensorflow_nufft/python/ops/*.py)
	pylint --rcfile=pylintrc tensorflow_nufft/python

pip_pkg: $(TARGET_LIB)
	./build_pip_pkg.sh make artifacts

.PHONY: clean
clean: mostlyclean
	$(MAKE) clean -C $(FINUFFT_DIR_CPU)
	$(MAKE) clean -C $(FINUFFT_DIR_GPU)

.PHONY: mostlyclean
mostlyclean:
	rm -f $(TARGET_LIB)
	rm -f $(TARGET_LIB_GPU)
	rm -f $(NUFFT_OBJS_GPU)

.PHONY: install_dependencies
install_dependencies:
	apt-get update
	apt-get -y install libfftw3-dev

$(TARGET_LIB): $(NUFFT_SRCS) $(TARGET_LIB_GPU) $(FINUFFT_LIB_CPU) $(FINUFFT_LIB_GPU)
	$(CXX) $(CCFLAGS) -o $@ $^ $(NUFFT_OBJS_GPU) ${LDFLAGS}

$(TARGET_LIB_GPU): $(FINUFFT_LIB_GPU) $(NUFFT_SRCS_GPU)
	mkdir -p $(TF_INCLUDE)/third_party/gpus/cuda/include
	cp -r $(CUDA_INCLUDE)/* $(TF_INCLUDE)/third_party/gpus/cuda/include
	$(NVCC) -dc $(filter-out $<, $^) $(CUFLAGS) -odir tensorflow_nufft/cc/kernels -Xcompiler "-fPIC" -lcudadevrt -lcudart
	$(NVCC) -dlink $(NUFFT_OBJS_GPU) $(FINUFFT_LIB_GPU) -o $(TARGET_LIB_GPU) -Xcompiler "-fPIC" -lcudadevrt -lcudart

$(FINUFFT_LIB_CPU): $(FINUFFT_DIR_CPU)
	$(MAKE) lib -C $(FINUFFT_DIR_CPU)

$(FINUFFT_LIB_GPU): $(FINUFFT_DIR_GPU)
	$(MAKE) lib -C $(FINUFFT_DIR_GPU)

$(FINUFFT_DIR_CPU): | clone_third_party

$(FINUFFT_DIR_GPU): | clone_third_party

.PHONY: clone_third_party
clone_third_party:
	if [ ! -d $(FINUFFT_DIR_CPU) ]; 											\
	then 																		\
		git clone https://github.com/mrphys/finufft.git $(FINUFFT_DIR_CPU);		\
	fi

	if [ ! -d $(FINUFFT_DIR_GPU) ]; 															\
	then 																						\
		git clone https://github.com/mrphys/cufinufft.git $(FINUFFT_DIR_GPU) --branch release;	\
	fi
