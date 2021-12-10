CXX := g++
NVCC := nvcc
PY_VERSION ?= 3.8
PYTHON = python$(PY_VERSION)

ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
FINUFFT_ROOT = tensorflow_nufft/cc/kernels/finufft/cpu
CUFINUFFT_ROOT = tensorflow_nufft/cc/kernels/finufft/gpu

KERNELS_DIR = tensorflow_nufft/cc/kernels
OPS_DIR = tensorflow_nufft/cc/ops

CUSOURCES = $(wildcard $(KERNELS_DIR)/*.cu.cc)
CUOBJECTS = $(patsubst %.cu.cc, %.cu.o, $(CUSOURCES))
CXXSOURCES = $(filter-out $(CUSOURCES), $(wildcard $(KERNELS_DIR)/*.cc)) $(wildcard $(OPS_DIR)/*.cc)
CXXHEADERS = $(wildcard $(KERNELS_DIR)/*.h) $(wildcard $(OPS_DIR)/*.h) 

TARGET_LIB = tensorflow_nufft/python/ops/_nufft_ops.so
TARGET_DLINK = tensorflow_nufft/cc/kernels/nufft_kernels.dlink.o

TF_CFLAGS := $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LDFLAGS := $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CUDA_INCLUDE = /usr/local/cuda/targets/x86_64-linux/include
CUDA_LIBDIR = /usr/local/cuda/targets/x86_64-linux/lib

CUDA ?= 1
OMP ?= 1
CFLAGS = -O3 -march=x86-64 -mtune=generic

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

FINUFFT_CFLAGS = -DFFTW_PLAN_SAFE -funroll-loops -fcx-limited-range
CUFINUFFT_CFLAGS = -funroll-loops


# ==============================================================================
# NVCC options
# ==============================================================================

# NVARCH ?= \
# 	-gencode=arch=compute_35,code=sm_35 \
# 	-gencode=arch=compute_50,code=sm_50 \
# 	-gencode=arch=compute_52,code=sm_52 \
# 	-gencode=arch=compute_60,code=sm_60 \
# 	-gencode=arch=compute_61,code=sm_61 \
# 	-gencode=arch=compute_70,code=sm_70 \
# 	-gencode=arch=compute_75,code=sm_75 \
# 	-gencode=arch=compute_80,code=sm_80 \
# 	-gencode=arch=compute_86,code=sm_86 \
# 	-gencode=arch=compute_86,code=compute_86

NVARCH ?= -gencode=arch=compute_61,code=sm_61

CUDAFE = --diag_suppress=174 --diag_suppress=611 --diag_suppress=20012 --display_error_number

CUFLAGS = $(NVARCH) -Xcompiler "$(CFLAGS)" $(TF_CFLAGS) -DNDEBUG --expt-relaxed-constexpr
CUFLAGS += -I$(ROOT_DIR)
CUFLAGS += -Xcudafe "$(CUDAFE)"

CUFINUFFT_CUFLAGS ?= -std=c++14 -ccbin=$(CXX) -O3 $(NVARCH) \
	-Wno-deprecated-gpu-targets --default-stream per-thread \
	-Xcompiler "$(CXXFLAGS)" --expt-relaxed-constexpr
CUFINUFFT_CUFLAGS += -I$(CUFINUFFT_ROOT)
CUFINUFFT_CUFLAGS += -Xcudafe "$(CUDAFE)"


# ==============================================================================
# Linker options
# ==============================================================================

LDFLAGS = $(TF_LDFLAGS)
LDFLAGS += -lfftw3 -lfftw3f

ifeq ($(OMP), 1)
LDFLAGS += -lgomp
LDFLAGS += -lfftw3_omp -lfftw3f_omp
endif

ifeq ($(CUDA), 1)
LDFLAGS += -L$(CUDA_LIBDIR)
LDFLAGS += -lcudart -lnvToolsExt
endif


# ==============================================================================
# FINUFFT
# ==============================================================================

FINUFFT_LIB = $(FINUFFT_ROOT)/libfinufft.a
FINUFFT_HEADERS = $(wildcard $(FINUFFT_ROOT)/*.h)

# spreader is subset of the library with self-contained testing, hence own objs:
# double-prec spreader object files that also need single precision...
SOBJS = $(FINUFFT_ROOT)/spreadinterp.o $(FINUFFT_ROOT)/utils.o
# their single-prec versions
SOBJSF = $(SOBJS:%.o=%_32.o)
# precision-dependent spreader object files (compiled & linked only once)...
SOBJS_PI = $(FINUFFT_ROOT)/utils_precindep.o
# spreader dual-precision objs
SOBJSD = $(SOBJS) $(SOBJSF) $(SOBJS_PI)

# double-prec library object files that also need single precision...
OBJS = $(SOBJS) $(FINUFFT_ROOT)/finufft.o
# their single-prec versions
OBJSF = $(OBJS:%.o=%_32.o)
# precision-dependent library object files (compiled & linked only once)...
OBJS_PI = $(SOBJS_PI)
# all lib dual-precision objs
OBJSD = $(OBJS) $(OBJSF) $(OBJS_PI)

finufft: $(FINUFFT_LIB)

$(FINUFFT_LIB): $(OBJSD)
	ar rcs $(FINUFFT_LIB) $(OBJSD)

# implicit rules for objects (note -o ensures writes to correct dir)
$(FINUFFT_ROOT)/%.o: $(FINUFFT_ROOT)/%.cpp $(FINUFFT_HEADERS)
	$(CXX) -c $(CXXFLAGS) $(FINUFFT_CFLAGS) $< -o $@
$(FINUFFT_ROOT)/%_32.o: $(FINUFFT_ROOT)/%.cpp $(FINUFFT_HEADERS)
	$(CXX) -DSINGLE -c $(CXXFLAGS) $(FINUFFT_CFLAGS) $< -o $@
$(FINUFFT_ROOT)/%.o: $(FINUFFT_ROOT)/%.c $(FINUFFT_HEADERS)
	$(CC) -c $(CFLAGS) $(FINUFFT_CFLAGS) $< -o $@
$(FINUFFT_ROOT)/%_32.o: $(FINUFFT_ROOT)/%.c $(FINUFFT_HEADERS)
	$(CC) -DSINGLE -c $(CFLAGS) $(FINUFFT_CFLAGS) $< -o $@


# ==============================================================================
# CUFINUFFT
# ==============================================================================

CUFINUFFT_LIB = $(CUFINUFFT_ROOT)/libcufinufft.a
CUFINUFFT_DLINK = $(CUFINUFFT_ROOT)/libcufinufft.dlink
CUFINUFFT_HEADERS = $(CUFINUFFT_ROOT)/cufinufft.h \
					$(CUFINUFFT_ROOT)/cudeconvolve.h \
					$(CUFINUFFT_ROOT)/profile.h \
					$(CUFINUFFT_ROOT)/cuspreadinterp.h \
					$(CUFINUFFT_ROOT)/cufinufft_eitherprec.h \
					$(CUFINUFFT_ROOT)/cufinufft_errors.h
CONTRIBOBJS=$(CUFINUFFT_ROOT)/contrib/spreadinterp.o \
			$(CUFINUFFT_ROOT)/contrib/utils_fp.o

# We create three collections of objects:
#  Double (_64), Single (_32), and floating point agnostic (no suffix)
CUFINUFFTOBJS=$(CUFINUFFT_ROOT)/precision_independent.o \
			  $(CUFINUFFT_ROOT)/profile.o \
			  $(CUFINUFFT_ROOT)/contrib/legendre_rule_fast.o \
			  $(CUFINUFFT_ROOT)/contrib/utils.o
CUFINUFFTOBJS_64=$(CUFINUFFT_ROOT)/spreadinterp2d.o \
				 $(CUFINUFFT_ROOT)/cufinufft2d.o \
				 $(CUFINUFFT_ROOT)/spread2d_wrapper.o \
				 $(CUFINUFFT_ROOT)/deconvolve_wrapper.o \
				 $(CUFINUFFT_ROOT)/cufinufft.o \
				 $(CUFINUFFT_ROOT)/spreadinterp3d.o \
				 $(CONTRIBOBJS)
CUFINUFFTOBJS_32=$(CUFINUFFTOBJS_64:%.o=%_32.o)


cufinufft: $(CUFINUFFT_LIB)

$(CUFINUFFT_LIB): $(CUFINUFFTOBJS) $(CUFINUFFTOBJS_64) $(CUFINUFFTOBJS_32) $(CONTRIBOBJS)
	$(NVCC) -dlink $(CUFINUFFT_CUFLAGS) $^ -o $(CUFINUFFT_DLINK)
	ar rcs $(CUFINUFFT_LIB) $^ $(CUFINUFFT_DLINK)

$(CUFINUFFT_ROOT)/%_32.o: $(CUFINUFFT_ROOT)/%.cpp $(CUFINUFFT_HEADERS)
	$(CXX) -DSINGLE -c $(CXXFLAGS) $(CUFINUFFT_CFLAGS) $< -o $@
$(CUFINUFFT_ROOT)/%_32.o: $(CUFINUFFT_ROOT)/%.c $(CUFINUFFT_HEADERS)
	$(CC) -DSINGLE -c $(CFLAGS) $(CUFINUFFT_CFLAGS) $< -o $@
$(CUFINUFFT_ROOT)/%_32.o: $(CUFINUFFT_ROOT)/%.cu $(CUFINUFFT_HEADERS)
	$(NVCC) -DSINGLE --device-c -c $(CUFINUFFT_CUFLAGS) $< -o $@
$(CUFINUFFT_ROOT)/%.o: $(CUFINUFFT_ROOT)/%.cpp $(CUFINUFFT_HEADERS)
	$(CXX) -c $(CXXFLAGS) $(CUFINUFFT_CFLAGS) $< -o $@
$(CUFINUFFT_ROOT)/%.o: $(CUFINUFFT_ROOT)/%.c $(CUFINUFFT_HEADERS)
	$(CC) -c $(CFLAGS) $(CUFINUFFT_CFLAGS) $< -o $@
$(CUFINUFFT_ROOT)/%.o: $(CUFINUFFT_ROOT)/%.cu $(CUFINUFFT_HEADERS)
	$(NVCC) --device-c -c $(CUFINUFFT_CUFLAGS) $< -o $@


# ==============================================================================
# TensorFlow NUFFT
# ==============================================================================

all: lib wheel

lib: $(TARGET_LIB)

%.cu.o: %.cu.cc
	$(NVCC) -ccbin $(CXX) -dc -x cu $(CUFLAGS) -o $@ -c $<

$(TARGET_DLINK): $(CUOBJECTS)
	$(NVCC) -ccbin $(CXX) -dlink $(CUFLAGS) -o $@ $^

$(TARGET_LIB): $(CXXSOURCES) $(CUOBJECTS) $(TARGET_DLINK) $(FINUFFT_LIB) $(CUFINUFFT_LIB)
	$(CXX) -shared $(CXXFLAGS) -o $@ $^ $(LDFLAGS)


# ==============================================================================
# Miscellaneous
# ==============================================================================

wheel:
	./tools/build/build_pip_pkg.sh make --python $(PYTHON) artifacts

test:
	$(PYTHON) -m unittest discover -v -p *_test.py

benchmark: $(wildcard tensorflow_nufft/python/ops/*.py) $(TARGET_LIB)
	$(PYTHON) tensorflow_nufft/python/ops/nufft_ops_test.py --benchmarks=.*

lint: $(wildcard tensorflow_nufft/python/ops/*.py)
	pylint --rcfile=pylintrc tensorflow_nufft/python

cpplint:
	python2.7 cpplint.py $(CXXSOURCES) $(CXXHEADERS)

docs: $(TARGET)
	ln -sf tensorflow_nufft tfft
	rm -rf tools/docs/_*
	$(MAKE) -C tools/docs html PY_VERSION=$(PY_VERSION)
	rm tfft

# Cleans only TensorFlow NUFFT additions.
clean:
	rm -f $(TARGET_LIB)
	rm -f $(TARGET_DLINK)
	rm -f $(CUOBJECTS)
	rm -rf artifacts/

# Cleans FINUFFT.
allclean: clean
	rm -f $(FINUFFT_LIB)
	rm -f $(FINUFFT_ROOT)/*.o $(FINUFFT_ROOT)/contrib/*.o
	rm -f $(CUFINUFFT_LIB)
	rm -f $(CUFINUFFT_DLINK)
	rm -f $(CUFINUFFT_ROOT)/*.o $(CUFINUFFT_ROOT)/contrib/*.o

.PHONY: all lib finufft wheel test benchmark lint docs clean allclean
