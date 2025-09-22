CXX := /usr/bin/g++-13
NVCC := nvcc --compiler-bindir=/usr/bin/g++-13

# PYTHON (already set earlier)
PY_VERSION ?= 3.9
PYTHON ?= /opt/python/cp39-cp39/bin/python

# Try to find site-packages using python but do not import tensorflow.
SITE_PACKAGES := $(shell $(PYTHON) -c 'import site,sys; \
if hasattr(site,"getsitepackages"): print(site.getsitepackages()[0]); \
else: print(next(p for p in sys.path if "site-packages" in p));' 2>/dev/null || echo "")

# TensorFlow include directory inside site-packages (do NOT import tensorflow)
TF_INCLUDE_SITE := $(SITE_PACKAGES)/tensorflow/include

# If the standard location doesn't exist, check a fallback that manylinux uses:
ifeq ($(wildcard $(TF_INCLUDE_SITE)),)
  ifneq ($(wildcard /opt/python/cp39-cp39/lib/python3.9/site-packages/tensorflow/include),)
    TF_INCLUDE_SITE := /opt/python/cp39-cp39/lib/python3.9/site-packages/tensorflow/include
  endif
endif

# Warn at make time if include dir missing (helps debugging), but don't try to import TF.
ifeq ($(wildcard $(TF_INCLUDE_SITE)),)
  $(warning WARNING: tensorflow include dir not found at $(TF_INCLUDE_SITE). \
  You may need to install tensorflow-cpu in the build image or adjust PYTHON.)
endif

# TF compile/link flags: use include dir and site-packages layout (do NOT import TF).
TF_CFLAGS := -I$(TF_INCLUDE_SITE) -I$(TF_INCLUDE_SITE)/external/eigen_archive \
             -I$(TF_INCLUDE_SITE)/external/com_google_absl \
             -I$(TF_INCLUDE_SITE)/external/nsync/public

# Linker flags: point to site-packages/tensorflow directory to link shared libs there.
ifeq ($(wildcard $(SITE_PACKAGES)/tensorflow),)
  TF_LDFLAGS :=
else
  TF_LDFLAGS := -L$(SITE_PACKAGES)/tensorflow -Wl,-rpath=$(SITE_PACKAGES)/tensorflow
endif



ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

KERNELS_DIR = tensorflow_nufft/cc/kernels
OPS_DIR = tensorflow_nufft/cc/ops
PROTO_DIR = tensorflow_nufft/proto

PROTO_SOURCES = $(wildcard $(PROTO_DIR)/*.proto)
PROTO_OBJECTS = $(patsubst $(PROTO_DIR)/%.proto, $(PROTO_DIR)/%.pb.cc, $(PROTO_SOURCES))
PROTO_HEADERS = $(patsubst $(PROTO_DIR)/%.proto, $(PROTO_DIR)/%.pb.h, $(PROTO_SOURCES))
PROTO_MODULES = $(patsubst $(PROTO_DIR)/%.proto, $(PROTO_DIR)/%_pb2.py, $(PROTO_SOURCES))

CUSOURCES = $(wildcard $(KERNELS_DIR)/*.cu.cc)
CUOBJECTS = $(patsubst %.cu.cc, %.cu.o, $(CUSOURCES))
CXXSOURCES = $(filter-out $(CUSOURCES), $(wildcard $(KERNELS_DIR)/*.cc) $(wildcard $(OPS_DIR)/*.cc))
CXXHEADERS = $(wildcard $(KERNELS_DIR)/*.h) $(wildcard $(OPS_DIR)/*.h)

TARGET_LIB = tensorflow_nufft/python/ops/_nufft_ops.so
TARGET_DLINK = tensorflow_nufft/cc/kernels/nufft_kernels.dlink.o



# TF_INCLUDE := $(shell $(PYTHON) -c 'import tensorflow as tf; print(tf.sysconfig.get_include())' 2>/dev/null || echo "")
# TF_INCLUDE_SITE := $(shell $(PYTHON) -c 'import site, os; import tensorflow as tf; print(os.path.join(site.getsitepackages()[0], "tensorflow/include"))' 2>/dev/null || echo "")

# # construct include flags safely
# INCLUDES := -I$(ROOT_DIR)
# ifeq ($(strip $(TF_INCLUDE_SITE)),)
#   ifneq ($(strip $(TF_INCLUDE)),)
#     INCLUDES += -I$(TF_INCLUDE)
#   endif
# else
#   INCLUDES += -I$(TF_INCLUDE_SITE)
# endif


# TF include path — computed lazily and WITHOUT importing tensorflow
# This finds the site-packages directory for the interpreter and points to the
# tensorflow/include subdirectory if present.

TF_INCLUDE := $(shell /opt/python/cp39-cp39/bin/python -c "import tensorflow as tf, os; print(os.path.join(os.path.dirname(tf.__file__), 'include'))")

# TF_INCLUDE = $(shell \
#   $(PYTHON) - <<'PY' 2>/dev/null || python - <<'PY' 2>/dev/null; \
# import site, os, sys; \
# site_pkgs = [] ; \
# try: \
#     site_pkgs = site.getsitepackages() \
# except Exception: \
#     site_pkgs = [p for p in sys.path if 'site-packages' in p]; \
# found = '' ; \
# for p in site_pkgs: \
#     tp = os.path.join(p, 'tensorflow', 'include'); \
#     if os.path.isdir(tp): \
#         found = tp; break; \
# sys.stdout.write(found) ; \
# PY \
# )


# add CUDA include only when present
ifeq ($(wildcard /usr/local/cuda/include),/usr/local/cuda/include)
  CUDA_INCLUDE := /usr/local/cuda/include
else
  CUDA_INCLUDE :=
endif
ifeq ($(strip $(CUDA_INCLUDE)),)
  # nothing
else
  INCLUDES += -I$(CUDA_INCLUDE)
endif

CUDA_LIBDIR  = /usr/local/cuda/lib64

CUDA ?= 1
OMP ?= 1
CFLAGS = -O3 -march=x86-64 -mtune=generic -funroll-loops -fcx-limited-range  -fPIC

ifeq ($(OMP), 1)
CFLAGS += -fopenmp
endif

CXXFLAGS = -std=c++17 $(CFLAGS) $(TF_CFLAGS) $(INCLUDES)\
           -I$(ROOT_DIR) \
           -I$(TF_INCLUDE) \
           -I$(TF_INCLUDE_SITE) \
           -I$(TF_INCLUDE_SITE)/external/eigen_archive \
           -I$(TF_INCLUDE_SITE)/external/com_google_absl \
           -I$(TF_INCLUDE_SITE)/external/nsync/public \
		   
ifeq ($(CUDA), 1)
CXXFLAGS += -I$(CUDA_INCLUDE)
endif


CXXFLAGS += -I/usr/local/include  
CXXFLAGS += -Wno-deprecated-declarations
CXXFLAGS += -Wno-macro-redefined

# ==============================================================================
# NVCC options
# ==============================================================================
NVARCH_FLAGS ?= -arch=sm_86

NVCC := nvcc --compiler-bindir=/usr/bin/g++-13
# pass diagnostics to the CUDA front end via -Xcudafe=...
CUDAFE_LIST := --diag_suppress=174 \
               --diag_suppress=177 \
               --diag_suppress=611 \
               --diag_suppress=20012 \
               --diag_suppress=1886 \
               --display_error_number

# Expand each item into its own -Xcudafe=<flag> token. This avoids quoting problems.
CUDAFE_XFLAGS := $(foreach f,$(CUDAFE_LIST),-Xcudafe=$(f))

CUFLAGS = $(NVARCH_FLAGS) \
          -Xcompiler "$(CFLAGS)" \
          $(TF_CFLAGS) \
          -DNDEBUG --expt-relaxed-constexpr \
          -I$(ROOT_DIR) \
          -I$(TF_INCLUDE) \
          -I$(TF_INCLUDE_SITE) \
          -I$(TF_INCLUDE_SITE)/external/eigen_archive \
          -I$(TF_INCLUDE_SITE)/external/com_google_absl \
          -I$(TF_INCLUDE_SITE)/external/nsync/public \
          $(CUDAFE_XFLAGS) \
          -Wno-deprecated-gpu-targets \
          --std=c++17 -allow-unsupported-compiler
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

LDFLAGS  += -L/usr/local/lib -lprotobuf

# ==============================================================================
# TensorFlow NUFFT
# ==============================================================================

all: lib 
# wheel

lib: proto $(TARGET_LIB)

%.cu.o: %.cu.cc
	$(NVCC) --device-c -x cu $(CUFLAGS) --threads 0 --output-file $@ --compile $<
# --compiler-bindir $(CXX) 

$(TARGET_DLINK): $(CUOBJECTS)
	$(NVCC)  --device-link $(CUFLAGS) --threads 0 --output-file $@ $^
# --compiler-bindir $(CXX)

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
