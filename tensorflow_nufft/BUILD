licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@rules_foreign_cc//foreign_cc:configure.bzl", "configure_make")

load("//tensorflow_nufft:tensorflow_nufft.bzl", "custom_op_library")

config_setting(
    name = "windows",
    constraint_values = ["@bazel_tools//platforms:windows"],
)

py_library(
    name = "tensorflow_nufft",
    srcs = glob(["*.py"]),
    data = [
        ":python/ops/_nufft_ops.so",
    ],
)

custom_op_library(
    name = "python/ops/_nufft_ops.so",
    srcs = [
        "cc/kernels/legendre_rule_fast.cc",
        "cc/kernels/legendre_rule_fast.h",
        "cc/kernels/nufft_kernels.cc",
        "cc/kernels/nufft_options.h",
        "cc/kernels/nufft_plan.cc",
        "cc/kernels/nufft_plan.h",
        "cc/kernels/nufft_util.cc",
        "cc/kernels/nufft_util.h",
        "cc/kernels/reverse_functor_cpu.cc",
        "cc/kernels/reverse_functor.h",
        "cc/kernels/transpose_functor_cpu.cc",
        "cc/kernels/transpose_functor.h",
        "cc/ops/nufft_ops.cc",
    ],
    cuda_srcs = [
        "cc/kernels/nufft_plan.h",
        "cc/kernels/nufft_plan.cu.cc",
        "cc/kernels/reverse_functor.h",
        "cc/kernels/reverse_functor_gpu.cu.cc",
        "cc/kernels/transpose_functor.h",
        "cc/kernels/transpose_functor_gpu.cu.cc",
    ],
    deps = [
        "fftw"
    ],
)

configure_make(
    name = "fftw",
    configure_in_place = True,
    lib_source = "@fftw//:all",
    out_include_dir = "api",
    out_lib_dir = ".libs",
    out_static_libs = ["libfftw3.a"],
)
