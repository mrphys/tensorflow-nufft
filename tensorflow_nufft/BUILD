licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
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
        ":nufft_options_py_proto",
    ],
)

custom_op_library(
    name = "python/ops/_nufft_ops.so",
    srcs = [
        "cc/kernels/fftw_api.h",
        "cc/kernels/omp_api.h",
        "cc/kernels/kernel_horner_sigma2.inc",
        "cc/kernels/kernel_horner_sigma125.inc",
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
        "@//build_deps/fftw3",
        ":nufft_options_cc_proto",
    ],
)

py_proto_library(
    name = "nufft_options_py_proto",
    srcs = ["proto/nufft_options.proto"],
)

cc_proto_library(
    name = "nufft_options_cc_proto",
    deps = [":nufft_options_proto"],
)

proto_library(
    name = "nufft_options_proto",
    srcs = ["proto/nufft_options.proto"],
)
