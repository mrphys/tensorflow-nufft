load("@rules_foreign_cc//foreign_cc:configure.bzl", "configure_make")

configure_make(
    name = "fftw3",
    configure_in_place = True,
    configure_options = [
        "CFLAGS=\"-fPIC\"",
        "--enable-openmp",
    ],
    lib_source = "@fftw3_archive//:all",
    out_static_libs = [
        "libfftw3.a",
        "libfftw3_omp.a",
    ],
    visibility = ["//visibility:public"],
)

configure_make(
    name = "fftw3f",
    configure_in_place = True,
    configure_options = [
        "CFLAGS=\"-fPIC\"",
        "--enable-openmp",
        "--enable-float",
    ],
    lib_source = "@fftw3_archive//:all",
    out_static_libs = [
        "libfftw3f.a",
        "libfftw3f_omp.a",
    ],
    visibility = ["//visibility:public"],
)
