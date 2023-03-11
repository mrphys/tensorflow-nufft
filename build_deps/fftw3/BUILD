load("@rules_foreign_cc//foreign_cc:configure.bzl", "configure_make")

configure_make(
    name = "fftw3",
    configure_in_place = True,
    configure_options = ["CFLAGS=\"-fPIC\""],
    lib_source = "@fftw3_archive//:all",
    out_static_libs = ["libfftw3.a"],
    visibility = ["//visibility:public"],
)
