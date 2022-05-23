make lib PY_VERSION=3.8  CXX=/dt7/usr/bin/g++ -j
make wheel PY_VERSION=3.8
cp artifacts/tensorflow_nufft-0.7.3* /volatile/.
