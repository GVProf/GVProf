# Install

## Install spack

```bash
git clone https://github.com/spack/spack.git
export SPACK_ROOT=/path/to/spack
source ${SPACK_ROOT}/share/spack/setup-env.sh
```

## Install gpu-patch

If you install cuda toolkit in somewhere else, you need to change the value of `SANITIZER_PATH`.

```bash
git clone --recursive git@github.com:Jokeren/GVProf.git
cd GVProf
make PREFIX=/path/to/gpu-patch/installation SANITIZER_PATH=/usr/local/cuda/compute-sanitizer/ install
```

## Install dependencies

```bash
spack spec hpctoolkit
spack install --only dependencies hpctoolkit ^dyninst@master
```

## Install redshow

```bash
cd redshow
# Tip: get boost libarary path 'spack find --path' and append include to that path
make install -j8 PREFIX=/path/to/redshow/installation BOOST_DIR=/path/to/boost/installation GPU_PATH_DIR=/path/to/gpu-patch/installation
```

## Install hpctoolkit

### hpctoolkit

```bash
cd /path/to/hpctoolkit
mkdir build && cd build
# Tip: check spack libraries' root->spack find --path.  
# For example: --with-spack=/home/username/spack/opt/spack/linux-ubuntu18.04-zen/gcc-7.4.0/
../configure --prefix=/path/to/hpctoolkit/installation --with-cuda=/usr/local/cuda-11.0 --with-sanitizer=/path/to/sanitizer --with-gpu-patch=/path/to/gpu-patch/installation --with-redshow=/path/to/redshow/installation  --with-spack=/path/to/spack/libraries/root
make install -j8
```

### hpcviewer

http://hpctoolkit.org/download/hpcviewer/

### Add to environment

Add following lines into your `.bashrc` file and source it.

```bash
export HPCTOOLKIT=/path/to/hpctoolkit
export PATH=$HPCTOOLKIT/bin/:$PATH
```

## Use gvprof script

```bash
./bin/gvprof
export PATH=`pwd`/bin/gvprof
```

## Test

```bash
cd ./samples/vectorAdd.f32
make
gvprof -e redundancy ./vectorAdd
hpcviewer gvprof-database
```
