## Install spack

```bash
git clone https://github.com/spack/spack.git
export SPACK_ROOT=/path/to/spack
export PATH=${SPACK_ROOT}/bin:${PATH}
source ${SPACK_ROOT}/share/spack/setup-env.sh
```

## Clone hpctoolkit-gpu-sanitizer

```bash
git clone --recursive git@github.com:Jokeren/hpctoolkit-gpu-patch.git
cd hpctoolkit-gpu-patch
```

## Compile dyninst

```bash
cd dyninst/
cmake /path/to/dyninst/source -DCMAKE_INSTALL_PREFIX=/path/to/installation
make install -j4
```

If this does not work for you, please refer to the [Wiki](https://github.com/dyninst/dyninst/wiki) for detailed instructions. If you encounter any errors, see the [Building Dyninst](https://github.com/dyninst/dyninst/wiki/Building-Dyninst) or leave a [GitHub issue](https://github.com/dyninst/dyninst/issues).

## Compile GPU patch

```bash
# back to hpctoolkit-gpu-patch
cd ../
make PREFIX=/path/to/install/gpu/patch/lib install
```

## Configure spack

- config.yaml

```bash
cd /path/to/spack
cd spack/etc/spack
cp defaults/config.yaml .
# change build_job to number of cores on your machine
vim config.yaml 
```

- packages.yaml

```bash
cd spack/etc/spack 
cp /path/to/hpctoolkit/spack/packages.yaml ./
# change packages settings, including cuda@10.1, dyninst@10.1, cmake, perl, gcc@7.3.0
# you can set the version to a system install according to commonts in packages.yaml
vim packages.yaml
```

- package.py

```bash
cd spack/var/spack/repos/builtin/packages/hpctoolkit 
cp /path/to/hpctoolkit/spack/package.py ./
```

- check

```
spack spec hpctoolkit
```

## Install hpctoolkit

```bash
spack install --only dependencies hpctoolkit 
cd /path/to/hpctoolkit
mkdir build && cd build
# Tip: check spack libraries' root->spack find --path.  
# For example: --with-spack=/home/username/spack/opt/spack/linux-ubuntu18.04-ivybridge/gcc-7.4.0/
../configure --prefix=/path/to/install/hpctoolkit --with-dyninst=/path/to/dyninst/installation --with-cuda=/usr/local/cuda-10.1 --with-sanitizer=/path/to/sanitizer/lib --with-cupti=/usr/local/cuda-10.1/extras/CUPTI --with-gpu-patch=/path/to/install/gpu/patch/lib --with-spack=/path/to/spack/libraries/root --enable-develop

make install -j8
```

### Test sanitizer

```bash
git clone git@github.com:Jokeren/hpctoolkit-gpu-samples.git
cd hpctoolkit-gpu-samples/cuda_vec_add
export OMP_NUM_THREADS=1 [you can set any number of threads as you want]
hpcrun -e gpu=nvidia,sanitizer ./main &> log [you can enable block sampling by nvidia-cuda-memory@sampling frequency]
more log
```