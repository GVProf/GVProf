## Install spack

```bash
git clone https://github.com/spack/spack.git
export SPACK_ROOT=/path/to/spack
export PATH=${SPACK_ROOT}/bin:${PATH}
source ${SPACK_ROOT}/share/spack/setup-env.sh
```

## Install gpu-patch

```bash
git clone --recursive git@github.com:Jokeren/GVProf.git
cd GVProf
make PREFIX=/path/to/gpu-patch install
```

## Install dependencies

```bash
spack spec hpctoolkit
spack install --only dependencies hpctoolkit 
```

## Install dyninst

Currently, we need a specific version of dyninst. So we have to compile it by ourself.

### Get required packages' paths

Since we have build the required packages by spack, now we have to get the paths dyninst needed.

Run`spack install --only dependencies hpctoolkit` again, and the console will output every libraries installed by spack. And we need the following libaraies' paths:

```
elfutils
boost
intel-tbb
```

### Compile

```bash
cd dyninst/
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/installation -DBoost_ROOT_DIR=/boost/path -DLibElf_ROOT_DIR=/path/to/elfutils -DTBB_ROOT_DIR=/path/to/intel-tbb
# Before make, make sure there is no error in the last step
make install -j8
```

## Install hpctoolkit

### hpctoolkit

```bash
cd /path/to/hpctoolkit
mkdir build && cd build
# Tip: check spack libraries' root->spack find --path.  
# For example: --with-spack=/home/username/spack/opt/spack/linux-ubuntu18.04-zen/gcc-7.4.0/
../configure --prefix=/path/to/hpctoolkit --with-dyninst=/path/to/dyninst --with-cuda=/usr/local/cuda-11.0 --with-sanitizer=/path/to/sanitizer --with-gpu-patch=/path/to/gpu-patch --with-redshow=/path/to/redshow  --with-spack=/path/to/spack/libraries/root
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

## Test

```bash
cd GVProf/samples/vectorAdd.f32
make
hpcrun -e gpu=nvidia ./vectorAdd [dump cubins]
hpcstruct --gpucfg yes hpctoolkit-vectorAdd-measurements [analyze cubins]
hpcrun -e gpu=nvidia,sanitizer ./vectorAdd [profile execution]
hpcprof hpctoolkit-vectorAdd-measurements
hpcviewer hpctoolkit-vectorAdd-database
```
