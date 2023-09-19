#!/bin/bash

if [ -z "${NOMP_INSTALL_DIR}" ] || [ -z "${NOMP_CLANG_DIR}" ]; then
  echo "Warning: One of NOMP_INSTALL_DIR or NOMP_CLANG_DIR is not defined."
  export BP5_NOMP=OFF
else
  echo "Info: nomp backend is enabled."
  export BP5_NOMP=ON
  export NOMP_LIB_DIR=${NOMP_INSTALL_DIR}/lib
  export NOMP_INC_DIR=${NOMP_INSTALL_DIR}/include

  export BP5_CC=${NOMP_CLANG_DIR}/clang
  export BP5_CFLAGS="-O3 -fnomp -I${NOMP_INC_DIR} -include nomp.h"
  export LDFLAGS="-Wl,-rpath,${NOMP_LIB_DIR} -L${NOMP_LIB_DIR} -lnomp"
fi

: ${BP5_CC:=clang}
: ${BP5_CFLAGS:=-O3}
: ${BP5_INSTALL_DIR:=`pwd`/install}
: ${BP5_OPENCL:=ON}
: ${BP5_CUDA:=OFF}
: ${BP5_HIP:=OFF}
: ${BP5_NOMP:=OFF}

### Don't touch anything that follows this line. ###
BP5_CURRENT_DIR=`pwd`
BP5_BUILD_DIR=${BP5_CURRENT_DIR}/build
mkdir -p ${BP5_BUILD_DIR} 2> /dev/null

cmake -DCMAKE_INSTALL_PREFIX=${BP5_INSTALL_DIR} \
  -B ${BP5_BUILD_DIR} \
  -S ${BP5_CURRENT_DIR} \
  -DCMAKE_C_COMPILER=${BP5_CC} \
  -DCMAKE_C_FLAGS="${BP5_CFLAGS}" \
  -DENABLE_OPENCL=${BP5_OPENCL} \
  -DENABLE_CUDA=${BP5_CUDA} \
  -DENABLE_HIP=${BP5_HIP} \
  -DENABLE_NOMP=${BP5_NOMP} \
  -DOpenCL_LIBRARY=${CONDA_PREFIX}/lib/libOpenCL.dylib \
  -DOpenCL_INCLUDE_DIR=${CONDA_PREFIX}/include
  
cmake --build ${BP5_BUILD_DIR} --target install -j10
