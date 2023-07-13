#!/bin/bash

: ${BP5_INSTALL_DIR:=`pwd`/install}
: ${BP5_OPENCL:=OFF}
: ${BP5_CUDA:=OFF}
: ${BP5_HIP:=OFF}
: ${BP5_NOMP:=ON}

### Don't touch anything that follows this line. ###
BP5_CURRENT_DIR=`pwd`
BP5_BUILD_DIR=${BP5_CURRENT_DIR}/build

mkdir -p ${BP5_BUILD_DIR} 2> /dev/null

cmake -DCMAKE_INSTALL_PREFIX=${BP5_INSTALL_DIR} \
  -B ${BP5_BUILD_DIR} \
  -S ${BP5_CURRENT_DIR} \
  -DENABLE_OPENCL=${BP5_OPENCL} \
  -DENABLE_CUDA=${BP5_CUDA} \
  -DENABLE_HIP=${BP5_HIP} \
  -DENABLE_NOMP=${BP5_NOMP}
  
cmake --build ${BP5_BUILD_DIR} --target install -j10
