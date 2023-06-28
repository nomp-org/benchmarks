#!/bin/bash

: ${BP5_INSTALL_DIR:=`pwd`/install}
: ${BP5_CC:=cc}

### Don't touch anything that follows this line. ###
BP5_CURRENT_DIR=`pwd`
BP5_BUILD_DIR=${BP5_CURRENT_DIR}/build

mkdir -p ${BP5_BUILD_DIR} 2> /dev/null

cmake -DCMAKE_C_COMPILER=${BP5_CC} -DCMAKE_INSTALL_PREFIX=${BP5_INSTALL_DIR} \
  -B ${BP5_BUILD_DIR} -S ${BP5_CURRENT_DIR}
  
cmake --build ${BP5_BUILD_DIR} --target install -j10
