#!/bin/bash

function print_help() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --help Print this help message and exit."
  echo "  --cc <compiler> Set compiler to be used for the build."
  echo "  --cflags <cflags> Set compiler flags for the build."
  echo "  --build-type <Release|Debug> Set build type."
  echo "  --build-dir <build directory> Set build directory."
  echo "  --install-prefix <install prefix> Set install prefix."
  echo "  --enable-backend <backend> Set backend to be used for the build."
}

: ${BP5_CC:=}
: ${BP5_CFLAGS:=-O3}
: ${BP5_BUILD_TYPE:=RelWithDebInfo}
: ${BP5_BUILD_DIR:=`pwd`/build}
: ${BP5_INSTALL_PREFIX:=`pwd`/install}
: ${BP5_OPENCL:=OFF}
: ${BP5_CUDA:=OFF}
: ${BP5_HIP:=OFF}
: ${BP5_NOMP:=OFF}

backend_set=0

function set_backend() {
  if [[ ${backend_set} -eq 1 ]]; then
    return
  fi

  echo "Backend: $1"
  case $1 in
    "opencl")
      BP5_OPENCL=ON
      backend_set=1
      ;;
    "cuda")
      BP5_CUDA=ON
      backend_set=1
      ;;
    "hip")
      BP5_HIP=ON
      backend_set=1
      ;;
    "nomp")
      BP5_NOMP=ON
      backend_set=1
      ;;
    *)
      echo "Invalid backend: $1"
      exit 0
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --help)
      print_help
      exit 0
      ;;
    --cc)
      BP5_CC="$2"
      shift
      shift
      ;;
    --cflags)
      BP5_CFLAGS="$2"
      shift
      shift
      ;;
    --build-type)
      BP5_BUILD_TYPE="$2"
      shift
      shift
      ;;
    --build-dir)
      BP5_BUILD_DIR="$2"
      shift
      shift
      ;;
    --install-prefix)
      BP5_INSTALL_PREFIX="$2"
      shift
      shift
      ;;
    --enable-backend)
      set_backend "$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      print_help
      exit 1
      ;;
  esac
done

if [[ "${BP5_NOMP}" ==  "ON" ]]; then
  if [[ -z "${NOMP_INSTALL_DIR}"  ||  -z "${NOMP_CLANG_DIR}" ]]; then
    echo "Error: One of NOMP_INSTALL_DIR or NOMP_CLANG_DIR is not defined."
    exit 1
  else
    export NOMP_LIB_DIR=${NOMP_INSTALL_DIR}/lib
    export NOMP_INC_DIR=${NOMP_INSTALL_DIR}/include
  
    BP5_CC=${NOMP_CLANG_DIR}/bin/clang
    BP5_CFLAGS="-O3 -fnomp -I${NOMP_INC_DIR} -include nomp.h"
    export LDFLAGS="-Wl,-rpath,${NOMP_LIB_DIR} -L${NOMP_LIB_DIR} -lnomp"
  fi
fi

### Don't touch anything that follows this line. ###
BP5_CMAKE_CMD="-DENABLE_OPENCL=${BP5_OPENCL} -DENABLE_CUDA=${BP5_CUDA}"
BP5_CMAKE_CMD="${BP5_CMAKE_CMD} -DENABLE_HIP=${BP5_HIP}"
BP5_CMAKE_CMD="${BP5_CMAKE_CMD} -DENABLE_NOMP=${BP5_NOMP}"

if [[ ! -z "${BP5_CC}" ]]; then
  export CC=${BP5_CC}
fi
if [[ ! -z "${BP5_CFLAGS}" ]]; then
  export CFLAGS=${BP5_CFLAGS}
fi
if [[ ! -z "${BP5_BUILD_TYPE}" ]]; then
  BP5_CMAKE_CMD="${BP5_CMAKE_CMD} -DCMAKE_BUILD_TYPE=${BP5_BUILD_TYPE}"
fi
if [[ ! -z "${BP5_BUILD_DIR}" ]]; then
  BP5_CMAKE_CMD="${BP5_CMAKE_CMD} -B ${BP5_BUILD_DIR}"
fi
if [[ ! -z "${BP5_INSTALL_PREFIX}" ]]; then
  BP5_CMAKE_CMD="${BP5_CMAKE_CMD} -DCMAKE_INSTALL_PREFIX=${BP5_INSTALL_PREFIX}"
fi

BP5_CURRENT_DIR=`pwd`
mkdir -p ${BP5_BUILD_DIR} 2> /dev/null

echo "cmake -S ${BP5_CURRENT_DIR} ${BP5_CMAKE_CMD}"
cmake -S ${BP5_CURRENT_DIR} ${BP5_CMAKE_CMD}
#  -DOpenCL_LIBRARY=${CONDA_PREFIX}/lib/libOpenCL.dylib \
#  -DOpenCL_INCLUDE_DIR=${CONDA_PREFIX}/include
  
cmake --build ${BP5_BUILD_DIR} --target install -j10
