#!/bin/bash

# ./nekbone.sh --enable-backend nomp --cc ~/.nomp/clang/bin/clang
# ./scripts/frontier.sh nomp 7 100 1:00:00
#
# ./nekbone.sh --enable-backend hip --cc hipcc
# ./scripts/frontier.sh hip 7 100 1:00:00

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

: ${NEKBONE_CC:=""}
: ${NEKBONE_CFLAGS:="-O3"}
: ${NEKBONE_BUILD_TYPE:=RelWithDebInfo}
: ${NEKBONE_BUILD_DIR:=`pwd`/build}
: ${NEKBONE_INSTALL_PREFIX:=`pwd`/install}
: ${NEKBONE_LIB_SUFFIX:=.so}
: ${NEKBONE_OPENCL:=OFF}
: ${NEKBONE_OPENCL_INC_DIR:=${CONDA_PREFIX}/include}
: ${NEKBONE_OPENCL_LIB_DIR:=${CONDA_PREFIX}/lib/libOpenCL${NEKBONE_LIB_SUFFIX}}
: ${NEKBONE_CUDA:=OFF}
: ${NEKBONE_HIP:=OFF}
: ${NEKBONE_NOMP:=OFF}

backend_set=0

function set_backend() {
  if [[ ${backend_set} -eq 1 ]]; then
    return
  fi

  case $1 in
    "sycl")
      NEKBONE_SYCL=ON
      backend_set=1
    "opencl")
      NEKBONE_OPENCL=ON
      backend_set=1
      ;;
    "cuda")
      NEKBONE_CUDA=ON
      backend_set=1
      ;;
    "hip")
      NEKBONE_HIP=ON
      backend_set=1
      ;;
    "nomp")
      NEKBONE_NOMP=ON
      backend_set=1
      ;;
    *)
      echo "Error: Invalid backend: $1"
      exit 1
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
      NEKBONE_CC="$2"
      shift
      shift
      ;;
    --cflags)
      NEKBONE_CFLAGS="$2"
      shift
      shift
      ;;
    --build-type)
      NEKBONE_BUILD_TYPE="$2"
      shift
      shift
      ;;
    --build-dir)
      NEKBONE_BUILD_DIR="$2"
      shift
      shift
      ;;
    --install-prefix)
      NEKBONE_INSTALL_PREFIX="$2"
      shift
      shift
      ;;
    --enable-backend)
      set_backend "$2"
      shift
      shift
      ;;
    *)
      echo "Error: Unknown option: $1"
      print_help
      exit 1
      ;;
  esac
done

### Don't touch anything that follows this line. ###
cmake -S ${PWD} -B ${NEKBONE_BUILD_DIR} \
  -DCMAKE_C_COMPILER=${NEKBONE_CC} \
  -DCMAKE_C_FLAGS=${NEKBONE_CFLAGS} \
  -DCMAKE_BUILD_TYPE=${NEKBONE_BUILD_TYPE} \
  -DCMAKE_INSTALL_PREFIX=${NEKBONE_INSTALL_PREFIX} \
  -DENABLE_SYCL=${NEKBONE_SYCL} \
  -DENABLE_OPENCL=${NEKBONE_OPENCL} \
  -DOpenCL_INCLUDE_DIR=${NEKBONE_OPENCL_INC_DIR} \
  -DOpenCL_LIBRARY=${NEKBONE_OPENCL_LIB_DIR} \
  -DENABLE_CUDA=${NEKBONE_CUDA} \
  -DENABLE_HIP=${NEKBONE_HIP} \
  -DENABLE_NOMP=${NEKBONE_NOMP}

cmake --build ${NEKBONE_BUILD_DIR} --target install -j4
