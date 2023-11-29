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

: ${NEKBONE_CC:=}
: ${NEKBONE_CFLAGS:=-O3}
: ${NEKBONE_BUILD_TYPE:=RelWithDebInfo}
: ${NEKBONE_BUILD_DIR:=`pwd`/build}
: ${NEKBONE_INSTALL_PREFIX:=`pwd`/install}
: ${NEKBONE_LIB_SUFFIX:=".so"}
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
      echo "Invalid backend: $1"
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
      echo "Unknown option: $1"
      print_help
      exit 1
      ;;
  esac
done

if [[ "${NEKBONE_NOMP}" ==  "ON" ]]; then
  if [[ -z "${NOMP_INSTALL_DIR}"  ||  -z "${NOMP_CLANG_DIR}" ]]; then
    echo "Error: NOMP_INSTALL_DIR or NOMP_CLANG_DIR is not defined."
    exit 1
  else
    export NOMP_LIB_DIR=${NOMP_INSTALL_DIR}/lib
    export NOMP_INC_DIR=${NOMP_INSTALL_DIR}/include
  
    NEKBONE_CC=${NOMP_CLANG_DIR}/bin/clang
    NEKBONE_CFLAGS="-O3 -fnomp -I${NOMP_INC_DIR} -include nomp.h"
    export LDFLAGS="-Wl,-rpath,${NOMP_LIB_DIR} -L${NOMP_LIB_DIR} -lnomp"
  fi
fi

### Don't touch anything that follows this line. ###
if [[ -z "${NEKBONE_CC}" ]]; then
  echo "Error: NEKBONE_CC is not set."
  exit 1
fi
export CC=${NEKBONE_CC}

NEKBONE_CFLAGS="${NEKBONE_CFLAGS} -Wno-unknown-pragmas"
export CFLAGS="${NEKBONE_CFLAGS}"

NEKBONE_CMAKE_CMD="-DENABLE_OPENCL=${NEKBONE_OPENCL} -DENABLE_CUDA=${NEKBONE_CUDA}"
NEKBONE_CMAKE_CMD="${NEKBONE_CMAKE_CMD} -DENABLE_HIP=${NEKBONE_HIP}"

if [[ ${NEKBONE_OPENCL} == "ON" ]]; then
  if [[ ! -z ${NEKBONE_OPENCL_INC_DIR} ]]; then
    NEKBONE_CMAKE_CMD="${NEKBONE_CMAKE_CMD} -DOpenCL_INCLUDE_DIR=${NEKBONE_OPENCL_INC_DIR}"
  fi
  if [[ ! -z ${NEKBONE_OPENCL_INC_DIR} ]]; then
    NEKBONE_CMAKE_CMD="${NEKBONE_CMAKE_CMD} -DOpenCL_LIBRARY=${NEKBONE_OPENCL_LIB_DIR}"
  fi
fi

if [[ -z "${NEKBONE_BUILD_DIR}" ]]; then
  echo "Error: NEKBONE_BUILD_DIR is not set."
  exit 1
fi
NEKBONE_CMAKE_CMD="${NEKBONE_CMAKE_CMD} -B ${NEKBONE_BUILD_DIR}"
mkdir -p ${NEKBONE_BUILD_DIR} 2> /dev/null

if [[ ! -z "${NEKBONE_BUILD_TYPE}" ]]; then
  NEKBONE_CMAKE_CMD="${NEKBONE_CMAKE_CMD} -DCMAKE_BUILD_TYPE=${NEKBONE_BUILD_TYPE}"
fi
if [[ ! -z "${NEKBONE_INSTALL_PREFIX}" ]]; then
  NEKBONE_CMAKE_CMD="${NEKBONE_CMAKE_CMD} -DCMAKE_INSTALL_PREFIX=${NEKBONE_INSTALL_PREFIX}"
fi

NEKBONE_CURRENT_DIR=`pwd`

echo "cmake -S ${NEKBONE_CURRENT_DIR} ${NEKBONE_CMAKE_CMD}"
cmake -S ${NEKBONE_CURRENT_DIR} ${NEKBONE_CMAKE_CMD}
cmake --build ${NEKBONE_BUILD_DIR} --target install -j10
