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

: ${NEKBONE_CC:="icx"}
: ${NEKBONE_CFLAGS:="-O3 -g"}
: ${NEKBONE_CXX:="icpx"}
: ${NEKBONE_CXXFLAGS:="-O3 -g"}
: ${NEKBONE_BUILD_TYPE:=RelWithDebInfo}
: ${NEKBONE_BUILD_DIR:=`pwd`/build}
: ${NEKBONE_INSTALL_PREFIX:=`pwd`/install}
: ${NEKBONE_LIB_SUFFIX:=.so}

: ${NEKBONE_OCCA_DIR:=${HOME}/libocca/occa/install}
: ${NEKBONE_OPENCL_INC_DIR:=}
: ${NEKBONE_OPENCL_LIB_DIR:=}

: ${NEKBONE_SYCL_FLAGS:="-fsycl -fsycl-targets=intel_gpu_pvc -ftarget-register-alloc-mode=pvc:auto -fma"}
: ${NEKBONE_OMP_FLAGS:="-fiopenmp -fopenmp-targets=spir64=\"-O3\""}

backend_set=0
NEKBONE_OCCA="OFF"
NEKBONE_SYCL="OFF"
NEKBONE_CUDA="OFF"
NEKBONE_HIP="OFF"
NEKBONE_NOMP="OFF"
NEKBONE_OPENCL="OFF"
NEKBONE_OMP="OFF"

function check_backend() {
  if [[ ${backend_set} -eq 1 ]]; then
    return
  fi

  backend=$1
  backend=$( echo ${backend} | awk '{ print tolower($0) }' )
  case ${backend} in
    occa)
      NEKBONE_OCCA="ON"
      backend_set=1
      ;;
    sycl)
      NEKBONE_SYCL="ON"
      backend_set=1
      ;;
    cuda)
      NEKBONE_CUDA="ON"
      backend_set=1
      ;;
    hip)
      NEKBONE_HIP="ON"
      backend_set=1
      ;;
    nomp)
      NEKBONE_NOMP="ON"
      backend_set=1
      ;;
    opencl)
      NEKBONE_OPENCL="ON"
      backend_set=1
      ;;
    omp)
      NEKBONE_OMP="ON"
      backend_set=1
      ;;
    *)
      echo "Error: invalid backend = $1"
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
    --cxx)
      NEKBONE_CXX="$2"
      shift
      shift
      ;;
    --cxxflags)
      NEKBONE_CXXFLAGS="$2"
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
      check_backend "$2"
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

if [[ -z "${NEKBONE_CC}" ]]; then
  echo "Error: Compiler is not set!"
  exit 1
fi

### Don't touch anything that follows this line. ###
NEKBONE_CMAKE_CMD="cmake -S ${PWD} -B ${NEKBONE_BUILD_DIR} "\
"-DCMAKE_BUILD_TYPE=${NEKBONE_BUILD_TYPE} "\
"-DCMAKE_INSTALL_PREFIX=${NEKBONE_INSTALL_PREFIX} "\
"-DENABLE_OCCA=${NEKBONE_OCCA} "\
"-DENABLE_SYCL=${NEKBONE_SYCL} "\
"-DENABLE_CUDA=${NEKBONE_CUDA} "\
"-DENABLE_HIP=${NEKBONE_HIP} "\
"-DENABLE_NOMP=${NEKBONE_NOMP} "\
"-DENABLE_OPENCL=${NEKBONE_OPENCL} "\
"-DENABLE_OMP=${NEKBONE_OMP}"

if [[ "${NEKBONE_SYCL}" == "ON" ]]; then
  NEKBONE_CXXFLAGS="${NEKBONE_CXXFLAGS} ${NEKBONE_SYCL_FLAGS}"
fi

if [[ "${NEKBONE_OMP}" == "ON" ]]; then
  NEKBONE_CFLAGS="${NEKBONE_CFLAGS} ${NEKBONE_OMP_FLAGS}"
fi

if [[ "${NEKBONE_OCCA}" == "ON" ]]; then
  NEKBONE_CMAKE_CMD="${NEKBONE_CMAKE_CMD} -DCMAKE_PREFIX_PATH=${NEKBONE_OCCA_DIR}"
fi

NEKBONE_CMAKE_CMD="${NEKBONE_CMAKE_CMD} -DCMAKE_CXX_COMPILER=${NEKBONE_CXX}"
NEKBONE_CMAKE_CMD="${NEKBONE_CMAKE_CMD} -DCMAKE_CXX_FLAGS=\"${NEKBONE_CXXFLAGS}\""
NEKBONE_CMAKE_CMD="${NEKBONE_CMAKE_CMD} -DCMAKE_C_COMPILER=${NEKBONE_CC}"
NEKBONE_CMAKE_CMD="${NEKBONE_CMAKE_CMD} -DCMAKE_C_FLAGS=\"${NEKBONE_CFLAGS}\""

if [[ "${NEKBONE_OPENCL}" == "ON" ]]; then
  NEKBONE_CMAKE_CMD="${NEKBONE_CMAKE_CMD} -DOpenCL_INCLUDE_DIR=${NEKBONE_OPENCL_INC_DIR}"
  NEKBONE_CMAKE_CMD="${NEKBONE_CMAKE_CMD} -DOpenCL_LIBRARY=${NEKBONE_OPENCL_LIB_DIR}"
fi

echo "command: ${NEKBONE_CMAKE_CMD}"
eval ${NEKBONE_CMAKE_CMD}

cmake --build ${NEKBONE_BUILD_DIR} --target install -j4
