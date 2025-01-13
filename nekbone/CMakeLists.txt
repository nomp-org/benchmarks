cmake_minimum_required(VERSION 3.27)
project(nekbone VERSION 0.0.1 DESCRIPTION "" LANGUAGES C)

set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${CMAKE_INSTALL_PREFIX}/lib64")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_C_STANDARD_REQUIRED ON)

# Build Options:
option(ENABLE_OCCA "Build OCCA backend" OFF)
option(ENABLE_SYCL "Build SYCL backend" OFF)
option(ENABLE_OPENCL "Build OPENCL backend" OFF)
option(ENABLE_CUDA "Build CUDA backend" OFF)
option(ENABLE_HIP "Build HIP backend" OFF)
option(ENABLE_NOMP "Build NOMP backend" OFF)

# Build defaults:
set(NEKBONE_SCRIPTS_DIR ${CMAKE_SOURCE_DIR}/backends/)
set(NEKBONE_VERBOSE 1)
set(NEKBONE_DEVICE 0)
set(NEKBONE_PLATFORM 0)
set(NEKBONE_MAX_ITER 100)

configure_file(${CMAKE_SOURCE_DIR}/include/nekbone-defs.h.in
  ${CMAKE_SOURCE_DIR}/include/nekbone-defs.h @ONLY)

set(SOURCES src/nekbone-api.c src/nekbone-setup.c src/nekbone-backend.c
  backends/nekbone-backend-dummy.c backends/nekbone-backend-nomp.c)

if (ENABLE_OCCA)
  find_package(OCCA REQUIRED)
  enable_language(CXX)
  if (OCCA_FOUND)
    list(APPEND SOURCES backends/nekbone-backend-occa.cpp)
  endif()
endif()

if (ENABLE_OPENCL)
  find_package(OpenCL REQUIRED)
  if (OpenCL_FOUND)
    list(APPEND SOURCES backends/nekbone-backend-opencl.c)
  else()
    message(FATAL_ERROR "ENABLE_OPENCL is ON but unable to find OPENCL.")
  endif()
endif()

if (ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    list(APPEND SOURCES backends/nekbone-backend-cuda.cu)
  else()
    message(FATAL_ERROR "ENABLE_CUDA is ON but unable to find CUDA.")
  endif()
endif()

if (ENABLE_HIP)
  enable_language(HIP)
  list(APPEND SOURCES backends/nekbone-backend-hip.c)
  set_source_files_properties(backends/nekbone-backend-hip.c PROPERTIES LANGUAGE HIP)
endif()

if (ENABLE_SYCL)
  enable_language(CXX)
  list(APPEND SOURCES backends/nekbone-backend-sycl.cpp)
endif()

add_library(nekbone SHARED ${SOURCES})
set_target_properties(nekbone PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1
  PUBLIC_HEADER include/nekbone.h)
target_include_directories(nekbone PRIVATE include)
target_compile_options(nekbone PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
# Add -Wpedantic if compiler is not MSVC and language CUDA is not enabled.
target_compile_options(nekbone PRIVATE
  $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-Wpedantic>)
# Add flags from the user.
target_compile_options(nekbone PRIVATE "${NEKBONE_CFLAGS}")
target_link_libraries(nekbone PRIVATE m)

if (ENABLE_OCCA)
  target_link_libraries(nekbone PRIVATE OCCA::libocca)
endif()

if (ENABLE_OPENCL)
  target_link_libraries(nekbone PRIVATE OpenCL::OpenCL)
endif()

if (ENABLE_CUDA)
  target_link_libraries(nekbone PRIVATE CUDA::cudart CUDA::nvrtc)
endif()

if (ENABLE_NOMP)
  if (NOT DEFINED ENV{NOMP_INSTALL_DIR})
    message(FATAL_ERROR "ENABLE_NOMP is ON but NOMP_INSTALL_DIR is not set.")
  endif()
  target_compile_options(nekbone PRIVATE -fnomp -I$ENV{NOMP_INSTALL_DIR}/include 
    -include nomp.h)
  target_link_options(nekbone PRIVATE -Wl,-rpath,$ENV{NOMP_INSTALL_DIR}/lib
    -L$ENV{NOMP_INSTALL_DIR}/lib -lnomp)
else()
  target_compile_options(nekbone PRIVATE -Wno-unknown-pragmas
    -Wno-unused-parameter -Wno-unused-variable)
endif()

if (ENABLE_SYCL OR ENABLE_OCCA)
  set_source_files_properties(${SOURCES} PROPERTIES CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON)
else()
  set_source_files_properties(${SOURCES} PROPERTIES C_STANDARD 11
    C_STANDARD_REQUIRED ON)
endif()

add_executable(nekbone-driver bin/nekbone-driver.c)
set_target_properties(nekbone-driver PROPERTIES VERSION ${PROJECT_VERSION})
target_link_libraries(nekbone-driver PRIVATE nekbone)
target_include_directories(nekbone-driver PRIVATE include)
target_compile_options(nekbone-driver PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
# Add -Wpedantic if compiler is not MSVC and language CUDA is not enabled.
target_compile_options(nekbone-driver PRIVATE
  $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-Wpedantic>)

find_program(CLANG_FORMAT NAMES clang-format)
if (CLANG_FORMAT)
  add_custom_target(format COMMAND ${CLANG_FORMAT} -i
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.[ch]
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running clang-format ...")
endif()

install(TARGETS nekbone nekbone-driver RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)
