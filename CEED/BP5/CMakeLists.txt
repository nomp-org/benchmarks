cmake_minimum_required(VERSION 3.23)
project(bp5 VERSION 0.0.1 DESCRIPTION "" LANGUAGES C)

set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${CMAKE_INSTALL_PREFIX}/lib64")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(ENABLE_OPENCL "Build OPENCL backend" OFF)
option(ENABLE_CUDA "Build CUDA backend" OFF)
option(ENABLE_HIP "Build HIP backend" OFF)
option(ENABLE_NOMP "Build NOMP backend" OFF)

set(BP5_SCRIPTS_DIR ${CMAKE_SOURCE_DIR}/backends/)

configure_file(${CMAKE_SOURCE_DIR}/include/bp5-defs.h.in
  ${CMAKE_SOURCE_DIR}/include/bp5-defs.h @ONLY)

set(SOURCES src/bp5-impl.c src/bp5-setup.c backends/bp5-backend.c
  backends/bp5-backend-dummy.c)

if (ENABLE_OPENCL)
  find_package(OpenCL REQUIRED)
  if (OpenCL_FOUND)
    list(APPEND SOURCES backends/bp5-backend-opencl.c)
  else()
    message(FATAL_ERROR "ENABLE_OPENCL is ON but unable to find OPENCL.")
  endif()
endif()

if (ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  if (CUDAToolkit_FOUND)
    enable_language(CUDA)
    list(APPEND SOURCES backends/bp5-backend-cuda.cu)
  else()
    message(FATAL_ERROR "ENABLE_CUDA is ON but unable to find CUDA.")
  endif()
endif()

if (ENABLE_HIP)
  find_package(HIP)
  if(HIP_FOUND)
    list(APPEND SOURCES backends/bp5-backend-hip.c)
  else()
    message(FATAL_ERROR "ENABLE_HIP is ON but unable to find HIP.")
  endif()
endif()

if (ENABLE_NOMP)
  list(APPEND SOURCES backends/bp5-backend-nomp.c)
endif()

add_library(bp5 SHARED ${SOURCES})
set_target_properties(bp5 PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1
  PUBLIC_HEADER include/bp5.h)
target_include_directories(bp5 PRIVATE include)
target_compile_options(bp5 PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
# Add -Wpedantic if compiler is not MSVC and language CUDA is not enabled.
target_compile_options(bp5 PRIVATE
  $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-Wpedantic>)
target_link_libraries(bp5 PRIVATE m)

if (ENABLE_OPENCL)
  target_link_libraries(bp5 PRIVATE OpenCL::OpenCL)
endif()
if (ENABLE_CUDA)
  target_link_libraries(bp5 PRIVATE CUDA::cudart CUDA::nvrtc)
endif()
if (ENABLE_HIP)
  target_link_libraries(bp5 PRIVATE bp5::HIP)
endif()

add_executable(bp5-driver bin/bp5-driver.c)
set_target_properties(bp5-driver PROPERTIES VERSION ${PROJECT_VERSION})
target_link_libraries(bp5-driver PRIVATE bp5)
target_include_directories(bp5-driver PRIVATE include)
target_compile_options(bp5-driver PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra>)
# Add -Wpedantic if compiler is not MSVC and language CUDA is not enabled.
target_compile_options(bp5-driver PRIVATE
  $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<NOT:$<COMPILE_LANGUAGE:CUDA>>>:-Wpedantic>)

find_program(CLANG_FORMAT NAMES clang-format)
if (CLANG_FORMAT)
  add_custom_target(format COMMAND ${CLANG_FORMAT} -i
    ${CMAKE_CURRENT_SOURCE_DIR}/*/*.[ch]
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running clang-format ...")
endif()

install(TARGETS bp5 bp5-driver RUNTIME DESTINATION bin LIBRARY DESTINATION lib
  PUBLIC_HEADER DESTINATION include
)
