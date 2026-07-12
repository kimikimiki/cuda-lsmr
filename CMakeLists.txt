cmake_minimum_required(VERSION 3.22)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING
      "CUDA architectures; 86 is the tested RTX 3060 baseline")
endif()

project(cuda_lsmr VERSION 0.1.0 LANGUAGES CXX CUDA)

find_package(CUDAToolkit 12.9 REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS 12.9 OR
   CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.9)
  message(FATAL_ERROR "cuda-lsmr requires CUDA Toolkit 12.9 or newer")
endif()

option(CUDA_LSMR_BUILD_TESTS "Build C++ tests" ON)
option(CUDA_LSMR_BUILD_PYTHON "Build the pybind11 benchmark module" OFF)
option(CUDA_LSMR_BUILD_EXAMPLES "Build C++ examples" ON)
option(CUDA_LSMR_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)

add_library(cuda_lsmr SHARED
  src/runtime/context.cu
  src/operators/dense_matrix.cu
  src/operators/linear_operator.cpp
  src/operators/sparse_matrix.cu
  src/kernels/vector_ops.cu
  src/solver/solver.cu
  src/version.cpp
)
add_library(cuda_lsmr::cuda_lsmr ALIAS cuda_lsmr)

target_compile_features(cuda_lsmr PUBLIC cxx_std_17 PRIVATE cuda_std_17)
target_include_directories(cuda_lsmr
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(cuda_lsmr
  PUBLIC
    CUDA::cudart
  PRIVATE
    CUDA::cublas
    CUDA::cusparse
)
set_target_properties(cuda_lsmr PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  VERSION ${PROJECT_VERSION}
  SOVERSION ${PROJECT_VERSION_MAJOR}
)

set(_cuda_lsmr_warning_flags -Wall -Wextra -Wpedantic)
if(CUDA_LSMR_WARNINGS_AS_ERRORS)
  list(APPEND _cuda_lsmr_warning_flags -Werror)
endif()
target_compile_options(cuda_lsmr PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${_cuda_lsmr_warning_flags}>
)

if(CUDA_LSMR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(CUDA_LSMR_BUILD_PYTHON)
  add_subdirectory(bindings)
endif()

if(CUDA_LSMR_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
install(TARGETS cuda_lsmr EXPORT cuda_lsmrTargets)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT cuda_lsmrTargets
  FILE cuda_lsmrTargets.cmake
  NAMESPACE cuda_lsmr::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuda_lsmr
)
configure_package_config_file(
  cmake/cuda_lsmrConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cuda_lsmrConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuda_lsmr
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/cuda_lsmrConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/cuda_lsmrConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cuda_lsmrConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cuda_lsmr
)
