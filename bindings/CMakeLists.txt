find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_cuda_lsmr MODULE module.cpp)
target_link_libraries(_cuda_lsmr PRIVATE cuda_lsmr::cuda_lsmr)
set_target_properties(_cuda_lsmr PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)
