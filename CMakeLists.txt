cmake_minimum_required(VERSION 3.11)

add_compile_options(-ggdb3)

project(
  Numba
  VERSION 1.0
  LANGUAGES CXX CUDA
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
set(RAYLIB_VERSION 4.5.0)
find_package(raylib ${RAYLIB_VERSION} REQUIRED)

file(GLOB_RECURSE NumbaSources src/*.cu src/*.cc)
add_executable(
  Numba
  ${NumbaSources}
  )
target_include_directories(
  Numba
  PRIVATE
  inc
  )
target_link_libraries(Numba raylib)
set_property(TARGET Numba PROPERTY CUDA_ARCHITECTURES 80)