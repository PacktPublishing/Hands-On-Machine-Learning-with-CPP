cmake_minimum_required(VERSION 3.10)
project(mlp-dlib)

set(DLIB_PATH "" CACHE PATH "path to dlib install dir")

if (NOT DLIB_PATH)
  message(FATAL_ERROR "Missigng Dlib install path, please specify DLIB_PATH")
else()
  message("Dlib path is ${DLIB_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(requiredlibs "cblas")
list(APPEND requiredlibs "lapack")

find_package(CUDA)
if (CUDA_FOUND)
  list(APPEND requiredlibs ${CUDA_LIBRARIES})
  list(APPEND requiredlibs ${CUDA_CUBLAS_LIBRARIES})
  list(APPEND requiredlibs ${CUDA_cusolver_LIBRARY})
  list(APPEND requiredlibs ${CUDA_curand_LIBRARY})

  get_filename_component(cudnn_hint_path "${CUDA_CUBLAS_LIBRARIES}" PATH)
  find_library(cudnn cudnn
      HINTS ${cudnn_hint_path} ENV CUDNN_LIBRARY_DIR  ENV CUDNN_HOME
      PATHS /usr/local /usr/local/cuda ENV LD_LIBRARY_PATH
      PATH_SUFFIXES lib64 lib x64
      )
  message("CUDNN found ${cudnn}")
  list(APPEND requiredlibs ${cudnn})
endif()

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include_directories(${DLIB_PATH}/include)
link_directories(${DLIB_PATH}/lib)
link_directories(${DLIB_PATH}/lib64)

set(SOURCES
    mlp-dlib.cc
    ../data/data.h
    ../data/data.cc
    )

add_executable(mlp-dlib ${SOURCES})
target_link_libraries(mlp-dlib optimized dlib debug dlibd)
target_link_libraries(mlp-dlib  ${requiredlibs})


