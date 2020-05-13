cmake_minimum_required(VERSION 3.0)
project(mlpack-recommender)

set(MLPACK_PATH "" CACHE PATH "path to mlpack install dir")

if (NOT MLPACK_PATH)
  message(FATAL_ERROR "Missigng mlpack install path, please specify MLPACK_PATH")
else()
  message("mlpack path is ${MLPACK_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost REQUIRED serialization)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include_directories(${MLPACK_PATH}/include)
link_directories(${MLPACK_PATH}/lib)
link_directories(${MLPACK_PATH}/lib64)

set(SOURCES mlpack-recommender.cc)

add_executable(mlpack-recommender ${SOURCES})
target_link_libraries(mlpack-recommender mlpack armadillo cblas lapack ${Boost_LIBRARIES} stdc++fs )

