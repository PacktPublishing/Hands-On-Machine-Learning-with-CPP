cmake_minimum_required(VERSION 3.0)
project(mlp-sharkml)

set(SHARK_PATH "" CACHE PATH "path to SharkML install dir")

if(NOT SHARK_PATH)
  message(FATAL_ERROR "Missing SharkML install path, please specify SHARK_PATH")
else()
  message("SharkML path is ${SHARK_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost REQUIRED serialization)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include_directories(${SHARK_PATH}/include)
link_directories(${SHARK_PATH}/lib)
link_directories(${SHARK_PATH}/lib64)

set(SOURCES
    mlp-sharkml.cc
    ../data/data.h
    ../data/data.cc
    )

add_executable(mlp-sharkml  ${SOURCES})
target_link_libraries(mlp-sharkml shark cblas ${Boost_LIBRARIES})

