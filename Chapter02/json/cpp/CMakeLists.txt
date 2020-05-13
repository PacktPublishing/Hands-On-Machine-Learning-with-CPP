cmake_minimum_required(VERSION 3.0)
project(json_sample)

set(JSON_LIB_PATH "" CACHE PATH "Path to json library include dir")
set(EIGEN_LIB_PATH "" CACHE PATH "Path to Eigen library include dir")

if (NOT JSON_LIB_PATH)
  message(FATAL_ERROR "Missigng Json lib install path, please specify JSON_LIB_PATH")
else()
  message("Json lib path is ${JSON_LIB_PATH}")
endif()

if (NOT EIGEN_LIB_PATH)
  message(FATAL_ERROR "Missing Eigen install path, please specify EIGEN_LIB_PATH")
else()
  message("Eigen path is ${EIGEN_LIB_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra ")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

set(requiredlibs "stdc++fs")
list(APPEND requiredlibs "stdc++")

include_directories(${JSON_LIB_PATH})
include_directories(${EIGEN_LIB_PATH})

set(SOURCES json.cc
            review.h
            paper.h
            reviewsreader.h
            reviewsreader.cpp)

add_executable(json_sample ${SOURCES})
target_link_libraries(json_sample ${requiredlibs})

