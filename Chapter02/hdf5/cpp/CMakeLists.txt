cmake_minimum_required(VERSION 3.0)
project(hdf5_sample)

find_package(HDF5 REQUIRED)

set(HDF5_LIB_PATH "" CACHE PATH "Path to hdf5 library include dir")
set(JSON_LIB_PATH "" CACHE PATH "Path to json library include dir")

if (NOT HIGHFIVE_LIB_PATH)
  message(FATAL_ERROR "Missigng HIGHFIVE lib install path, please specify HIGHFIVE_LIB_PATH")
else()
  message("HIGHFIVE lib path is ${HIGHFIVE_LIB_PATH}")
endif()

if (NOT JSON_LIB_PATH)
  message(FATAL_ERROR "Missing Json lib install path, please specify JSON_LIB_PATH")
else()
  message("Json lib path is ${JSON_LIB_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++17 -msse3 -fopenmp -Wall -Wextra ")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

set(requiredlibs "stdc++fs")
list(APPEND requiredlibs "stdc++")
list(APPEND requiredlibs ${HDF5_LIBRARIES})

include_directories(${HDF5_INCLUDE_DIR})
include_directories(${HIGHFIVE_LIB_PATH})
include_directories(${JSON_LIB_PATH})

set(SOURCES hdf5.cc
            ../../json/cpp/paper.h
            ../../json/cpp/review.h
            ../../json/cpp/reviewsreader.h
            ../../json/cpp/reviewsreader.cpp)

add_executable(hdf5_sample ${SOURCES})
target_link_libraries(hdf5_sample ${requiredlibs})

