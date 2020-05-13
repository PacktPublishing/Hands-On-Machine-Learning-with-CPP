cmake_minimum_required(VERSION 3.10)
project(shogun-anomaly)

set(SHOGUN_PATH "" CACHE PATH "path to shogun install dir")

if(NOT DEFINED SHOGUN_PATH)
  message(FATAL_ERROR "Missing Shogun install path, please specify SHOGUN_PATH")
else()
  message("Shogun path is ${SHOGUN_PATH}")
endif()

set(PLOTCPP_PATH "" CACHE PATH "path to poltcpp install dir")

if (NOT PLOTCPP_PATH)
  message(FATAL_ERROR "Missigng plotcpp include path, please specify PLOTCPP_PATH")
else()
  message("plotcpp path is ${PLOTCPP_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++14 -msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")


set(requiredlibs "stdc++fs")
list(APPEND requiredlibs "stdc++")
list(APPEND requiredlibs json-c ${PROTOBUF_LIBRARIES} cblas lapack)

include_directories(${PLOTCPP_PATH})
include_directories(${SHOGUN_PATH}/include)
link_directories(${SHOGUN_PATH}/lib)

add_executable(shogun-anomaly "shogun-anomaly.cc")
target_link_libraries(shogun-anomaly shogun ${requiredlibs})

