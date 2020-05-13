cmake_minimum_required(VERSION 3.0)
project(sharkml-classify)

set(SHARK_PATH "" CACHE PATH "path to SharkML install dir")

if(NOT SHARK_PATH)
  message(FATAL_ERROR "Missing SharkML install path, please specify SHARK_PATH")
else()
  message("SharkML path is ${SHARK_PATH}")
endif()

set(PLOTCPP_PATH "" CACHE PATH "path to poltcpp install dir")

if (NOT PLOTCPP_PATH)
  message(FATAL_ERROR "Missigng plotcpp include path, please specify PLOTCPP_PATH")
else()
  message("plotcpp path is ${PLOTCPP_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(Boost_USE_STATIC_LIBS ON)
set(Boost_USE_STATIC_RUNTIME ON)
find_package(Boost REQUIRED serialization)

set(CMAKE_CXX_FLAGS "-msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include_directories(${SHARK_PATH}/include)
include_directories(${PLOTCPP_PATH})
link_directories(${SHARK_PATH}/lib)
link_directories(${SHARK_PATH}/lib64)

set(SOURCES sharkml-classify.cc)

add_executable(sharkml-classify ${SOURCES})
target_link_libraries(sharkml-classify shark cblas ${Boost_LIBRARIES} stdc++fs )

