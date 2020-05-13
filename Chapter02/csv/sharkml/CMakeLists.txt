cmake_minimum_required(VERSION 3.0)
project(sharkml-csv-sample)

set(SHARK_PATH "" CACHE PATH "path to SharkML install dir")

if(NOT SHARK_PATH)
  message(FATAL_ERROR "Missing SharkML install path, please specify SHARK_PATH")
else()
  message("SharkML path is ${SHARK_PATH}")
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Boost REQUIRED serialization)

set(CMAKE_CXX_FLAGS "-msse3 -fopenmp -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include_directories(${SHARK_PATH}/include)
link_directories(${SHARK_PATH}/lib)
link_directories(${SHARK_PATH}/lib64)

set(requiredlibs "stdc++fs")
list(APPEND requiredlibs "stdc++")
list(APPEND requiredlibs shark cblas)

add_executable(csv_sample csv_shark.cc)
target_link_libraries(csv_sample ${Boost_LIBRARIES} ${requiredlibs})

