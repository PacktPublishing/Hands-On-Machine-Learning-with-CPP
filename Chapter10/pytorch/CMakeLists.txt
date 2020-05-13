cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(lenet-pytorch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

include_directories(${TORCH_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter -pthread -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -O0")

set(SOURCE_FILES main.cpp
                 mnistdataset.h
                 mnistdataset.cpp
                 lenet5.h
                 lenet5.cpp)

set(REQUIRED_LIBS "stdc++fs")
list(APPEND REQUIRED_LIBS ${TORCH_LIBRARIES})
list(APPEND REQUIRED_LIBS ${OpenCV_LIBS})


add_executable("${CMAKE_PROJECT_NAME}" ${SOURCE_FILES})
target_link_libraries("${CMAKE_PROJECT_NAME}" ${REQUIRED_LIBS})

