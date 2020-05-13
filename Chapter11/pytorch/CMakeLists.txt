cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(rnn-pytorch)

find_package(Torch REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter -pthread -fopenmp")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -O0")

set(SOURCE_FILES main.cc
                 imdbdataset.h
                 imdbdataset.cc
                 rnn.h
                 rnn.cc
                 glovedict.h
                 glovedict.cc
                 imdbreader.cc
                 imdbreader.h
                 vocabulary.h
                 vocabulary.cc
                 )

set(REQUIRED_LIBS "stdc++fs")
list(APPEND REQUIRED_LIBS ${TORCH_LIBRARIES})


add_executable("${CMAKE_PROJECT_NAME}" ${SOURCE_FILES})
target_link_libraries("${CMAKE_PROJECT_NAME}" ${REQUIRED_LIBS})

