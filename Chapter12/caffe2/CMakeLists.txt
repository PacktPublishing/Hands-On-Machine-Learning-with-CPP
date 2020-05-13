cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(caffe2-load)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 17)

find_package(Protobuf REQUIRED)
find_package(Torch REQUIRED)
find_package(ONNX REQUIRED)
find_package(OpenCV REQUIRED)


include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${ONNX_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "-msse3 -fopenmp -Wall -Wextra -Wno-unused-parameter -pthread -fopenmp -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb -O0")

add_definitions(-DONNX_NAMESPACE=onnx_torch)
add_definitions(-DONNX_ML=1)

set(SOURCE_FILES main.cc)

set(REQUIRED_LIBS "")
list(APPEND REQUIRED_LIBS ${TORCH_LIBRARIES})
list(APPEND REQUIRED_LIBS ${PROTOBUF_LIBRARY})
list(APPEND REQUIRED_LIBS ${OpenCV_LIBRARIES})

add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${CMAKE_PROJECT_NAME} ${REQUIRED_LIBS})

