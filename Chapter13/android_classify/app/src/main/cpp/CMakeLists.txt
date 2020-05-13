# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(PATH_TO_PYTORCH CACHE STRING "")

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include_directories(${PATH_TO_PYTORCH}/include)

add_definitions(-DONNX_NAMESPACE=onnx_torch)

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
        native-lib

        # Sets the library as a shared library.
        SHARED

        # Provides a relative path to your source file(s).
        native-lib.cpp)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
        log-lib

        # Specifies the name of the NDK library that
        # you want CMake to locate.
        log)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

set(pytorch-lib ${PATH_TO_PYTORCH}/lib/libtorch.so)
set(c10-lib ${PATH_TO_PYTORCH}/lib/libc10.so)

#set(pytorch-lib ${PATH_TO_PYTORCH}/lib/libtorch.a)
#set(c10-lib ${PATH_TO_PYTORCH}/lib/libc10.a)
#set(cpuinfo-lib ${PATH_TO_PYTORCH}/lib/libcpuinfo.a)
#set(clog-lib ${PATH_TO_PYTORCH}/lib/libclog.a)
#set(caffe2-protos-lib ${PATH_TO_PYTORCH}/lib/libcaffe2_protos.a)
#set(protobuf-lib ${PATH_TO_PYTORCH}/lib/libprotobuf.a)
#set(nnpack-lib ${PATH_TO_PYTORCH}/lib/libnnpack.a)
#set(qnnpack-lib ${PATH_TO_PYTORCH}/lib/libqnnpack.a)
#set(thread-lib ${PATH_TO_PYTORCH}/lib/libpthreadpool.a)

target_link_libraries( # Specifies the target library.
        native-lib

#        -Wl,--whole-archive -Wl,--no-as-needed

        ${log-lib}
        ${pytorch-lib}
        ${c10-lib}
#        ${cpuinfo-lib}
#        ${clog-lib}
#        ${caffe2-protos-lib}
#        ${protobuf-lib}
#        ${nnpack-lib}
#        ${qnnpack-lib}
#        ${thread-lib}
        android
        )