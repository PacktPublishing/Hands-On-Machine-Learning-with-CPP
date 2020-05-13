#!/usr/bin/env bash
set -x
set -e

START_DIR=$(pwd)
mkdir $START_DIR/android
cd $START_DIR/android

wget https://dl.google.com/android/repository/sdk-tools-linux-4333796.zip
unzip sdk-tools-linux-4333796.zip

yes | ./tools/bin/sdkmanager --licenses
yes | ./tools/bin/sdkmanager "platform-tools"
yes | ./tools/bin/sdkmanager "platforms;android-25"
yes | ./tools/bin/sdkmanager "build-tools;25.0.2"
yes | ./tools/bin/sdkmanager "system-images;android-25;google_apis;armeabi-v7a"
yes | ./tools/bin/sdkmanager --install "ndk;20.0.5594570"

git clone https://github.com/pytorch/pytorch.git
cd pytorch/
git checkout v1.2.0
git submodule update --init --recursive

export ANDROID_NDK=$START_DIR/android/ndk/20.0.5594570
export ANDROID_ABI='armeabi-v7a'

$START_DIR/android/pytorch/scripts/build_android.sh \
-DBUILD_CAFFE2_MOBILE=OFF \
-DBUILD_SHARED_LIBS=ON \
-DCMAKE_PREFIX_PATH=$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') \
-DPYTHON_EXECUTABLE=$(python -c 'import sys; print(sys.executable)') \

# don't forget to upadte android project gradle local.properties file with next variables:
# sdk.dir=$START_DIR/android/
# pytorch.dir=$START_DIR/android/pytorch/build_android/install
# and copy pytorch shared libraries
# cp $START_DIR/android/pytorch/build_android/install/lib/*.so app/src/main/jniLibs/armeabi-v7a/
# build app with './gradlew build' command

