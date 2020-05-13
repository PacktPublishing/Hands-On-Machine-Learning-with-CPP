#!/usr/bin/env bash
set -x
set -e

START_DIR=$(pwd)
REPOSITORY=$1
COMMIT_HASH=$2
shift; shift;
EXTRA_CMAKE_PARAMS=$@

cd $START_DIR/libs/sources
git clone $REPOSITORY
cd "$(basename "$REPOSITORY" .git)"
git checkout $COMMIT_HASH
git submodule update --init --recursive
mkdir build
cd build 
cmake -DCMAKE_INSTALL_PREFIX=$START_DIR/libs $EXTRA_CMAKE_PARAMS ..
cmake --build . --target install -- -j8
cd ..
rm -rf build
cd $START_DIR
