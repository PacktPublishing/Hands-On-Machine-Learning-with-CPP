#!/usr/bin/env bash
set -x
set -e

START_DIR=$(pwd)
REPOSITORY=$1
COMMIT_HASH=$2
shift; shift;
EXTRA_CMAKE_PARAMS=$@

cd $START_DIR/libs/sources

if [ "$REPOSITORY" = "https://gitlab.com/conradsnicta/armadillo-code" ] && [ "$COMMIT_HASH" = "48b45db6ca1d2839e42332dcdf04f55dcec83e20" ]; then
    ZIP_URL="https://gitlab.com/conradsnicta/armadillo-code/-/archive/48b45db6ca1d2839e42332dcdf04f55dcec83e20/armadillo-code-48b45db6ca1d2839e42332dcdf04f55dcec83e20.zip"
    ZIP_FILE="armadillo-code-48b45db6ca1d2839e42332dcdf04f55dcec83e20.zip"
    DIR_NAME="armadillo-code-48b45db6ca1d2839e42332dcdf04f55dcec83e20"

    wget "$ZIP_URL" -O "$ZIP_FILE"
    unzip "$ZIP_FILE"
    rm "$ZIP_FILE"
    cd "$DIR_NAME"
elif [ "$REPOSITORY" = "https://github.com/mlpack/mlpack" ] && [ "$COMMIT_HASH" = "e2f696cfd5b7ccda2d3af1c7c728483ea6591718" ]; then
    ENSMALLEN_URL="https://www.ensmallen.org/files/ensmallen-1.10.0.tar.gz"
    ENSMALLEN_FILE="ensmallen-1.10.0.tar.gz"
    ENSMALLEN_DIR="ensmallen-1.10.0"

    wget "$ENSMALLEN_URL" -O "$ENSMALLEN_FILE"
    tar -xzf "$ENSMALLEN_FILE"
    rm "$ENSMALLEN_FILE"
    cd "$ENSMALLEN_DIR"
    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX="$START_DIR/libs" ..
    cmake --build . --target install -- -j8
    cd ../..
    rm -rf "$ENSMALLEN_DIR"

    git clone "$REPOSITORY"
    cd "$(basename "$REPOSITORY" .git)"
    git checkout "$COMMIT_HASH"

    if [ -f ".gitmodules" ]; then
        sed -i 's/git:\/\//https:\/\//g' ".gitmodules"
        git submodule update --init --recursive
    fi
else
    git clone "$REPOSITORY"
    cd "$(basename "$REPOSITORY" .git)"
    git checkout "$COMMIT_HASH"

    if [ -f ".gitmodules" ]; then
        sed -i 's/git:\/\//https:\/\//g' ".gitmodules"
        git submodule update --init --recursive
    fi
fi

mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$START_DIR/libs $EXTRA_CMAKE_PARAMS ..
cmake --build . --target install -- -j8
cd ..
rm -rf build
cd $START_DIR
