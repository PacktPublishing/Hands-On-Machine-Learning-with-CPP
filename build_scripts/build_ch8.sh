START_DIR=${PWD%/*}

# change this directory path according to your configuration
LIBS_DIR=/development/libs

#Chapter 8
cd $START_DIR/ch8/eigen
mkdir build
cd build/
cmake -DEIGEN_PATH=$LIBS_DIR/include/eigen3 ..
cmake --build . --target all

cd $START_DIR/ch8/mlpack
mkdir build
cd build/
cmake -DMLPACK_PATH=$LIBS_DIR ..
cmake --build . --target all

