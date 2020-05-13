START_DIR=${PWD%/*}

# change this directory path according to your configuration
LIBS_DIR=/development/libs


# Chapter 1
cd $START_DIR/ch1/dlib_samples/
mkdir build 
cd build/
cmake -DDLIB_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/ch1/eigen_samples/
mkdir build
cd build
cmake -DEIGEN_PATH=$LIBS_DIR/include/eigen3/ ..
cmake --build . --target all

cd $START_DIR/ch1/sharkml_samples/
mkdir build
cd build/
cmake -DSHARK_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/ch1/shogun_samples/
mkdir build
cd build/
cmake -DSHOGUN_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/ch1/xtensor_samples/
mkdir build
cd build/
cmake -DXTENSOR_PATH=$LIBS_DIR/include/ ..
cmake --build . --target all

