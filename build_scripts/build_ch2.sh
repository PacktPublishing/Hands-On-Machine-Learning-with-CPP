START_DIR=${PWD%/*}

# change this directory path according to your configuration
LIBS_DIR=/development/libs


#Chapter 2
cd $START_DIR/Chapter02/csv/cpp
mkdir build
cd build/
cmake -DCSV_LIB_PATH=$LIBS_DIR/sources/fast-cpp-csv-parser/ -DEIGEN_LIB_PATH=$LIBS_DIR/include/eigen3/ ..
cmake --build . --target all

cd $START_DIR/Chapter02/csv/dlib
mkdir build
cd build/
cmake -DDLIB_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter02/csv/sharkml
mkdir build
cd build/
cmake -DSHARK_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter02/csv/shogun
mkdir build
cd build/
cmake -DSHOGUN_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter02/hdf5/cpp
mkdir build
cd build/
cmake -DHIGHFIVE_LIB_PATH=$LIBS_DIR/include/ -DJSON_LIB_PATH=$LIBS_DIR/include/rapidjson/ ..
cmake --build . --target all

cd $START_DIR/Chapter02/json/cpp
mkdir build
cd build/
cmake -DJSON_LIB_PATH=$LIBS_DIR/include/ -DEIGEN_LIB_PATH=$LIBS_DIR/include/eigen3/ ..
cmake --build . --target all

cd $START_DIR/Chapter02/img/dlib/
mkdir build
cd build/
cmake -DDLIB_PATH=$LIBS_DIR ..
cmake --build . --target all

cd $START_DIR/Chapter02/img/opencv/
mkdir build
cd build/
cmake ..
cmake --build . --target all

