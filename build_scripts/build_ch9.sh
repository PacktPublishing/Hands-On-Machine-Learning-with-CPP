START_DIR=${PWD%/*}

# change this directory path according to your configuration
LIBS_DIR=/development/libs

#Chapter 9
cd $START_DIR/ch9/sharkml
mkdir build
cd build/
cmake -DSHARK_PATH=$LIBS_DIR -DPLOTCPP_PATH=$LIBS_DIR/sources/plotcpp/ ..
cmake --build . --target all

cd $START_DIR/ch9/shogun
mkdir build
cd build/
cmake -DSHOGUN_PATH=$LIBS_DIR -DPLOTCPP_PATH=$LIBS_DIR/sources/plotcpp/ ..
cmake --build . --target all
