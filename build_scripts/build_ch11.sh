START_DIR=${PWD%/*}

# change this directory path according to your configuration
LIBS_DIR=/development/libs

#Chapter 11
cd $START_DIR/ch11/pytorch
mkdir build
cd build/
cmake -DCMAKE_INSTALL_PREFIX=$LIBS_DIR ..
cmake --build . --target all

# Please use the following command line to run the pytorch sample:
# LD_LIBRARY_PATH=$LIBS_DIR/sources/pytorch/third_party/ideep/mkl-dnn/external/mklml_lnx_2019.0.3.20190220/lib/ ./rnn-pytorch

