START_DIR=${PWD%/*}

# change these directories paths according to your configuration, also change them in the ch13/android_classify/local.proreties file
ANDROID_SDK_DIR=/development/android
ANDROID_PYTORCH_DIR=/development/android/pytorch/build_android/install

#Chapter 13
cd $START_DIR/ch13/python
./export.sh
cp model.pt $START_DIR/ch13/android_classify/app/src/main/assets

cd $START_DIR/ch13/android_classify
mkdir app/src/main/jniLibs
mkdir app/src/main/jniLibs/armeabi-v7a
cp $ANDROID_PYTORCH_DIR/lib/libc10.so app/src/main/jniLibs/armeabi-v7a/
cp $ANDROID_PYTORCH_DIR/lib/libtorch.so app/src/main/jniLibs/armeabi-v7a/

./gradlew build

# Find resulting APK file in the ch13/android_classify/app/build/outputs/apk/release/ folder
# Notice that this script may fail if you run it into Docker container under Windows platform