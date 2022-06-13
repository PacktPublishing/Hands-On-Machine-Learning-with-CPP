# Building development environment
There two main approaches to build development environment:
1. Configure local computer environment
2. Create a Docker image and container  

# Configure you GitHub account first
To be able to clone 3rd-party repositories you need a [GitHub](https://github.com) account. Then you will be able to configure GitHub authenticating with SSH as it is described in the article [Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) this is the preffered way. Or using HTTPS and providing your username and password each time when a new repository will be cloned. Also, If you use 2FA to secure your GitHub account then you’ll need to use a personal access token instead of a password, as explained in the article [Creating a personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token).

# Building development environment with Docker
1. Build Docker image.
```
cd env_scripts
docker build -t buildenv:1.0 .
```

2. Run the Docker container and build third party libaries there.
```
docker run -it buildenv:1.0 bash
cd /development
./install_env.sh
./install_android.sh
```

3. Identify active container ID (the following commands can be executed in the separate console session).
```
docker container ls
```
remember the container ID.

3. Save the container as new a image.
```
docker commit [container id]
```

4. Identify new image ID.
```
docker image ls
```
remember the image ID.

5. Give the name for the new image.
```
docker tag [image id] buildenv_libs`
```

6. Stop the initial container in the original console session.
```
exit
```

7. Run a new container from the created image, and share code samples folder.
```
docker run -it -v [host_samples_path]:[container_samples_path] buildenv_libs bash
```

8. Samples from chapter 2 require accsess to your graphical environment to show images. You can share you X11 server with a Docker container. The following script shows how to run a container with graphics environment:
```
xhost +local:root
docker run --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -it -v [host_samples_path]:[container_samples_path] buildenv_libs bash
```

9. In the started container console session navigate to the `container_samples_path\build_scripts` folder. You will find there scripts to build samples for each chapter. The following script shows how to build samples for chapter 1:
```
./build_ch1.sh
```

# Building local development environment

1. Install Ubuntu 18.04

2. Run the following commands to configure the system:
```
apt-get install -y build-essential
apt-get install -y gdb
apt-get install -y git
apt-get install -y cmake
apt-get install -y cmake-curses-gui
apt-get install -y python
apt-get install -y python-pip
apt-get install -y libblas-dev
apt-get install -y libopenblas-dev
apt-get install -y libatlas-base-dev
apt-get install -y liblapack-dev
apt-get install -y libboost-all-dev
apt-get install -y libopencv-core3.2
apt-get install -y libopencv-imgproc3.2
apt-get install -q -y libopencv-dev
apt-get install -y libopencv-highgui3.2
apt-get install -y libopencv-highgui-dev
apt-get install -y libhdf5-dev
apt-get install -y libjson-c-dev
apt-get install -y libx11-dev
apt-get install -y openjdk-8-jdk
apt-get install -y wget
apt-get install -y ninja-build
apt-get install -y gnuplot
apt-get install -y vim
apt-get install -y python3-venv
pip install pyyaml
RUN pip install typing
```

3. Create build environment with the following commands \(We assume that the path "/path/to/examples/package/" contains extracted code samples package\):
```
cd ~/
mkdir development
cd ~/development
cp /path/to/examples/package/docker/checkout_lib.sh ~/development
cp /path/to/examples/package/docker/install_lib.sh ~/development
cp /path/to/examples/package/docker/install_env.sh ~/development
cp /path/to/examples/package/docker/install_android.sh ~/development
chmod 777 ~/development/checkout_lib.sh
chmod 777 ~/development/install_lib.sh
chmod 777 ~/development/install_env.sh
chmod 777 ~/development/install_android.sh
./install_env.sh
./android_env.sh
```

4. All third party libraries will be installed into the following directory:
```
$HOME/development/libs
```

5. Navigate to the `/path/to/examples/package/build_scripts` folder.

6. Choose the build script for the chapter you want to build, for example build script for the first chapter is `build_ch1.sh`

7. Updated the `LIBS_DIR` varibale in the script with the `$HOME/development/libs` value, or another one but it should the folder where all third party libraries are installed.

8. Run the build script to compile samples for the selected chapter.

# List of all third-party libraries
Name - commit hash - branch name - repository

Shogun - f7255cf2cc6b5116e50840816d70d21e7cc039bb - master - https://github.com/shogun-toolbox/shogun

SharkML - 221c1f2e8abfffadbf3c5ef7cf324bc6dc9b4315 - master - https://github.com/Shark-ML/Shark

Armadillo - 442d52ba052115b32035a6e7dc6587bb6a462dec- branch 9.500.x - https://gitlab.com/conradsnicta/armadillo-code

DLib - 929c630b381d444bbf5d7aa622e3decc7785ddb2 - v19.15 - https://github.com/davisking/dlib

Eigen - cf794d3b741a6278df169e58461f8529f43bce5d - 3.3.7 - https://github.com/eigenteam/eigen-git-mirror

mlpack - e2f696cfd5b7ccda2d3af1c7c728483ea6591718 - master - https://github.com/mlpack/mlpack

plotcpp - c86bd4f5d9029986f0d5f368450d79f0dd32c7e4 - master - https://github.com/Kolkir/plotcpp

PyTorch - 8554416a199c4cec01c60c7015d8301d2bb39b64 - v1.2.0 - https://github.com/pytorch/pytorch

xtensor - 02d8039a58828db1ffdd2c60fb9b378131c295a2 - master - https://github.com/xtensor-stack/xtensor

xtensor-blas - 89d9df93ff7306c32997e8bb8b1ff02534d7df2e - master - https://github.com/xtensor-stack/xtensor-blas

xtl - 03a6827c9e402736506f3ded754e890b3ea28a98 - master - https://github.com/xtensor-stack/xtl

OpenCV 3 - from the distribution installation package - https://github.com/opencv/opencv_contrib/releases/tag/3.3.0

fast-cpp-csv-parser - 3b439a664090681931c6ace78dcedac6d3a3907e - master - https://github.com/ben-strasser/fast-cpp-csv-parser

RapidJson - 73063f5002612c6bf64fe24f851cd5cc0d83eef9 - master - https://github.com/Tencent/rapidjson

