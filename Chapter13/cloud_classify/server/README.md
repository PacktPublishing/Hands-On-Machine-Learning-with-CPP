1. Go to GCP -> VM Instances

2. Create new instance
   ```
   Name: classify-server
   Zone: choose appropriate to you, us-central1-a
   Generation: first
   Machine-type: n1-standard-1
   CPU platform: automatic
   Hardware configuration: Choose at lease 4 cores and 16GB of RAM
   Optionaly: Add GPU
   Boot disk: New 10 GB standard persistent disk
   Image: Debian GNU/Linux 9 (stretch)
   Identity and API access: Compute Engine default service account
   Access scopes: Allow default access
   Firewall: Allow HTTP traffic
   ```

3. On a web page with a list of VM instances press SSH in the row with instance you created. 
   It will open a new browser window with a command line session.

4. Install development packages:
   ```
   sudo apt-get install git
   sudo apt-get install cmake
   sudo apt-get install g++
   sudo apt-get install libopencv-dev
   sudo apt-get install libprotobuf-dev
   sudo apt-get install unzip
   sudo apt-get install python-pip
   sudo apt-get install libopenblas-dev
   sudo apt-get install pybind11-dev
   pip install pyyaml
   pip install typing
   ```

5. Install GCP SDK to the local computer
   1. Go to the `https://cloud.google.com/sdk/docs/` URL
   2. Make sure that your system has Python 2 with a release number of Python 2.7.9 or higher.
   3. Download SDK archive
   4. Extract the contents of the file to any location on your file system. If you would like to replace an existing installation, remove the existing google-cloud-sdk directory and extract the archive to the same location.
   5. Run gcloud init to initialize the SDK:
      ```
      ./google-cloud-sdk/bin/gcloud init
      ```
      The sample init session :
      ```
      You must log in to continue. Would you like to log in (Y/n)?  y
      Your browser has been opened to visit:
        https://accounts.google.com/o/oauth2/auth?...
      Pick cloud project to use: 
        [1] hardy-aleph-253219
        [2] Create a new project
      Please enter numeric choice or text value (must exactly match list item):  1
      ```
      Select default choices during the init session.

6. Go to the `Chapter13/python` directory and execute `export.sh` script to download the `synset.txt` file and and create the `model.pt` file.

7. Copy applications source code(the client and server) to the remote instance
   ```
   gcloud compute scp --recurse [LOCAL_FILE_PATH] [INSTANCE_NAME]:~/[DEST_PATH]
   ```

   Copy model snapshoot to the remote instance:
   ```
   gcloud compute scp [LOCAL_FILE_PATH]/model.pt [INSTANCE_NAME]:~/[DEST_PATH]/model
   gcloud compute scp [LOCAL_FILE_PATH]/synset.txt [INSTANCE_NAME]:~/[DEST_PATH]/model
   ```
   make sure you use same user name on local machine and in the cloud instanse, otherwize copoes will be placed in different directory

8. Switch to the remote instance, and navigate to the development folder where you placed the application source code:
   ```
   cd ~/[DEST_PATH]/server
   ```

9. Clone the http server third-party library:
   ```
   git clone https://github.com/yhirose/cpp-httplib third-party/httplib
   ```
10.  Clone the PyTorch third-party library:
    ```
    cd third-party
    wget --no-check-certificate https://download.pytorch.org/libtorch/cu100/libtorch-shared-with-deps-1.2.0.zip
    unzip libtorch-shared-with-deps-1.2.0.zip
    cd ..
    ```

11. Build PyTorch from sources, because official binaries requires CUDA which can be missed on the server instance:
    ```
    cd third-party
    git clone https://github.com/pytorch/pytorch.git
    cd pytorch/
    git checkout v1.2.0
    git submodule update --init
    mkdir build
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=~/dev/server/third-party/pytorch -DUSE_CUDA=OFF -DUSE_CUDNN=OFF -DUSE_OPENMP=ON -DBUILD_TORCH=ON -DUSE_FBGEMM=OFF -DBUILD_PYTHON=OFF
    cmake --build . --target install -- -j8
    ```

12. Build our server apllication:
    ```
    cd ~/[DEST_PATH]/server
    mkdir build
    cd build
    cmake .. -DCMAKE_PREFIX_PATH=~/dev/server/third-party/pytorch
    cmake --build . --target all
    ```
    
13. Open GCP console and create a firewall rule to allow client requests to the instance:
    1. Go to the VPC network.
    2. Go to the Firewall rules page
    3. In the Create a firewall rule page, enter the following information:
    ```
        Name: classify-server
        Target tags: http-server
        Actions on match: allow
        Source IP ranges: 0.0.0.0/0
        Protocol and ports: tcp:8080
    ```
    4. Click Create.

14. Find the instance IP addresses in the GCP console. There are two addresses the internal and external.
    Got to the: `Compute Engine->VM Instances->three dots` on selected `instance-> View Network details`
    Remember these addresses

15. On the remove instance start the server application:
    ```
    ./classify-server ~/[DEST_PATH]/model/model.pt ~/[DEST_PATH]/model/synset.txt ~/[DEST_PATH]/client/ [internal ip] 8080
    ```

16. On the remote instance update client's source code the `upload.js` file with the instance external IP address. Change the value of the "url" variable.

17. On the local computer open the `http://[external ip]:8080` URL in the browser.

