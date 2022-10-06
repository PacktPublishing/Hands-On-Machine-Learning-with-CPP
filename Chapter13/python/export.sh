#! /bin/bash

wget https://raw.githubusercontent.com/onnx/models/master/vision/classification/synset.txt

#python version should be less then 3.8 i.e. 3.6 or 3.7 because pytorch1.2 doesn't support python 3.8 >
python3.6 -m venv venv
source venv/bin/activate
pip install numpy
pip install pillow
pip install torchvision==0.10.0
python model_export.py
deactivate
