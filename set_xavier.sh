!# bin/bash


echo 'start install pytorch'
sudo apt-get update

sudo apt-get -y install python3-pip libopenblas-base libopenmpi-dev 
pip3 install Cython
#pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl
#rm  torch-1.8.0-cp36-cp36m-linux_aarch64.whl
pip3 install numpy torch-1.7.0-cp36-cp36m-linux_aarch64.whl
rm  torch-1.7.0-cp36-cp36m-linux_aarch64.whl


echo 'pytorch sucsessfully installed'

echo 'start install torchvision'
sudo apt-get update

sudo apt-get -y install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.8.1 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.8.0
sudo python3 setup.py install
cd ..
pip install 'pillow<7'

echo 'torchvision sucsessfully installed'

echo 'install yolo requirements'
sudo apt-get update
sudo apt-get install -y build-essential libatlas-base-dev
