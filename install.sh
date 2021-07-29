
# conda create --name MEGA -y python=3.7
# conda activate MEGA

# this installs the right pip and dependencies for the fresh python
conda install -y ipython pip

# mega and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python scipy

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 10.0
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install cityscapesScripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python setup.py build_ext install

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
# git clone https://github.com/Scalsol/mega.pytorch.git
# cd mega.pytorch

# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

pip install 'pillow<7.0.0'

unset INSTALL_DIR

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
