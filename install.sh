#! /bin/bash

root=`pwd`

# Install dependecies
pip install numpy matplotlib pillow scipy tqdm scikit-learn
pip install torchdiffeq==0.0.1
pip install open3d==0.9

# Compile CUDA kernel for CD/EMD loss
cd metrics/pytorch_structural_losses/
make clean
make
cd $root

