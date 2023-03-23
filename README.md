# bsiPytorch
A repo to implement functionalities of bsiCPP in PyTorch to do performance testing/verification.

## Setup

1. Create a conda environment and install pytorch
```
conda create -n <env_name>
conda activate <env_name>
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Note: we can probably skip cuda for now since we are running only cpu operations, but this is in case we want to extend to gpu soon


2. Clone and initialise this repo
```
git clone https://github.com/raspuchin/bsiPytorch.git
cd bsiPytorch
git submodule update --init --recursive
```

## How to build and run operations
Done using the guide provided by PyTorch [here](https://pytorch.org/tutorials/advanced/cpp_extension.html).

Commands that are to be used
```
# use the commands in the bsiPytorch/bsi_ops folder

# to install/build
python setup.py install

# to remove/clean
pip uninstall bsi_ops -y && python setup.py clean

# to see if bsi_ops is installed correctly
python test.py
```


# Testing bsi_ops on a neural network load
```
# Do after installing bsi_ops
python bsiPytorch/model_test/iris_test.py
```

iris_test.py loads the iris dataset and trains a simple SVC model. If we open the file we can see we can change which dot product function is being used and we can build our own dot product to replace the one available.

Copied from [here](https://github.com/mtrencseni/pytorch-playground/blob/master/03-svm/SVM%20with%20Pytorch.ipynb)
# Current bugs/issues in bsi_ops
1. Tensor is converted to bsi by converting it to a vector first, a direct conversion would be faster.
2. Not yet implemented a method to convert a tensor with "float" or "double" to bsi, we can run iris test properly with that.