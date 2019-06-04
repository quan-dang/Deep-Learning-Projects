# Image Caption Generator
The basic idea is that the features are extracted from the images using pre-trained VGG16 model, and then fed to the LSTM model along with the captions 
to train.

## How to run

__0. Step 0:__ Download [VGG16 weights](https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5) 
and put it into your root folder

__1. Step 1:__ Follow the IPython Notebook