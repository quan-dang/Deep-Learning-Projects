# Smile Detection using LeNet

## System requirement 
Verified on Windows 10 with GTX 1060 6GB and 16GB RAM. 

## Dataset

This project uses [SMILES dataset](https://github.com/hromi/SMILEsmileD), which consists of images of faces that are either smiling or not smiling. In total,
there are 13,165 grayscale images in the dataset, with each image having a size of 64×64 pixels.


## How to run
0. __Step 0__: Preparation

Download the dataset, to 'datasets' folder under the root path of the project, name the dataset as 'SMILEsmileD' which contains 3 subfolders 'appz', 'smileD' and 'SMILEs'.

1. __Step 1__: Train the LeNet model on the SMILEs dataset

```
python train_model.py --dataset ../datasets/SMILEsmileD --model output/lenet.hdf5
```
2. __Step 2__: Test the model

- Using webcam:

```
python detect_smile.py --cascade haarcascade_frontalface_default.xml --model output/lenet.hdf5
```

- Using a video:

```
python detect_smile.py --cascade haarcascade_frontalface_default.xml --model output/lenet.hdf5 --video quan.mp4
```
