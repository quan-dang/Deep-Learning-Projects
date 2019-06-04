# Correcting Image Orientation

## Project description
The project use VGG16 network architecture to extract features from the dataset, and then apply Logistic Regression classifier to correct rotated images.

## Dataset
This project use the Indoor Scene Recognition ([Indoor CVPR](http://web.mit.edu/torralba/www/indoor.html)) dataset released by MIT, which contains 67 indoor categories such as homes, offices, stores, .etc.


## System requirement
Verified on Windows 10, with GTX 1060 6GB and RAM 16GB

## Directory description

1. __indoor_cvpr/hdf5__: folder to store the features extracted from our input images using a pre-trained CNN.

2. __indoor_cvpr/images__: folder containing images from Indoor CVPR dataset.

3. __indoor_cvpr/rotated_images__: randomly rotated images that are from __indoor_cvpr/images__.

4. __build_dataset.py__:  build the training and testing sets for our input dataset.

5. __extract_features.py__: create an HDF5 file for the dataset splits.

6. __orient_images.py__: orient testing input images.

7. __train_model.py__:  train a Logistic Regression classifier to recognize image orientations.

8. __models/__: folder to save our model.

9. __utilities/__: folder contains utility functions

## How to run

0. __Step 0__: Download the dataset, extract, copy and rename 'Images' folder to 'images', then put it under __indoor_cvpr/__.

1. __Step 1__: Build the dataset

```
python build_dataset.py --dataset indoor_cvpr/images --output indoor_cvpr/rotated_images
```
2. __Step 2__: Extract features from our rotated images dataset

```
python extract_features.py --dataset indoor_cvpr/rotated_images --output indoor_cvpr/hdf5/orientation_features.hdf5
```
3. __Step 3__: Train our Logistic Regression classifier on the extracted features from VGG16 network

```
python train_model.py --db indoor_cvpr/hdf5/orientation_features.hdf5 --model models/orientation.cpickle
```
4. __Step 4__: Test the performance of our model

```
python orient_images.py --db indoor_cvpr/hdf5/orientation_features.hdf5 --dataset indoor_cvpr/rotated_images --model models/orientation.cpickle
```