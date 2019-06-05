from imutils import paths
import shutil
import random
import os 
from utilities import config

# retrieve the paths to all input images and shuffle them
imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)

# compute the training and testing split
index = int(len(imagePaths) * config.TRAIN_SPLIT)
trainPaths = imagePaths[:index]
testPaths = imagePaths[index:]

# using part of the training data for validation
index = int(len(trainPaths) * config.VAL_SPLIT)
valPaths = trainPaths[:index]
trainPaths = trainPaths[index:]

# define the datasets
datasets = [
    ("training", trainPaths, config.TRAIN_PATH),
    ("validation", valPaths, config.VAL_PATH),
    ("testing", testPaths, config.TEST_PATH)
]

# loop over the datasets
for (dtype, imagePaths, baseOutput) in datasets:
    print("Building '{}' split".format(dtype))

    # if the output base directory does not exist, create it
    if not os.path.exists(baseOutput):
        print("Creating '{}' directory".format(baseOutput))
        os.makedirs(baseOutput)

    # loop over the input image paths
    for inputPath in imagePaths:
        # extract the filename of the input image
        # along with its corresponding label
        filename = inputPath.split(os.path.sep)[-1]
        label = inputPath.split(os.path.sep)[-2]

        # build the path to the label directory
        labelPath = os.path.sep.join([baseOutput, label])

        # if the label output directory does not exist, create it
        if not os.path.exists(labelPath):
            print("Creating '{}' directory".format(labelPath))
            os.makedirs(labelPath)

        # construct the path to the destination image
        # and then copy the image itself
        p = os.path.sep.join([labelPath, filename])
        shutil.copy2(inputPath, p)

        