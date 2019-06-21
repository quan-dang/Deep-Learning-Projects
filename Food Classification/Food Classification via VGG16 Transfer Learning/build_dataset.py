from utilities import config 
from imutils import paths
import shutil
import os

# loop over our training, testing, and validation splits.
for split in (config.TRAIN, config.TEST, config.VAL):
    # grab all the image paths in the current split
    print("[INFO] processing '{} split'...".format(split))
    
    # create a list of all imagePaths in the split
    p = os.path.sep.join([config.ORIGIN_DATASET, split])
    imagePaths = list(paths.list_images(p))

    # loop over the image paths
    for imagePath in imagePaths:
        # extract class label from the filename
        filename = imagePath.split(os.path.sep)[-1]
        label = config.CLASSES[int(filename.split("_")[0])]

        # path to the output directory
        dirPath = os.path.sep.join([config.BASE_PATH, split, label])

        # if the output directory does not exist, then create it
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)

        # construct the path to the output image file and copy it
        p = os.path.sep.join([dirPath, filename])
        shutil.copy2(imagePath, p)

