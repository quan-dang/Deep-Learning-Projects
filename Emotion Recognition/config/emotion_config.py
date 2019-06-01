from os import path

# define the base path to the emotion dataset
BASE_PATH = "./fer2013/"

# use the base path to define the path to the input emotions file
INPUT_PATH = path.sep.join([BASE_PATH, "fer2013/fer2013.csv"])

# there are totally 7 classes in FER13, i.e.,  angry, disgust, fear, happy, sad, surprise, and neutral 
# however in "disgust" class, there are only 113 images, while each other class has 1000 images
# define the number of classes (set to 6 if you are ignoring the "disgust" class)
# NUM_CLASSES = 7
NUM_CLASSES = 6


# define the path to the output training, validation, and testing HDF5 files
TRAIN_HDF5 = path.sep.join([BASE_PATH, "hdf5/train.hdf5"])
VAL_HDF5 = path.sep.join([BASE_PATH, "hdf5/val.hdf5"])
TEST_HDF5 = path.sep.join([BASE_PATH, "hdf5/test.hdf5"])

# define the batch size
BATCH_SIZE = 128

# define the path to where output logs will be stored
OUTPUT_PATH = path.sep.join([BASE_PATH, "output"])