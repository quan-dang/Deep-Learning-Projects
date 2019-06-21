import os

# path to our original dataset directory
ORIGIN_DATASET = "Food-5K"

# path to the new directory containing our images 
# after the training and testing split
BASE_PATH = "dataset"

# names of training, testing, validation directories
TRAIN = "training"
TEST = "evaluation"
VAL = "validation"

# list of class label names
CLASSES = ["non_food", "food"]

# set the batch size
BATCH_SIZE = 32

# label encoder path
LE_PATH = os.path.sep.join(["output", "le.cpickle"])

# output directory to store extracted features (in .csv format)
BASE_CSV_PATH = "output"

# path to the serialized model after training
MODEL_PATH = os.path.sep.join(["output", "model.cpickle"])

