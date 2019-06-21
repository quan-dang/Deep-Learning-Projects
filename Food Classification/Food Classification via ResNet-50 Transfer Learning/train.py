from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from utilities import config
import numpy as np
import pickle
import os
 
"""
implement custom Keras data generators to yield data from a CSV file 
to train a NN with Keras with the assumption that the CSV file does 
not fit into the memory
""" 
def csv_feature_generator(inputPath, bs, numClasses, mode="train"):
    """
    yield batches of labels + data to the network
    
    params:
        inputPath: path to out input CSV containing the extracted features
        bs: batch size
        numClasses : number of classes in our data 
    
    """
    # open the input file for reading
    f = open(inputPath, "r")

    while True:
        # initialize our batch of data and labels
        data = []
        labels = []

        # keep looping util we reach our batch size
        while len(data) < bs:
            # attempt to retrieve the next row of the CSV file
            row = f.readline()

            # if the row is empty, it means we reached the end of the file
            if row == "":
                # reset the pointer to the beginning of the file
                # and re-read the row
                f.seek(0)
                row = f.readline()

                # if mode == "eval", we stop filling up the batch from 
                # samples at the beginining of the file
                if mode == "eval":
                    break
            
            # extract the class label and features from the row
            row = row.strip().split(",")
            label = row[0]
            label = to_categorical(label, num_classes=numClasses)
            features = np.array(row[1:], dtype="float")

            # update the data and label lists
            data.append(features)
            labels.append(label)

        # yield the batch
        yield(np.array(data), np.array(labels))

# load the label encoder from the disk
le = pickle.loads(open(config.LE_PATH, "rb").read())

# retrieve the paths to the training and testing csv files 
trainPath = os.path.sep.join([config.BASE_CSV_PATH,
                "{}.csv".format(config.TRAIN)])
valPath = os.path.sep.join([config.BASE_CSV_PATH,
                "{}.csv".format(config.VAL)])
testPath = os.path.sep.join([config.BASE_CSV_PATH,
                "{}.csv".format(config.TEST)])

# determine the total number of images in the training and validation sets
totalTrain = sum([1 for l in open(trainPath)])
totalVal = sum([1 for l in open(valPath)])

# extract the testing labels from the CSV file
# then determine the number of testing images
testLabels = [int(row.split(",")[0]) for row in open(testPath)]
totalTest = len(testLabels)

# construct the training, validation, and testing generators
trainGen = csv_feature_generator(trainPath, config.BATCH_SIZE,
	len(config.CLASSES), mode="train")
valGen = csv_feature_generator(valPath, config.BATCH_SIZE,
	len(config.CLASSES), mode="eval")
testGen = csv_feature_generator(testPath, config.BATCH_SIZE,
	len(config.CLASSES), mode="eval")

# define our model
model = Sequential()
model.add(Dense(256, input_shape=(7 * 7 * 2048,), activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(len(config.CLASSES), activation="softmax"))

# compile the model
opt = SGD(lr=1e-3, momentum=0.9, decay=1e-3 / 25) # lr will decay over 25 epochs
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training simple network...")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // config.BATCH_SIZE,
	validation_data=valGen,
	validation_steps=totalVal // config.BATCH_SIZE,
	epochs=25)
 
# make predictions on the testing images
print("[INFO] evaluating network...")
predIdxs = model.predict_generator(testGen,
	steps=(totalTest //config.BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testLabels, predIdxs,
	target_names=le.classes_))
