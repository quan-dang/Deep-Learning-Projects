# set the matplotlib backend so we can save our plot to disk
import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np 
import matplotlib.pyplot as plt
import argparse
from utilities.resnet import ResNet
from utilities import config
import keras 


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# define the total number of epochs, learning rate 
# and batch size for training
num_epochs = 20
lr = 1e-1
bs = 32 

# learning rate decay function to decay learning rate after each epoch
def poly_decay(epoch):
    # initialize the maximum number of epochs, base learning rate
    # and power of the polynomial
    maxEpochs = num_epochs
    baseLR = lr
    power = 1.0 

    # computer the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power 

    # return the new learning rate
    return alpha


# determine the number of image paths in training, validation, and testing directories
totalTrain = len(list(paths.list_images(config.TRAIN_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))
totalVal = len(list(paths.list_images(config.VAL_PATH)))

print("totalTrain: {}".format(totalTrain))
print("totalTest: {}".format(totalTest))
print("totalVal: {}".format(totalVal))

# initialize the training data augmentation object
trainAug = ImageDataGenerator(
	rescale=1 / 255.0,
	rotation_range=20,
	zoom_range=0.05,
	width_shift_range=0.05,
	height_shift_range=0.05,
	shear_range=0.05,
	horizontal_flip=True,
	fill_mode="nearest")
 
# initialize the validation and testing data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
	config.TRAIN_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=True,
	batch_size=bs)
 
# initialize the validation generator
valGen = valAug.flow_from_directory(
	config.VAL_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=bs)
 
# initialize the testing generator
testGen = valAug.flow_from_directory(
	config.TEST_PATH,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=bs)

# initialize the ResNet model
model = ResNet.build(64, 64, 3, 2, (3, 4, 6), (64, 128, 256, 512), reg=0.0005)

# compile the model
optimizer = SGD(lr=lr, momentum=0.9)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# define our set of callbacks and fit the model
callbacks = [LearningRateScheduler(poly_decay)]
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // bs,
	validation_data=valGen,
	validation_steps=totalVal // bs,
	epochs=num_epochs,
	callbacks=callbacks)


# reset the testing generator and use our trained model to predict on the data
print("Evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen, steps=(totalTest // bs) + 1)

# for each image in the testing set, find the index with maximum probability
predIdxs = np.argmax(predIdxs, axis=1)

# show the classification report
print(classification_report(testGen.classes, predIdxs,
                        target_names=testGen.class_indices.keys()))

# plot the training loss and accuracy
n = num_epochs

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, n), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, n), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, n), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, n), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])