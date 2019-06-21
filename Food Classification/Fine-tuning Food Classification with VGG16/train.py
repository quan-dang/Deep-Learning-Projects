# set the matplotlib backend to save figures in the background
import matplotlib
matplotlib.use("Agg")

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from utilities import config
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# function for plotting training history
def plot_training(H, N, plotPath):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plotPath)

# paths to the training, validation, and testing directories
trainPath = os.path.sep.join([config.BASE_PATH, config.TRAIN])
valPath = os.path.sep.join([config.BASE_PATH, config.VAL])
testPath = os.path.sep.join([config.BASE_PATH, config.TEST])

# determine the total number of image paths in training, validation,
# and testing directories
totalTrain = len(list(paths.list_images(trainPath)))
totalVal = len(list(paths.list_images(valPath)))
totalTest = len(list(paths.list_images(testPath)))

# define the training data augmentation object
trainAug = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# define the validation/testing data augmentation object
valAug = ImageDataGenerator()

# define the ImageNet mean subtraction (in RGB order)
mean = np.array([123.68, 116.779, 103.939], dtype="float32")

# set the mean subtraction value for each of the data augmentation
trainAug.mean = mean
valAug.mean = mean

"""
define generators that will load batches of images from their respective,
training, validation and testing splits, it helps to ensure the machine will
not load all the data at once
"""


# initialize the training generator
trainGen = trainAug.flow_from_directory(
	trainPath,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=True,
	batch_size=config.BATCH_SIZE)

# initialize the validation generator
valGen = valAug.flow_from_directory(
	valPath,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BATCH_SIZE)

# initialize the testing generator
testGen = valAug.flow_from_directory(
	testPath,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BATCH_SIZE)

 
# initialize the validation generator
valGen = valAug.flow_from_directory(
	valPath,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BATCH_SIZE)
 
# initialize the testing generator
testGen = valAug.flow_from_directory(
	testPath,
	class_mode="categorical",
	target_size=(224, 224),
	color_mode="rgb",
	shuffle=False,
	batch_size=config.BATCH_SIZE)

# load the VGG16 network
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top
# of the base model
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(config.CLASSES), activation="softmax")(headModel)

# place the head FC model on top of the base model
model = Model(inputs=baseModel.input, outputs=headModel)

# freeze all layers in the base model to prevent from being trained
for layer in baseModel.layers:
    layer.trainable = False

# compile the model
print("[INFO] compiling model...")
optimizer = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=optimizer,
	        metrics=["accuracy"])

# train the head of the network for a few epochs
print("[INFO] training head...")
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // config.BATCH_SIZE,
	validation_data=valGen,
	validation_steps=totalVal // config.BATCH_SIZE,
	epochs=50)

# reset the testing generator
print("[INFO] evaluating after fine-tuning network head...")
testGen.reset()

# evaluate our network on our testing data
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // config.BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)

# print classification statistics via terminal
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))

# plot the training history
plot_training(H, 50, config.WARMUP_PLOT_PATH)

# reset our data generators
trainGen.reset()
valGen.reset()

# unfreeze the final set of CONV layers
for layer in baseModel.layers[15:]:
	layer.trainable = True

# examine again which layer is trainable and not via terminal
for layer in baseModel.layers:
	print("{}: {}".format(layer, layer.trainable))

# re-compile the model
print("[INFO] re-compiling model...")
optimizer = SGD(lr=1e-4, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=optimizer,
	metrics=["accuracy"])
 
# train the model again
H = model.fit_generator(
	trainGen,
	steps_per_epoch=totalTrain // config.BATCH_SIZE,
	validation_data=valGen,
	validation_steps=totalVal // config.BATCH_SIZE,
	epochs=20)

# reset the testing generator
print("[INFO] evaluating after fine-tuning network...")
testGen.reset()

# evaluate our network on our testing data
predIdxs = model.predict_generator(testGen,
	steps=(totalTest // config.BATCH_SIZE) + 1)
predIdxs = np.argmax(predIdxs, axis=1)

# print classification statistics via terminal
print(classification_report(testGen.classes, predIdxs,
	target_names=testGen.class_indices.keys()))

# plot the training history
plot_training(H, 20, config.UNFROZEN_PLOT_PATH)
 
# serialize the model to disk
print("[INFO] serializing network...")
model.save(config.MODEL_PATH)