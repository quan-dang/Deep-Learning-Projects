from sklearn.preprocessing import LabelEncoder
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from utilities import config
from imutils import paths
import numpy as np
import pickle
import random
import os

# load the VGG16 network 
print("[INFO] loading VGG16 network...")
model = VGG16(weights="imagenet", include_top=False)

# initialize the label encoder
le = None

"""
ready to do transfer learning
"""

# loop over the data splits
for split in (config.TRAIN, config.TEST, config.VAL):
    # grab all image paths in the current split
    print("[INFO] processing '{} split...'".format(split))

    # list all image paths
    p = os.path.sep.join([config.BASE_PATH, split])
    imagePaths = list(paths.list_images(p))

    # randomly shuffle the image paths, then extract 
    # the class labels from the file paths 
    random.shuffle(imagePaths)
    labels = [p.split(os.path.sep)[-2] for p in imagePaths]

    # if the label encoder is None, then create it
    if le is None:
        le = LabelEncoder()
        le.fit(labels)

    # open the output CSV fike for writing
    csvPath = os.path.sep.join([config.BASE_CSV_PATH, 
                            "{}.csv".format(split)])

    csv = open(csvPath, "w")

    # loop over the images in batches
    for (b, i) in enumerate(range(0, len(imagePaths), config.BATCH_SIZE)):
        # extract the batch of images and labels, 
        # then initialize the list of actual images
        print("[INFO] processing batch {}/{}".format(b+1, 
                int(np.ceil(len(imagePaths)/ float(config.BATCH_SIZE)))))
        batchPaths = imagePaths[i: i+config.BATCH_SIZE]
        batchLabels = le.transform(labels[i: i+config.BATCH_SIZE])
        batchImages = []

        # polulate out batchImages
        for imagePath in batchPaths:
            image = load_img(imagePath, target_size=(224, 224))
            image = img_to_array(image)

            # preprocess the image
            image = np.expand_dims(image, axis=0)
            image = imagenet_utils.preprocess_input(image)

            # add the image to the batch
            batchImages.append(image)

        # pass the images through the network 
        batchImages = np.vstack(batchImages)
        features = model.predict(batchImages, batch_size=config.BATCH_SIZE)
        features = features.reshape((features.shape[0], 7 * 7 * 512))

        # loop over the class labels and extracted features
        for (label, vec) in zip(batchLabels, features):
            # create a row containing both label and extracted features
            # in the csv file, label is the first field of each row, 
            # following by feature vec
            vec = ",".join([str(v) for v in vec])
            csv.write("{},{}\n".format(label, vec))

    # close the CSv file
    csv.close()

# serialize the label encoder to disk
f = open(config.LE_PATH, "wb")
f.write(pickle.dumps(le))
f.close()

