from keras.models import load_model
from utilities import config
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to our input image")
args = vars(ap.parse_args())

# load the input image
image = cv2.imread(args["image"])
output = image.copy()
output = imutils.resize(output, width=400)

# convert BGR to RGB and resize to 224x224
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))

# load the trained model from the disk
print("[INFO] loading model...")
model = load_model(config.MODEL_PATH)

# make prediction
preds = model.predict(np.expand_dims(image, axis=0))[0]
i = np.argmax(preds)
label = config.CLASSES[i]

# draw the prediction on the output image
text = "{}: {:.2f}%".format(label, preds[i] * 100)
cv2.putText(output, text, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
	(0, 255, 0), 2)
 
# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)

