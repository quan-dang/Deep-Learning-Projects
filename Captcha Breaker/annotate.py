from imutils import paths
import argparse
import imutils
import cv2
import os

# construct the argument parse and parse the arguments
# --input: The input path to our raw captcha images (i.e., the downloads directory)
# --annot: The output path to where we’ll be storing the labeled digits (i.e., the dataset directory).
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to input directory of images")
ap.add_argument("-a", "--annot", required=True,
                help="path to output directory of annotations")
args = vars(ap.parse_args())

# grab the image paths then initialize the dictionary of character counts
imagePaths = list(paths.list_images(args["input"]))
counts = {}

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # only for windows, if you're using other OS, please comment the right below line
    imagePath = imagePath.replace("\\", "/")

    # display an update to the user
    print("Processing image {}/{}".format(i + 1, len(imagePaths)))

    try:
        # load the image and convert it to grayscale, then pad the
        # image to ensure digits caught on the border of the image
        # are retained
        image = cv2.imread(imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # threshold the image to reveal the digits
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # find contours in the image, keeping only the four largest ones
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:4]

        # loop over the contours
        for c in cnts:
            # compute the bounding box for the contour to extract the digit
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y - 5:y + h + 5, x - 5:x + w + 5]

            # display the character
            cv2.imshow("ROI", imutils.resize(roi, width=28))
            key = cv2.waitKey(0)

            # if the "``" key is presses, then ignore the character
            if key == ord("`"):
                print("Ignoring the character!")
                continue

            # grab the key that was pressed and construct the path (i.e., output directory)
            key = chr(key).upper()
            dirPath = os.path.sep.join([args["annot"], key])

            # if the output directory does not exist, then create it
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)

            # write the labeled character to file
            count = counts.get(key, 1)
            p = os.path.sep.join([dirPath, "{}.png".format(str(count).zfill(6))])

            cv2.imwrite(p, roi)

            # increase the count for the current key
            counts[key] = count + 1

    # break the loop by Ctrl-C 
    except KeyboardInterrupt:
        print("Manually leaving script...")
        break

    # unknown error occurs
    except Exception as e:
        print(e)