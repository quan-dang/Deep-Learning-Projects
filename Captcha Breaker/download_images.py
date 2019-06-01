import os
import time
import argparse
import requests

# construct the argument parse and parse the arguments
# --output: path to the output directory that will store our raw captcha images
# --num_images: number of captcha images we’re going to download
# there are 4 digits in each image (captcha)
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
                help="path to output directory of images")
ap.add_argument("-n", "--num_images", type=int, default=500,
                help="number of images to download")
args = vars(ap.parse_args())

# initialize the URL of the captcha image we are going to download along
# with the total number of images generated thus far
url = "https://www.e-zpassny.com/vector/jcaptcha.do"
total = 0

# loop over the number of images to download
for i in range(0, args["num_images"]):
    try:
        # try to grab a new captcha image
        r = requests.get(url, timeout=60)

        # save the image to disk
        p = os.path.sep.join([args["output"], "{}.jpg".format(str(total).zfill(5))])
        f = open(p, 'wb')
        f.write(r.content)
        f.close()

        # update the counter
        print("Downloaded: {}".format(p))
        total += 1

    except:
        print("Error downloading image...")

    # insert a small sleep to be better for the server
    time.sleep(0.1)