# Captcha Breaker using LeNet

## Directories
1. __dataset__: contains labelled digits.

2. __downloads__: contains the raw captcha .jpg files downloaded from E-ZPass website.

3. __download_images__: download the example captchas and save to disk.

4. __annotate.py__: after having downloaded a set of captchas, we will extract the digits from each image and hand-label every digit.

5. __train_model.py__: train LeNet on the labeled digits.

6. __test_model.py__: apply LeNet to captcha images themselves.

7. __utilities__: utility functions.

## Dataset
This project retrieved captcha from [E-ZPass](https://www.e-zpassny.com/vector/jcaptcha.do) website. 

## How to run
1. __Step 1__: Download captcha images from E-Zpass URL

```
python download_images.py --output downloads
```

2. __Step 2__: Annotate manually captcha images

```
python annotate.py --input downloads --annot dataset
```

3. __Step 3__: train LeNet model on the dataset

```
python train_model.py --dataset dataset --model output/lenet.hdf5
```

4. __Step 4__: test LeNet model

```
python test_model.py --input downloads --model output/lenet.hdf5
```

