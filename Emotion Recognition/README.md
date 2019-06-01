# Emotion Recognition using EmotionVGGNet


## System requirement
Verified on Windows 10 with GTX 1060 6GB and 16GB RAM


## Project structure

1. __config/emotion_config.py__: store configuration variables, including paths to the input dataset, output HDF5 files, and batch sizes.

2. __build_dataset.py__:  process the fer2013.csv dataset file and output set a set of HDF5 files; one for each of the training, validation, and testing splits, respectively.

3. __train_recognizer.py__: l train our CNN to recognize various emotions.

4. __test_recognizer.py__: evaluate the performance of CNN.

5. __emotion_detector.py__: detect faces in real-time.

## How to run
0. __Step 0__: Folder structure setup

* Create 'fer2013' in the root path of the project, which contains 3 subfolders inside. They are 'fer2013' containing 'fer2013.csv', an empty 'hdf5' folder, and an empty 'output' folder.

* Create 'checkpoints' in the root path of the project.


1. __Step 1__: Build the dataset to train.hdf5, val.hdf5 and test.hdf5

python build_dataset.py

2. __Step 2__: Train the model
python train_recognizer.py --checkpoints checkpoints

3. __Step 3__: Test the model 
python test_recognizer.py --model checkpoints/epoch_15.hdf5

4. __Step 4__: Detect emotion by webcam or video

* __Webcam__: python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model checkpoints/epoch_15.hdf5

* __video__: python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model checkpoints/epoch_15.hdf5 --video quan.mp4