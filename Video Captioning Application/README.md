# Video Captioning Application
Build a Video Captioning System leveraging the sequence-to-sequence—video to text architecture

## Datasets
[Data link for Videos] (http://www.cs.utexas.edu/users/ml/clamp/videoDescription/YouTubeClips.tar)

[Data link for Captions] (https://github.com/jazzsaxmafia/video_to_sequence/files/387979/video_corpus.csv.zip)

## System requirement
Verified on Windows 10 GTX 1060 6GB and 16GB RAM

## How to run
1. Step 1: 

Extract YouTubeClips.tar and video_corpus.csv.zip to 'data' folder (in the root directory) to create 'YouTubeClips' folder and file 'video_corpus.csv'.

2. Step 2:

python PreProcessing.py process_main --video_dest 'data/YouTubeClips' --feat_dir features --temp_dest temp --img_dim 224 --channels 3 --batch_size 128 --frames_step 80

3. Step 3:

python VideoSeq2Seq.py process_main --project_path . --caption_file 'data/video_corpus.csv' --feat_dir features --cnn_feat_dim 4096 --h_dim 512 --batch_size 32 --lstm_steps 80 --video_lstm_step 80 --caption_lstm_step 20 --learning_rate 1e-4 --epochs 100
