import cv2 
import os 
import numpy as np 
from keras.applications.vgg16 import VGG16
from keras.models import Model
import shutil
import subprocess
import glob
from tqdm import tqdm
import fire
from elapsedtimer import ElapsedTimer

class PreProcessing:
    def __init__(self, video_dest, feat_dir, temp_dest,
                img_dim=224, channels=3, batch_size=128, frames_step=80):

        self.img_dim = img_dim
        self.channels = channels
        self.video_dest = video_dest
        self.feat_dir = feat_dir
        self.temp_dest = temp_dest
        self.batch_size = batch_size
        self.frames_step = frames_step
    
        print("Video dest: {}".format(self.video_dest))
        print("Feature dir: {}".format(self.feat_dir))
        print("Temp dest: {}".format(self.temp_dest))
    
    # convert the input video into image feames at a specified sampling rate
    def video_to_frames(self,video):
        with open(os.devnull, "w") as ffmpeg_log:
            if os.path.exists(self.temp_dest):
                print(" cleanup: " + self.temp_dest + "/")
                shutil.rmtree(self.temp_dest)

            os.makedirs(self.temp_dest)
            video_to_frames_cmd = ["ffmpeg",'-y','-i', video, 
                                       '-vf', "scale=400:300", 
                                       '-qscale:v', "2", 
                                       '{0}/%06d.jpg'.format(self.temp_dest)]

            subprocess.call(video_to_frames_cmd,
                            stdout=ffmpeg_log, stderr=ffmpeg_log, shell=True)

    
    # load the pre-trained VGG16 model and extract the dense features 
    def model_cnn_load(self):
        model = VGG16(weights="imagenet", include_top=True, input_shape=(self.img_dim, self.img_dim, self.channels))

        out = model.layers[-2].output
        model_final = Model(inputs=model.input, outputs=out)

        return model_final


    #　load the video images
    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.img_dim, self.img_dim))

        return img


    # extract the features from the pre-trained CNN
    def extract_feats_pretrained_cnn(self):
        model = self.model_cnn_load()
        print("Model has been loaded successfully!")

        # make feature dir if it does not exist
        if not os.path.isdir(self.feat_dir):
            os.mkdir(self.feat_dir)

        video_list = glob.glob(os.path.join(self.video_dest, '*.avi'))

        for video in tqdm(video_list):
            video_id = video.split("\\")[-1].split(".")[0]
 

            self.video_to_frames(video)

            img_list = sorted(glob.glob(os.path.join(self.temp_dest, '*.jpg')))
            samples = np.round(np.linspace(0, len(img_list)-1, self.frames_step))

            img_list = [img_list[int(sample)] for sample in samples]

            imgs = np.zeros((len(img_list), self.img_dim, self.img_dim, self.channels))

            for i in range(len(img_list)):
                img = self.load_image(img_list[i])
                imgs[i] = img

            
            imgs = np.array(imgs)
            fc_feats = model.predict(imgs, batch_size=self.batch_size)
            img_feats = np.array(fc_feats)

            outfile = os.path.join(self.feat_dir, video_id + '.npy')
            np.save(outfile, img_feats)

            # do clean up
            shutil.rmtree(self.temp_dest)

        
    def process_main(self):
        self.extract_feats_pretrained_cnn()


if __name__ == "__main__":
    with ElapsedTimer('PreProcessing'):
        fire.Fire(PreProcessing)
