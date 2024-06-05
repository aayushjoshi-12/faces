import shutil as sh
import os
from roboflow import Roboflow

def get_face_detection_dataset(your_api_key):
    rf = Roboflow(api_key=your_api_key)
    project = rf.workspace("mohamed-traore-2ekkp").project("face-detection-mik1i")
    version = project.version(24)
    dataset = version.download("yolov8")
    sh.move('/content/Face-Detection-24/test', '/content/Face-Detection-24/Face-Detection-24/valid')
    sh.move('/content/Face-Detection-24/train', '/content/Face-Detection-24/Face-Detection-24/train')

def get_emotion_detection_dataset():
    os.system("kaggle datasets download -d msambare/fer2013")
    os.system("unzip fer2013.zip")