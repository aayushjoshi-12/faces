import os
from utils.dataset_utils import get_face_detection_dataset, get_emotion_detection_dataset
from utils.train_utils import train_face_detection_model, train_emotion_detection_model

# replace the string with your roboflow api key
ROBOFLOW_API_KEY = "$API_KEY$"

os.makedirs("models", exist_ok=True)

print("[INFO] Getting Face Detection Dataset\n")
get_face_detection_dataset(ROBOFLOW_API_KEY)
print("[INFO] Training Face Detection Model\n")
train_face_detection_model()

print("[INFO] Getting Emotion Detection Dataset\n")
get_emotion_detection_dataset()
print("[INFO] Training Emotion Detection Model\n")
train_emotion_detection_model()