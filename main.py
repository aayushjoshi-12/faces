from utils.video_utils import read_video, save_video
from face_tracker_yolo import FaceTracker
from emotion_detector import EmotionDetector

def main():
    video_path = "input_videos/input.avi"
    frames = read_video(video_path)

    face_tracker = FaceTracker("models/face_detection_model_yolo.pt")
    emotion_detector = EmotionDetector("models/emotion_recognition_model.pth")

    detections = face_tracker.detect_face(frames)
    emotions = emotion_detector.detect_emotion(frames, detections)
    new_frames = emotion_detector.write_emotion(frames, emotions,detections)
    save_video(new_frames, "output_videos/output.avi")

if __name__ == "__main__":
    main()

## new issue that we are facing now is not accurately getting results from the emotion detector.

## it is either because of poor choices during training or that the dataset is skewed and like last time when i used this dataset giving me same horrible results