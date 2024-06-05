import cv2 as cv
from utils.video_utils import read_video, save_video
from face_tracker_yolo import FaceTracker
from emotion_detector import EmotionDetector

emotion_detector = EmotionDetector("models/emotion_detection_model_resnet18.pth")
face_tracker = FaceTracker("models/face_detection_model_yolo.pt")

def detect_in_video(video_path, output_path):
    frames = read_video(video_path)
    detections = face_tracker.detect_frames(frames)
    emotions = emotion_detector.detect_frames(frames, detections)
    new_frames = emotion_detector.write_frames(frames, emotions, detections)
    save_video(new_frames, output_path)

def detect_using_webcam():
    # this function doesnt works on my system because i didn't built opencv so it doesn't have support for GUI
    # in future i might come back and work on real time detection as well
    webcam = cv.VideoCapture(0)
    while webcam.isOpened():
        ret, frame = webcam.read()
        if not ret:
            break
        bbox_dict = face_tracker.detect_frame(frame)
        emotions_dict = emotion_detector.detect_frame(frame, bbox_dict)
        new_frame = emotion_detector.write_frame(frame, emotions_dict, bbox_dict)
        cv.imshow("frame", new_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    detect_in_video('input_videos/input1.avi', 'output_videos/output.avi')