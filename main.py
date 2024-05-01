from utils import read_video, save_video
from face_tracker import FaceTracker

def main():
    video_path = "input_videos/input.avi"
    frames = read_video(video_path)
    tracker = FaceTracker("models/face_detection_model.pth")
    bboxes = tracker.detect_face(frames)
    new_frames = tracker.draw_bbox(frames, bboxes)
    save_video(new_frames, "output_videos/output.avi")

if __name__ == "__main__":
    main()