import cv2 as cv
from ultralytics import YOLO

class FaceTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def _detect_frame(self, frame):
        results = self.model.track(frame, persist=True, save=False)[0]
        bbox_dict = {}
        for box in results.boxes:
            id = int(box.id.tolist()[0])
            bbox = box.xyxy[0].tolist()
            bbox_dict[id] = bbox 
        return bbox_dict

    def detect_face(self, frames):
        detections = []
        for frame in frames:
            bbox_dict = self._detect_frame(frame)
            detections.append(bbox_dict)
        return detections
    
    def draw_bbox(self, frames, detections):
        new_frames = []
        for frame_num, frame in enumerate(frames):
            bbox_dict = detections[frame_num]
            for id, bbox in bbox_dict.items():
                x0, y0, x1, y1 = map(int, bbox)
                cv.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv.putText(frame, f"Face: {id}", (x0, y0-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            new_frames.append(frame)
        return new_frames