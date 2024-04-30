import torch
from torchvision import models
import cv2 as cv

class FaceTracker:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def detect_face(self, frames):
        bboxes = []
        for frame in frames:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).to(self.device)
            frame = frame.permute(2, 0, 1)
            frame = frame.unsqueeze(0).float()
            with torch.no_grad():
                bbox = self.model(frame)[0]
            bboxes.append(bbox)
        return bboxes
    
    def draw_bbox(self, frames, bboxes):
        for frame, bbox in zip(frames, bboxes):
            x0, y0, x1, y1 = bbox
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            cv.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return frames