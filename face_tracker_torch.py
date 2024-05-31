import torch
from torchvision import models, transforms
import cv2 as cv

class FaceTracker:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 4)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_face(self, frames):
        bboxes = []
        for frame in frames:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frame = self.transforms(frame).to(self.device)
            frame = frame.unsqueeze(0).float()
            with torch.no_grad():
                bbox = self.model(frame)[0]
            bboxes.append(bbox)
        return bboxes
    
    def draw_bbox(self, frames, bboxes):
        new_frames = []
        for frame, bbox in zip(frames, bboxes): 
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = bbox
            x0, y0, x1, y1 = x0 * w/256, y0 * h/256, x1 * w/256, y1 * h/256
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            cv.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
            new_frames.append(frame)
        return new_frames