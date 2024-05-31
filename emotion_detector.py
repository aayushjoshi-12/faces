import torch
import cv2 as cv
import torch.nn.functional as F
from torchvision import models, transforms

N_LABELS = 7 
LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

class EmotionDetector:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, N_LABELS)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def detect_emotion(self, frames, detections):
        emotions = []
        for frame_num, frame in enumerate(frames):
            emotions_dict = {}
            bbox_dict = detections[frame_num]
            for id, bbox in bbox_dict.items():
                x0, y0, x1, y1 = map(int, bbox)
                face = frame[y0:y1, x0:x1]
                face = cv.cvtColor(face, cv.COLOR_BGR2RGB)
                face = self.transforms(face).to(self.device)
                face = face.unsqueeze(0).float()
                with torch.no_grad():
                    output = F.softmax(self.model(face), dim=1)
                    idx = torch.argmax(output).item()
                    emotions_dict[id] = LABELS[idx]
            emotions.append(emotions_dict)
        return emotions
    
    def write_emotion(self, frames, emotions, detections):
        new_frames = []
        for frame_num, frame in enumerate(frames):
            bbox_dict = detections[frame_num]
            emotions_dict = emotions[frame_num]
            for id, bbox in bbox_dict.items():
                x0, y0, x1, y1 = map(int, bbox)
                emotion = emotions_dict[id]
                cv.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv.putText(frame, f"Face: {id} - {emotion}", (x0, y0-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            new_frames.append(frame)
        return new_frames