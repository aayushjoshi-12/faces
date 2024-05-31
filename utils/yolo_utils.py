from ultralytics import YOLO

model = YOLO('models/face_detection_model_yolo.pt')

result = model.predict('input_videos/input3.mp4', save=True, conf=0.1)
# result = model.track('input_videos/input.avi', save=True, conf=0.2)