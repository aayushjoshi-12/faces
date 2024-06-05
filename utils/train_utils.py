import shutil as sh
import torch
from ultralytics import YOLO
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_LABELS = 7
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
STEP_SIZE = 5
TRANSFORMS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229])
])

def train_face_detection_model():
    model = YOLO('yolov8n')
    model.train(data='/Face-Detection-24/data.yaml', epochs=20, batch_size=20, imgsz=640, pretrained=True, verbose=True)
    sh.move('runs/detect/train/weights/best.pt', 'models/face_detection_model_yolo.pt')

def train_emotion_detection_model():
    train_data = ImageFolder('./train', transform=TRANSFORMS)
    val_data = ImageFolder('./test', transform=TRANSFORMS)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    model = models.resnet18(pretrained=True)
    model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = torch.nn.Linear(model.fc.in_features, N_LABELS)
    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=0.1)

    history = {
        'max_val_accuracy' : 0,
        'train_accuracy' : [],
        'val_accuracy' : [],
        'train_loss' : [],
        'val_loss' : [],
    }

    for epoch in range(EPOCHS):
        model.train()

        total_train_loss = 0
        train_correct = 0

        total_val_loss = 0
        val_correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_correct += (torch.argmax(pred, 1) == y).sum().item()

        with torch.no_grad():
            model.eval()
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                loss = loss_fn(pred, y)

                total_val_loss += loss.item()
                val_correct += (torch.argmax(pred, 1) == y).sum().item()

        scheduler.step()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        train_accuracy = train_correct / len(train_loader.dataset)
        val_accuracy = val_correct / len(val_loader.dataset)

        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if history['max_val_accuracy'] < val_accuracy:
            history['max_val_accuracy'] = val_accuracy
            torch.save(model.state_dict(), 'models/emotion_recognition_model_resnet18.pth')

        print(f'[INFO] EPOCH: {epoch + 1}/{EPOCHS}')
        print(f'[INFO] Train Accuracy: {train_accuracy:.4f}, Train Loss: {avg_train_loss:.4f}')
        print(f'[INFO] Val Accuracy: {val_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}\n')