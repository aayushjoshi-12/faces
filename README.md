# Faces
This project focues on detecting emotions from videos using ResNet architecure and YOLO architecture. I took a multimodal approach towards the project where I detect faces using a YOLO architecture model and detect emotions using ResNet architecture. The model is able to classify 7 different emotion categories. 


<b>Emotion Categories:</b> ```['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']```<br>
<b>Emotion Detection Model Accuracy:</b> ```65.97%```<br>


## Features:
- Face detection using a YOLO model.
- Emotion detection using a fine-tuned ResNet18 model on FER-2013 dataset.
- Pre-trained model loading for quick inference.
- <B>Dependencies</B>:
   - PyTorch
   - Ultralytics
   - OpenCV

## Installation:
### Prerequisites:
Ensure you have Python 3.10 and all the necessary packages are installed:
```
sudo apt update
sudo apt install python3
```
### Setting up the Virtual Environment:
1. Clone the repository:
```git clone https://github.com/aayushjoshi-12/faces.git```
2. Create and activate a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
```
3. Install the required python packages:
```pip install -r requirements.txt```

### Usage:
#### For training models locally 
Skip these steps if you want to directly use the model trained by me.
1. Set up your roboflow account and create an api key. Put this api key in train.py file at the mentioned place.
2. Set up your kaggle account and download the api key file kaggle.json. Create a ```./kaggle``` directory in the root directory and save the file there.
3. Save the changes to ```train.py``` script and run it to train all the required models.
```python train.py```

Follow the instructions mentioned below after these steps.

#### For using the pretrained and fine-tuned model:
1. Add the input videos to a director called ```./input_videos```
2. Add the proper path of the input video in the ```main.py``` file and save it.
3. Run the ```main.py``` file.
```
python main.py
```
4. The results are saved at ```./output_videos/output.avi```

## Additional Information:
- All the notebooks were run on google colab and are kept only as log files, so please do not refer them for training purposes and use the training script provided for recreating the results. 
- The scripts might not run properly for windows user since it was developed on linux.

## Issues:
- The ResNet architecture faced overfitting issues and did not improve despite using various regularization techniques such as learning rate schedulers, dropout, L2 regularization, and data augmentation leading to such low accuracy of ```65.97%```.
- The LeNet architecture faced similar issues.
- The models struggled to achieve a validation accuracy above 0.52 - 0.54.
- I believe it is due to the lack of features variation in the FER-2013 dataset. All the images being grayscale and 48x48 pixels (creating a lack of features) lead to overfitting on a largescale complex model.

## Learning Outcome:
- This project helped me understand the importance of validation and the quirks of using different architectures such as YOLO, ResNet, and LeNet.

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.