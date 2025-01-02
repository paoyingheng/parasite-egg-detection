---
title: Detection and Classification of Parasite Eggs in Microscopic Images with YOLOv8 
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
license: cc
---
[Demo](https://huggingface.co/spaces/paoyingheng/parasite-egg-detection-classification)


# Parasite Egg Detection and Classification Using YOLOv8

## Project overview
This project uses YOLOv8, an object detection model, to automate the detection and classification of parasite eggs in microscopy images. The workflow includes data preparation, model training, validation, testing, and deployment to Hugging Face Spaces for real-time inference.

## Objectives
1. Automate the detection and classification of parasite eggs to improve diagnostic efficiency.
2. Train a custom YOLOv8 model on microscopy images labeled with specific parasite classes.
3. Deploy the trained model to Hugging Face Spaces for an interactive user experience.

## Features
- Supports detection and classification of multiple parasite egg species.
- Interactive interface for testing via Hugging Face Spaces.
- Scalable and reproducible workflow for training and evaluation.


## Installation and usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/parasite-egg-detection.git
cd parasite-egg-detection
```

### 2. Set up virtual environment
```bash
virtualenv -p python3 venv
source ./venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the model
Ensure your dataset is correctly formatted in YOLO format and update config.yaml with your dataset paths.

```bash
yolo detect train data=config.yaml model=yolov8n.yaml epochs=100 imgsz=640
```

Note: The trained model weights (best.pt) will be saved in runs/detect/train/weights/.

### 4. Test the model
Run predictions using the trained model:

```bash
yolo predict model=runs/detect/train/weights/best.pt source=test/images/testing.jpg save
```
### 5. Deploy to Hugging Face
Push the following files to Hugging Face Spaces:

app.py
best.pt
requirements.txt
README.md
```bash
git add app.py best.pt requirements.txt README.md
git commit -m "Deploy YOLOv8 model to Hugging Face"
git push
```

## Demo
Test the deployed model [here](https://huggingface.co/spaces/paoyingheng/parasite-egg-detection-classification) 

