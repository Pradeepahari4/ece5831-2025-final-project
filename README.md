# ece5831-2024-final-project
# Driver Drowsiness Detection System 
This repository contains the implementation and documentation for a **real-time Driver Drowsiness Detection System** developed as part of **ECE 5831: Pattern Recognition & Neural Networks**.  
The system uses **computer vision and deep learning** to detect prolonged eye closure and alert the driver in real time.

## Project Overview

Driver drowsiness is a major contributor to road accidents, especially during long-distance and night-time driving. Traditional physiological methods such as EEG or EOG are intrusive and impractical for real-world use.  
This project proposes a **non-intrusive, vision-based solution** that detects driver drowsiness using eye-state classification and temporal logic.

### Key Features
- Real-time webcam-based monitoring
- Haar Cascade–based face and eye detection
- CNN-based eye-state classification
- Temporal logic to reduce false positives
- Audible alert on prolonged eye closure
- Support for both **custom CNN** and **MobileNetV2**

## Project Structure
- final-project.ipynb # Model training, evaluation, and experiments
- live_demo.py # Real-time webcam drowsiness detection script

## Dataset

**MRL Eye Dataset**
- Total images: **85000(approx)**
- Binary classes:
  - Open eyes
  - Closed eyes
- Includes variations in:
  - Lighting conditions
  - Head pose
  - Glasses
- Used for training and validation of both models

## Jupyter Notebook (`final-project.ipynb`) – Code Flow

The notebook contains the **model development and evaluation pipeline**.

### 1. Data Loading & Preprocessing
- Load MRL Eye Dataset
- Resize eye images to fixed dimensions
- Normalize pixel values to `[0,1]`
- Prepare labels for binary classification

### 2. Custom CNN Model
- Built and trained from scratch
- Architecture includes:
  - Convolution layers with ReLU activation
  - Max-pooling layers
  - Fully connected layers
  - Softmax output layer
- Optimizer: Adam
- Loss function: Categorical cross-entropy

### 3. MobileNetV2 Model
- Pre-trained on ImageNet
- Transfer learning approach:
  - Freeze base layers
  - Retrain classification layers
- Lightweight and optimized for real-time inference

### 4. Model Evaluation
- Validation accuracy
- Confusion matrix
- Precision, recall, and F1-score
- Comparison between CNN and MobileNetV2

##  Real-Time Detection Script (`live_demo.py`) – Code Flow

This script integrates the trained model with a live webcam feed.

### 1. Model Loading
- Loads a trained `.h5` Keras model
- Automatically infers:
  - Input size
  - Number of channels
  - Grayscale or RGB input

### 2. Face & Eye Detection
- Uses OpenCV Haar Cascade classifiers:
  - `haarcascade_frontalface_default.xml`
  - `haarcascade_eye.xml`

### 3. Eye Preprocessing
- Crop detected eye regions
- Resize to model input shape
- Normalize pixel values
- Convert to grayscale or RGB based on model

### 4. Eye-State Prediction
- Model outputs probability of **eye being open**
- Probabilities averaged across detected eyes
- Threshold-based classification

### 5. Temporal Logic
- Tracks consecutive closed-eye frames
- If closed for **15 consecutive frames**, classify as drowsy
- Reduces false positives due to blinking

### 6. Alert System
- Non-blocking audio alarm triggered on drowsiness
- Visual feedback:
  - Bounding boxes
  - Eye probabilities
  - Driver state (AWAKE / DROWSY)
  - FPS display
  - 
## Running the Real-Time Demo

### Install Dependencies
```bash
pip install tensorflow opencv-python numpy simpleaudio tqdm

## Links to youtube demo video, pre-recorded presentation, google drive
youtube - https://www.youtube.com/watch?v=Y35h7drmuGY
pre-recorded presentation - https://drive.google.com/file/d/1WRvLIiarNU7grCFw4rYefb6TwxcnmXyG/view
google drive - https://drive.google.com/drive/u/1/folders/1W7izULOJvaqRENbOU9sPPKe3HjoR28P3
