# HandSignRecognition ✋

This is a **Hand Sign Recognition** project that detects hand gestures in real time using **OpenCV** and **MediaPipe**.  
It supports special gestures for **SPACE** and **BACKSPACE**, making it possible to type or control actions through hand movements.

# ✋ Hand Sign Recognition using Deep Learning (CNN + OpenCV)


## 🎯 Objective
To build a system that:
- Detects the hand region from live camera feed  
- Classifies gestures using a trained CNN model  
- Runs in real-time with high accuracy  


## 📦 Features
- Custom dataset for hand gestures  
- Deep learning model built from scratch using Keras  
- OpenCV-based frame preprocessing  
- Real-time prediction using webcam  
- Optimized for low-latency inference  


## 🧠 Model Architecture
- Convolution + MaxPooling layers  
- Batch normalization  
- Dropout for regularization  
- Dense layers for classification  
- Softmax output layer  

Training techniques used:
- Data augmentation  
- Learning rate scheduling  
- Early stopping  


## 🧰 Technologies Used
- Python  
- OpenCV  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  


## 🛠 System Workflow

Live Webcam Feed
       ↓
Hand ROI Detection (OpenCV)
       ↓
Image Preprocessing (Resize, Normalize)
       ↓
CNN Model Inference
      ↓
Predicted Gesture






