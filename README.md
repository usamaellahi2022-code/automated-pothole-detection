# 🚧 Automated Pothole Detection System

An AI-based computer vision system for detecting potholes in road images using YOLO object detection. The project is designed to support smart road maintenance by automatically identifying potholes and providing visual detection results.

## 📌 Project Overview

Road potholes can negatively affect vehicle safety, driving comfort, and road infrastructure. Manual inspection of roads can be time-consuming and difficult to scale.

This project presents an automated pothole detection system using YOLO-based object detection and computer vision.

The system can:

- Detect potholes in uploaded road images
- Display detected potholes with bounding boxes and confidence scores
- Compare multiple YOLO model variants
- Provide an interactive Streamlit dashboard
- Present model performance results

## ✨ Features

- Automated pothole detection
- YOLO-based object detection
- Computer vision
- Multiple YOLO model comparison
- Confidence threshold control
- Bounding box visualization
- Detection count
- Downloadable annotated images
- Streamlit dashboard
- Model performance analysis

## 🛠️ Technology Stack

### Programming Language
- Python

### AI & Machine Learning
- YOLO
- Computer Vision
- Object Detection

### Python Libraries
- Ultralytics
- OpenCV
- NumPy
- Pandas
- Streamlit

### Development Tools
- Git
- GitHub
- Google Colab / Local Python Environment

## 📁 Project Structure

```text
automated-pothole-detection/
│
├── README.md
├── .gitignore
├── requirements.txt
├── data.yaml
│
├── train.py
├── train_all_models.py
├── detect.py
├── dashboard.py
│
├── best.pt
├── full_model_comparison.csv
└── different yolo model results.png
