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
├── README.md
├── .gitignore
├── requirements.txt
├── data.yaml
├── train.py
├── train_all_models.py
├── detect.py
├── dashboard.py
├── best.pt
├── full_model_comparison.csv
├── different yolo model results.png
├── 1.jpg
├── 2.jpg
└── 3.jpg
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/usamaellahi2022-code/automated-pothole-detection.git
```

### 2. Open the project directory

```bash
cd automated-pothole-detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## 🗂️ Dataset Configuration

The YOLO dataset configuration is provided in `data.yaml`.

The project uses one object-detection class:

```text
pothole
```

The dataset itself is not included in this repository. Configure the dataset paths in `data.yaml` before training.

## 🏋️ Model Training

Train the main YOLO model:

```bash
python train.py
```

Train and compare multiple YOLO models:

```bash
python train_all_models.py
```

## 🔍 Pothole Detection

Run the detection script:

```bash
python detect.py
```

The system detects potholes and produces visual results with bounding boxes and confidence scores.

## 🌐 Streamlit Dashboard

Run the interactive dashboard:

```bash
python -m streamlit run dashboard.py
```

The dashboard supports:

- Model selection
- Confidence threshold adjustment
- Image upload
- Pothole detection
- Detection count
- Annotated result visualization
- Downloadable detected images

## 📊 Model Comparison

The repository contains model comparison results in:

`full_model_comparison.csv`

The comparison includes:

- mAP50
- mAP50-95
- Precision
- Recall
- Epochs trained

### Compared Models

- YOLOv11m
- YOLOv8m
- YOLOv11s
- YOLOv8s
- YOLOv8n
- YOLOv11n
- YOLOv8n (100 epochs)
- YOLOv5s
- YOLOv5n

## 🖼️ Detection Results

Sample detection results are included in:

- `1.jpg`
- `2.jpg`
- `3.jpg`

## 🚀 Future Improvements

- Real-time pothole detection from video
- GPS-based pothole location mapping
- Mobile or web deployment
- Pothole severity classification
- Cloud deployment
- Database integration
- Automated maintenance alerts
- Smart-city infrastructure integration

## 🎓 Academic Project

This project was developed as a Bachelor-level Software Engineering project at:

**Iqra National University, Peshawar**

## 👨‍💻 Author

**Usama Ellahi**

Software Engineering Graduate

### Interests

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Data Science
- Generative AI

### Connect

LinkedIn: https://www.linkedin.com/in/usama-ellahi-64ba27306/

GitHub: https://github.com/usamaellahi2022-code

---

⭐ If you find this project useful, consider starring the repository.
