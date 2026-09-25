# Road Traffic Object Detection with YOLOv3 & ImageAI

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![ImageAI](https://img.shields.io/badge/ImageAI-3.0%2B-orange.svg)](https://github.com/OlayinkaPeter/ImageAI)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

A Computer Vision pipeline utilizing **YOLOv3** object detection models to identify, classify, and filter road participants (vehicles, pedestrians, traffic signals) for traffic safety analysis.

---

## 🚀 Key Features

* **YOLOv3 Deep Learning Integration**: Employs deep convolutional networks for fast multi-object bounding box prediction.
* **Traffic Class Filtering**: Automated filtering isolate specific transportation entities (`car`, `bus`, `pedestrian`, `traffic light`, `stop sign`).
* **Confidence Thresholding**: Dynamic minimum probability detection constraints to reduce false positives.
* **Structured Spatial Extraction**: Extracts exact coordinate bounding box metrics (`box_points`) for analytics.

---

## 🛠️ Tech Stack

* **Language**: Python 3.10+
* **Computer Vision**: ImageAI, OpenCV
* **Deep Learning Backend**: PyTorch / Torchvision

---

## ⚙️ Configuration & Setup

### 1. Clone the repository
git clone https://github.com/DrRafael/road-object-detection-yolo.git
cd road-object-detection-yolo

### 2. Install dependencies
pip install -r requirements.txt

### 3. Download Model Weights
Download the pre-trained `yolov3.pt` weights file and place it in the root directory.

### 4. Run detection
python main.py

---

**Author**: QA Automation Engineer & Python Developer
