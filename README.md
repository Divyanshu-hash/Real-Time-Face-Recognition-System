# 🧠 Real-Time Face Recognition System

This project is a Python-based face recognition system that detects faces from images, encodes them, and stores the embeddings in a `pickle` file (`embeddings.pkl`). It can dynamically update with new faces and supports real-time identification.

---

## 📂 Project Structure

```bash
├── dataset/
│ ├── Person1/
│ │ ├── img1.jpg
│ │ └── img2.jpg
│ ├── Person2/
│ │ └── img1.jpg
├──  collectingDatasets.py
├── embeddings.pkl
├── generate_embeddings.py
├── face_recognition_live.py
├── requirements.txt
└── README.md
```

---

## 🔧 Features

- Generate face encodings using [`face_recognition`](https://github.com/ageitgey/face_recognition)
- Automatically update `embeddings.pkl` if it already exists
- Warns if no face is found in an image
- Real-time face recognition support (optional)
- Error-handling and logging for easy debugging

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/face-recognition-system.git
cd face-recognition-system
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

## ▶️ Usage
To generate/update face embeddings:
```bash
python generate_embeddings.py
```
If embeddings.pkl already exists, new embeddings will be added without duplicating existing ones.

To recognize faces (optional):
```bash
python recognize_faces.py
```

## 🗂️ Dataset Format
Your dataset/ folder should be structured like:
```bash
dataset/
├── Alice/
│   ├── alice_1.jpg
│   └── alice_2.jpg
├── Bob/
│   ├── bob_1.jpg
```
Each subfolder represents a different person. Images should have clear frontal faces.

### 📦 Requirements
Python 3.8+

face_recognition

numpy

Pillow

dlib (C++ build tools required)

📌 Note: Installing dlib requires Microsoft C++ Build Tools on Windows.

### 📄 License
This project is licensed under the MIT License.

### 🙋‍♂️ Author
Divyansu Giri
---
