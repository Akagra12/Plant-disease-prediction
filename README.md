# 🌿 Plant Disease Classifier

A CNN-based web application that identifies **38 types of plant diseases** across **14 crop species** from leaf images. Built with **Streamlit** and **TensorFlow**.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red?logo=streamlit)

---

## ✨ Features

- 📸 Upload a leaf image and get instant disease prediction
- 🧠 Pre-trained CNN model (auto-downloads on first run)
- 🎨 Beautiful dark-themed UI with glassmorphism effects
- 🌿 Supports 14 plants: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Akagra12/Plant-disease-prediction.git
cd Plant-disease-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run main.py
```

> **Note:** The trained model (~100 MB) is automatically downloaded from Google Drive on the first run. No manual setup needed!

---

## 📂 Project Structure

```
Plant-disease-prediction/
├── .streamlit/
│   └── config.toml          # Streamlit theme & server config
├── notebooks/
│   └── *.ipynb               # Model training notebooks
├── trained_model/
│   └── (auto-downloaded)     # .h5 model file (not in git)
├── .gitignore
├── class_indices.json        # Disease class mapping
├── main.py                   # Streamlit web application
├── requirements.txt          # Python dependencies
├── runtime.txt               # Python version for deployment
└── README.md
```

---

## 🌐 Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/).
3. Click **New app** → select this repo → set main file to `main.py`.
4. Click **Deploy**. Done!

The model auto-downloads from Google Drive when the app boots — no Git LFS needed.

---

## 📊 Dataset

Trained on the [PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) from Kaggle.

---

## 📝 License

This project is open source and available for educational purposes.
