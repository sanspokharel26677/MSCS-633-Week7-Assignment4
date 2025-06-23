# Credit Card Fraud Detection using PyOD AutoEncoder

This project uses an **unsupervised deep learning model** from the PyOD library to detect anomalies (fraudulent transactions) in a credit card dataset.

---

## 📁 Dataset

- Source: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/whenamancodes/fraud-detection)
- Records: 284,807 transactions
- Fraud cases: 492
- Class distribution is highly imbalanced.
-⚠️ The dataset file (creditcard.csv) exceeds GitHub’s 100 MB limit and is not uploaded.
Please download it directly from Kaggle Fraud Detection Dataset.

---

## ⚙️ Setup & Installation

You must use **Python 3.8+** (we used Python 3.13 on macOS).

### 🔧 1. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 📦 2. Install Required Libraries

```bash
pip install pandas numpy scikit-learn pyod matplotlib seaborn tqdm torch
```

> Note: `torch` is required because PyOD's AutoEncoder depends on it. `tqdm` is used for training progress display.

---

## 🚀 How to Run

```bash
python3 fraud_detection.py
```

This script performs the following:
- Loads and preprocesses the dataset
- Normalizes 'Time' and 'Amount'
- Splits into training/testing sets
- Trains an AutoEncoder for anomaly detection
- Outputs classification metrics and AUC

---

## 📊 Evaluation Metrics

- Confusion Matrix
- Classification Report (Precision, Recall, F1)
- ROC-AUC Score

---

## ✅ Output Sample

```
ROC-AUC Score: 0.9498
```

---

## 💡 Notes

- AutoEncoders are sensitive to imbalance; consider further threshold tuning or ensemble models for production.
- The project was executed using terminal-based CLI, Python, and virtual environments.

---

## 👨‍💻 Author

Sandesh Pokharel

