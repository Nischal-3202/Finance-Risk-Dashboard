# 💳 Financial Risk Intelligence Dashboard

A modular, end-to-end financial risk assessment system built from scratch using:

- Logistic Regression (Probability of Default)
- Linear Regression (Expected Loss)
- Policy Engine (Risk & Decision Logic)
- Desktop GUI (Tkinter)

This project simulates a simplified banking risk evaluation workflow.

---

## 🚀 Features

### 1️⃣ Probability of Default (PD)
- Implemented using Logistic Regression
- L2 Regularization
- Custom gradient descent
- Optimized threshold (0.35)

### 2️⃣ Credit Score Generator
- PD mapped to 300–850 score range
- Risk bands classification

### 3️⃣ Expected Loss Model
- Linear Regression with L2 Regularization
- Predicts outstanding exposure
- R² ≈ 0.994 on test data

### 4️⃣ Policy Engine
- Risk categorization (Low / Medium / High)
- Loan approval decision rules
- Premium calculation using Expected Loss logic

### 5️⃣ Modern GUI Dashboard
- User-friendly financial inputs
- Real-time evaluation
- Color-coded decision output
- Clean fintech-style interface

---

## 🏗 System Architecture

```
User Input (GUI)
        ↓
Inference Engine
        ↓
Logistic Model → Probability of Default
Linear Model   → Expected Loss
        ↓
Policy Engine
        ↓
Final Decision + Premium + Credit Score
```

Architecture is fully modular:

- `src/logistic_model.py`
- `src/linear_model.py`
- `src/policy_engine.py`
- `src/inference_engine.py`
- `gui/app.py`

---

## 📊 Models

### Logistic Regression
- Features:
  - Credit Limit
  - Age
  - Payment Delays
  - Bill Amount
  - Payment Amount
- ROC-AUC ≈ 0.70
- Deployment threshold: 0.35

### Linear Regression
- Predicts: Outstanding Loss
- R² ≈ 0.9946
- RMSE ≈ 5,273

---

## 🖥 How to Run

### 1️⃣ Install Python 3.12 (recommended)

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install numpy pandas scikit-learn
```

### 4️⃣ Run Application

```bash
python gui/app.py
```

---

## 📁 Project Structure

```
finance_risk_dashboard/
│
├── data/
├── models/
├── src/
│   ├── logistic_model.py
│   ├── linear_model.py
│   ├── policy_engine.py
│   ├── inference_engine.py
│   └── export_models.py
│
├── gui/
│   └── app.py
│
└── README.md
```

## 🎯 Learning Objectives

This project demonstrates:
	•	Building ML models from scratch
	•	Feature selection strategy
	•	Regularization techniques
	•	Model evaluation (AUC, RMSE, R²)
	•	Clean system architecture
	•	Separation of ML and business logic
	•	Desktop application deployment

⸻
## 🔮 Future Improvements
	•	Add interactive charts (PD gauge, risk meter)
	•	Export PDF risk reports
	•	Add database persistence
	•	Convert to web app (FastAPI/Flask)
	•	Add advanced feature engineering

⸻

## 📌 Dataset

Based on UCI Credit Card Default Dataset (Taiwan).
