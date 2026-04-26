

---

# 🧠 Stroke Risk Prediction — ML Web Application

A full-stack Machine Learning web application that predicts **stroke risk** from patient clinical data using a trained model exposed through a Flask API and a React frontend.

> Demonstrates end-to-end ML deployment: **Data → Model → API → Frontend → Cloud**

---

## 🚀 Live Application

* 🌐 Frontend: [https://foresight-health-app.lovable.app](https://foresight-health-app.lovable.app)
* ⚙️ API Health Check: [https://stroke-ml-api.onrender.com/health](https://stroke-ml-api.onrender.com/health)

> ⏳ First request may take **30–60 seconds** due to Render free tier cold start.

---

## 🏗️ System Architecture

```
Dataset → ML Training → Saved Model (.pkl)
        ↓
     Flask API (Render)
        ↓
  React Frontend (Lovable)
```

---

## 🧠 Machine Learning Pipeline

| Step               | Details                                                              |
| ------------------ | -------------------------------------------------------------------- |
| Dataset            | Kaggle Stroke Prediction Dataset (5,110 records)                     |
| Target             | Stroke (0 = No, 1 = Yes)                                             |
| Models Evaluated   | Logistic Regression, Random Forest, XGBoost                          |
| Final Model        | **Random Forest (Best ROC-AUC)**                                     |
| Missing Values     | Median Imputation (BMI)                                              |
| Encoding           | OneHotEncoder (categorical features)                                 |
| Scaling            | StandardScaler (numerical features)                                  |
| Evaluation Metrics | ROC-AUC, Precision, Recall, F1-Score                                 |
| Artifacts Saved    | `stroke_model.pkl`, `scaler.pkl`, `encoder.pkl`, `feature_names.pkl` |

---

## 🛠️ Tech Stack

| Layer       | Technology                          |
| ----------- | ----------------------------------- |
| ML & Data   | Python, pandas, numpy, scikit-learn |
| Backend API | Flask, gunicorn                     |
| Frontend    | React, TypeScript                   |
| Deployment  | Render (API), Lovable (Frontend)    |
| Monitoring  | UptimeRobot (keep-alive ping)       |

---

## 📁 Repository Structure

```
.
├── app.py                 # Flask API
├── requirements.txt       # Python dependencies
├── Procfile               # Render deployment config
├── stroke_model.pkl       # Trained ML model
├── scaler.pkl             # StandardScaler object
├── encoder.pkl            # OneHotEncoder object
└── feature_names.pkl      # Feature ordering for inference
```

---

## 🔌 API Usage

### Endpoint

`POST /predict`

### Sample Request

```json
{
  "age": 67,
  "gender": "Male",
  "hypertension": 1,
  "heart_disease": 1,
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "smoking_status": "formerly smoked"
}
```

### Sample Response

```json
{
  "prediction": 1,
  "probability": 46.0,
  "risk_level": "High"
}
```

---

## 🧪 Testing the API

```bash
curl https://stroke-ml-api.onrender.com/health
```

---

## 💡 Key Highlights

* End-to-end ML deployment 
* Proper preprocessing pipeline saved and reused during inference
* Production-style separation of **model**, **API**, and **frontend**
* Cloud deployment with monitoring
* Real-world healthcare use case for demonstration

---

## ⚠️ Disclaimer

This project is for **educational and demonstration purposes only**.
It is **not** intended for medical diagnosis or clinical use.
No patient data is stored or retained.

---


