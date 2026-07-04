# 🎬 StreamPulse — Video Streaming Engagement Prediction

> An end-to-end Machine Learning project that predicts user engagement levels (Low / Medium / High) for a video streaming platform using synthetic data, ML modeling, a REST API, and an interactive dashboard.

---

## 🔗 Live Links

| | URL |
|--|--|
| 🎨 Dashboard | https://streampulse-seven.vercel.app |
| 🔧 API | https://streampulse-cvmg.onrender.com |

---

## 📁 Project Structure

```
streampulse/
│
├── api/
│   ├── app.py                  # Flask REST API
│   ├── best_model.joblib       # Trained ML model
│   ├── requirements.txt        # Python dependencies
│   └── Procfile                # Render deployment config
│
├── dashboard/
│   ├── dashboard.html          # Interactive frontend
│   └── vercel.json             # Vercel deployment config
│
├── milestone1/
│   └── generate_data.py        # Synthetic data generation
│
├── milestone2/
│   └── eda_feature_engineering.py  # EDA & feature engineering
│
├── milestone3/
│   └── model_training.py       # ML modeling & evaluation
│
└── milestone4/
    └── test_api.py             # API unit tests
```

---

## 🧠 Project Overview

### Domain
Video Streaming (Entertainment)

### Task
Multi-class Classification — Predict user engagement level:
- 🔴 **Low** — At-risk users, likely to churn
- 🟡 **Medium** — Moderately engaged users
- 🟢 **High** — Highly engaged users

### Features Used
| Feature | Description |
|---------|-------------|
| `age` | User age |
| `weekly_watch_hours` | Hours watched per week |
| `skip_rate` | Ratio of content skipped (0–1) |
| `avg_rating_given` | Average content rating (1–5) |
| `logins_per_week` | App opens per week |
| `subscription_type` | Free / Basic / Premium |
| `content_diversity_score` | Variety of genres explored |
| `days_since_last_login` | Recency of activity |
| `notifications_clicked` | Responsiveness to alerts |
| `watch_streak_days` | Consecutive days watched |

---

## 🚀 Milestones

### Milestone 1 — Synthetic Data Engineering
- Generated 5000 synthetic users using AI-assisted scripting
- Injected real-world flaws: 50/30/20 class imbalance, 12% missing values, correlated anomalies

### Milestone 2 — EDA & Feature Engineering
- Handled missing values (median imputation)
- Capped outliers using IQR method
- Encoded categorical variables
- Engineered 5 new features: `engagement_ratio`, `activity_score`, `premium_watch`, `inactive_flag`, `notification_rate`
- Balanced classes using oversampling
- Scaled features using StandardScaler

### Milestone 3 — ML Modeling
| Model | F1 Score |
|-------|----------|
| Logistic Regression (Baseline) | 0.8944 |
| Random Forest (Tuned) ✅ | 0.9249 |

- Hyperparameter tuning via GridSearchCV
- 5-Fold Cross Validation: **0.9376 ± 0.0058**
- Best params: `max_depth=20`, `n_estimators=200`

### Milestone 4 — Backend API
- Built with **Flask**
- Endpoints: `/predict`, `/health`, `/model-info`
- Input validation with range checks
- **8 unit tests** — all passing ✅

### Milestone 5 — Frontend Dashboard
- Built with HTML, CSS, Chart.js
- Live charts: watch trends, engagement distribution, skip rate scatter, subscription breakdown
- Real-time notification feed
- Predict form connected to live API
- Deployed on **Vercel**

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data | Python, Pandas, NumPy |
| ML | Scikit-learn, Joblib |
| API | Flask, Flask-CORS, Gunicorn |
| Frontend | HTML, CSS, Chart.js |
| Deployment | Render (API), Vercel (Dashboard) |
| Version Control | Git, GitHub |

---

## ⚙️ Run Locally

### API
```bash
cd api
pip install -r requirements.txt
python app.py
```
API runs at `http://localhost:5000`

### Test the API
```bash
cd milestone4
python test_api.py
```

### Sample API Request
```bash
curl -X POST https://streampulse-cvmg.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 28,
    "weekly_watch_hours": 15.5,
    "skip_rate": 0.2,
    "avg_rating_given": 4.1,
    "logins_per_week": 5,
    "subscription_type": "Premium",
    "content_diversity_score": 0.75,
    "days_since_last_login": 2,
    "notifications_clicked": 4,
    "watch_streak_days": 12
  }'
```

### Sample Response
```json
{
  "prediction": "High",
  "confidence_pct": 76.5,
  "probabilities": {
    "Low": 8.2,
    "Medium": 15.3,
    "High": 76.5
  }
}
```

---

## 👨‍💻 Author
**Vamshi** — [@Vamshirk-18](https://github.com/Vamshirk-18)


