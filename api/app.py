"""
Milestone 4 - Backend REST API
Domain: Entertainment (Video Streaming)
Framework: Flask (lightweight, production-ready)

AI Tool Prompt Used (Claude Code / Aider):
"Generate a lightweight, production-ready REST API using Flask.
Ensure it validates incoming JSON using strong data schemas.
The API must load the saved ML model and return predictions
for new user data payloads."

Endpoints:
  GET  /health          → Check if API is running
  POST /predict         → Send user data, get engagement prediction
  GET  /model-info      → Info about the loaded model
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# ── Load the saved ML model once at startup ──────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.joblib')
model = joblib.load(MODEL_PATH)

LABEL_MAP     = {0: 'Low', 1: 'Medium', 2: 'High'}
REQUIRED_KEYS = [
    'age', 'weekly_watch_hours', 'skip_rate', 'avg_rating_given',
    'logins_per_week', 'subscription_type', 'content_diversity_score',
    'days_since_last_login', 'notifications_clicked', 'watch_streak_days'
]
SUBSCRIPTION_MAP = {'Free': 0, 'Basic': 1, 'Premium': 2}

# ── Validation Rules ──────────────────────────────────────────
VALIDATION_RULES = {
    'age':                     (16,   65),
    'weekly_watch_hours':      (0,   150),
    'skip_rate':               (0,     1),
    'avg_rating_given':        (1,     5),
    'logins_per_week':         (0,    50),
    'content_diversity_score': (0,     1),
    'days_since_last_login':   (0,    60),
    'notifications_clicked':   (0,    50),
    'watch_streak_days':       (0,    30),
}


def validate_input(data):
    """Validate incoming JSON payload. Returns (errors list)."""
    errors = []

    # Check all required keys are present
    for key in REQUIRED_KEYS:
        if key not in data:
            errors.append(f"Missing required field: '{key}'")

    if errors:
        return errors

    # subscription_type must be valid string
    if data['subscription_type'] not in SUBSCRIPTION_MAP:
        errors.append("'subscription_type' must be one of: Free, Basic, Premium")

    # Numeric range checks
    for field, (min_val, max_val) in VALIDATION_RULES.items():
        val = data.get(field)
        if val is None:
            continue
        if not isinstance(val, (int, float)):
            errors.append(f"'{field}' must be a number")
        elif not (min_val <= val <= max_val):
            errors.append(f"'{field}' must be between {min_val} and {max_val}, got {val}")

    return errors


def build_feature_vector(data):
    """
    Transform raw input into the same feature vector
    the model was trained on (including engineered features).
    """
    age                     = data['age']
    weekly_watch_hours      = data['weekly_watch_hours']
    skip_rate               = data['skip_rate']
    avg_rating_given        = data['avg_rating_given']
    logins_per_week         = data['logins_per_week']
    subscription_type       = SUBSCRIPTION_MAP[data['subscription_type']]
    content_diversity_score = data['content_diversity_score']
    days_since_last_login   = data['days_since_last_login']
    notifications_clicked   = data['notifications_clicked']
    watch_streak_days       = data['watch_streak_days']

    # Engineered features (same as Milestone 2)
    engagement_ratio     = weekly_watch_hours / (skip_rate + 0.01)
    activity_score       = logins_per_week * 0.5 + watch_streak_days * 0.5
    premium_watch        = subscription_type * weekly_watch_hours
    inactive_flag        = 1 if days_since_last_login > 14 else 0
    notification_rate    = notifications_clicked / (logins_per_week + 1)

    return np.array([[
        age, weekly_watch_hours, skip_rate, avg_rating_given,
        logins_per_week, subscription_type, content_diversity_score,
        days_since_last_login, notifications_clicked, watch_streak_days,
        engagement_ratio, activity_score, premium_watch,
        inactive_flag, notification_rate
    ]])


# ══════════════════════════════════════════════════════════════
# ENDPOINT 1: Health Check
# ══════════════════════════════════════════════════════════════
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'message': 'Streaming Engagement Prediction API is running'
    }), 200


# ══════════════════════════════════════════════════════════════
# ENDPOINT 2: Predict Engagement Level
# ══════════════════════════════════════════════════════════════
@app.route('/predict', methods=['POST'])
def predict():
    # Check content type
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 415

    data = request.get_json()

    # Validate
    errors = validate_input(data)
    if errors:
        return jsonify({'error': 'Validation failed', 'details': errors}), 400

    # Build feature vector & predict
    features    = build_feature_vector(data)
    prediction  = model.predict(features)[0]
    proba       = model.predict_proba(features)[0]
    label       = LABEL_MAP[prediction]
    confidence  = round(float(proba[prediction]) * 100, 2)

    return jsonify({
        'prediction':      label,
        'confidence_pct':  confidence,
        'probabilities': {
            'Low':    round(float(proba[0]) * 100, 2),
            'Medium': round(float(proba[1]) * 100, 2),
            'High':   round(float(proba[2]) * 100, 2),
        },
        'input_received': data
    }), 200


# ══════════════════════════════════════════════════════════════
# ENDPOINT 3: Model Info
# ══════════════════════════════════════════════════════════════
@app.route('/model-info', methods=['GET'])
def model_info():
    return jsonify({
        'model_type':   'Random Forest Classifier (Tuned)',
        'n_estimators': model.n_estimators,
        'n_features':   model.n_features_in_,
        'classes':      ['Low', 'Medium', 'High'],
        'task':         'Multi-class Classification',
        'domain':       'Video Streaming Engagement Prediction'
    }), 200


if __name__ == '__main__':
    print("🚀 Starting Streaming Engagement Prediction API...")
    print("   POST /predict    → Get engagement prediction")
    print("   GET  /health     → Health check")
    print("   GET  /model-info → Model details")
    app.run(debug=True, port=5000)