from flask import Flask, request, render_template, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
import pickle
import re
import requests
from datetime import datetime
from collections import defaultdict

# =========================
# App Config
# =========================
app = Flask(__name__)
app.secret_key = "your_secret_key"

app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///news_analysis.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ===========================
# Database Model
# ===========================

class NewsAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    url = db.Column(db.String(500))
    content = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ===========================
# Load ML Model and Vectorizer
# ===========================

with open('model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('vector.pickle', 'rb') as f:
    vectorizer = pickle.load(f)

cleanup_re = re.compile('[^a-z]+')

def cleanup(sentence):
    sentence = str(sentence).lower()
    sentence = cleanup_re.sub(' ', sentence).strip()
    return sentence

# ===========================
# Live Stock News Config
# ===========================

NEWS_API_KEY = "a158542351194272b57d9a5638e89986"
NEWS_API_URL = "https://newsapi.org/v2/everything"

# ===========================
# UI Route
# ===========================

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    confidence = None

    if request.method == 'POST' and 'content' in request.form:
        content = request.form.get('content', '').strip()
        title = request.form.get('title', 'Manual Entry')
        url = request.form.get('url', '')

        cleaned_text = cleanup(content)
        vect_text = vectorizer.transform([cleaned_text])
        pred = model.predict(vect_text)[0]

        try:
            confidence = max(model.predict_proba(vect_text)[0]) * 100
        except:
            confidence = None

        prediction = "Real" if pred == 1 else "Fake"

        db.session.add(NewsAnalysis(
            title=title,
            url=url,
            content=content,
            prediction=prediction,
            confidence=confidence
        ))
        db.session.commit()

    return render_template(
        'home.html',
        prediction=prediction,
        confidence=confidence,
        recent_analyses=get_recent_analyses(),
        stats=get_statistics(),
        chart_data=get_chart_data()
    )

# ===========================
# REAL-TIME ML API
# ===========================

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    data = request.get_json()

    if not data or 'text' not in data:
        return jsonify({"error": "Text is required"}), 400

    cleaned_text = cleanup(data['text'])
    vect_text = vectorizer.transform([cleaned_text])
    pred = model.predict(vect_text)[0]

    try:
        confidence = max(model.predict_proba(vect_text)[0]) * 100
    except:
        confidence = None

    return jsonify({
        "prediction": "Real" if pred == 1 else "Fake",
        "confidence": round(confidence, 2) if confidence else None
    })

# ===========================
# LIVE STOCK NEWS (ON-DEMAND)
# ===========================

@app.route('/api/fetch-live-news', methods=['POST'])
def fetch_live_news():
    params = {
        "q": "stock market OR shares OR finance OR trading",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 3,  # TOP 3 RECENT NEWS
        "apiKey": NEWS_API_KEY
    }

    response = requests.get(NEWS_API_URL, params=params)
    data = response.json()

    if "articles" not in data:
        return jsonify({"error": "Failed to fetch news"}), 500

    results = []

    for article in data["articles"]:
        content = article.get("description") or article.get("content")
        if not content:
            continue

        cleaned_text = cleanup(content)
        vect_text = vectorizer.transform([cleaned_text])
        pred = model.predict(vect_text)[0]

        try:
            confidence = max(model.predict_proba(vect_text)[0]) * 100
        except:
            confidence = None

        prediction = "Real" if pred == 1 else "Fake"

        news = NewsAnalysis(
            title=article.get("title", "Live Stock News"),
            url=article.get("url", ""),
            content=content,
            prediction=prediction,
            confidence=confidence
        )

        db.session.add(news)

        results.append({
            "title": news.title,
            "prediction": prediction,
            "confidence": round(confidence, 2) if confidence else None,
            "url": news.url
        })

    db.session.commit()
    return jsonify(results)

# ===========================
# Helper Functions
# ===========================

def get_recent_analyses():
    return NewsAnalysis.query.order_by(NewsAnalysis.timestamp.desc()).limit(5).all()

def get_statistics():
    analyses = NewsAnalysis.query.all()
    total = len(analyses)
    fake = sum(1 for a in analyses if a.prediction == 'Fake')
    return {
        "total": total,
        "verified": total - fake,
        "fake": fake,
        "fake_rate": round((fake / total * 100), 1) if total else 0
    }

def get_chart_data():
    analyses = NewsAnalysis.query.all()

    fake_count = sum(1 for a in analyses if a.prediction == 'Fake')
    real_count = len(analyses) - fake_count

    confidence_ranges = {
        '0-20%': 0,
        '20-40%': 0,
        '40-60%': 0,
        '60-80%': 0,
        '80-100%': 0
    }

    for a in analyses:
        if a.confidence is not None:
            if a.confidence < 20:
                confidence_ranges['0-20%'] += 1
            elif a.confidence < 40:
                confidence_ranges['20-40%'] += 1
            elif a.confidence < 60:
                confidence_ranges['40-60%'] += 1
            elif a.confidence < 80:
                confidence_ranges['60-80%'] += 1
            else:
                confidence_ranges['80-100%'] += 1

    time_data = defaultdict(lambda: {'real': 0, 'fake': 0})

    for a in analyses:
        date_key = a.timestamp.strftime('%Y-%m-%d')
        if a.prediction == 'Real':
            time_data[date_key]['real'] += 1
        else:
            time_data[date_key]['fake'] += 1

    sorted_dates = sorted(time_data.keys())

    return {
        'pie': {
            'labels': ['Real', 'Fake'],
            'data': [real_count, fake_count]
        },
        'bar': {
            'labels': list(confidence_ranges.keys()),
            'data': list(confidence_ranges.values())
        },
        'line': {
            'labels': sorted_dates,
            'real': [time_data[d]['real'] for d in sorted_dates],
            'fake': [time_data[d]['fake'] for d in sorted_dates]
        }
    }

# ===========================
# Run App
# ===========================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
