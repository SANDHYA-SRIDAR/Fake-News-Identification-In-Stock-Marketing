from flask import Flask, request, render_template, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
import pickle
import re
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change for production

# SQLite DB config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///news_analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ===========================
# Database Model
# ===========================

class NewsAnalysis(db.Model):   
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    url = db.Column(db.String(500), nullable=True)
    content = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)  # 'Real' or 'Fake'
    confidence = db.Column(db.Float, nullable=True)
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
# Routes
# ===========================

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    confidence = None

    if request.method == 'POST' and 'content' in request.form:
        title = request.form.get('title', '').strip()
        url = request.form.get('url', '').strip()
        content = request.form.get('content', '').strip()

        if not content:
            flash("News content is required.", "error")
            return render_template(
                'home.html',
                recent_analyses=get_recent_analyses(),
                stats=get_statistics(),
                chart_data=get_chart_data()
            )

        cleaned_text = cleanup(content)
        vect_text = vectorizer.transform([cleaned_text])
        pred = model.predict(vect_text)[0]

        try:
            pred_proba = model.predict_proba(vect_text)[0]
            confidence = max(pred_proba) * 100
        except Exception:
            confidence = None

        prediction = "Real" if pred == 1 else "Fake"

        new_analysis = NewsAnalysis(
            title=title if title else "Untitled",
            url=url if url else "",
            content=content,
            prediction=prediction,
            confidence=confidence
        )
        db.session.add(new_analysis)
        db.session.commit()

    return render_template(
        'home.html',
        prediction=prediction,
        confidence=confidence,
        recent_analyses=get_recent_analyses(),
        stats=get_statistics(),
        chart_data=get_chart_data()
    )

@app.route('/chart-data')
def chart_data_route():
    return jsonify(get_chart_data())

# ===========================
# Helper Functions
# ===========================

def get_recent_analyses():
    return NewsAnalysis.query.order_by(NewsAnalysis.timestamp.desc()).limit(5).all()

def get_statistics():
    analyses = NewsAnalysis.query.all()
    total = len(analyses)
    fake = sum(1 for a in analyses if a.prediction == 'Fake')
    verified = total - fake
    fake_rate = (fake / total * 100) if total > 0 else 0

    return {
        'total': total,
        'verified': verified,
        'fake': fake,
        'fake_rate': round(fake_rate, 1)
    }

def get_chart_data():
    analyses = NewsAnalysis.query.all()
    
    # Prepare data for pie chart
    fake_count = sum(1 for a in analyses if a.prediction == 'Fake')
    real_count = len(analyses) - fake_count
    
    # Prepare data for bar chart (confidence distribution)
    confidence_ranges = defaultdict(int)
    for analysis in analyses:
        if analysis.confidence is not None:
            if analysis.confidence < 20:
                confidence_ranges['0-20%'] += 1
            elif analysis.confidence < 40:
                confidence_ranges['20-40%'] += 1
            elif analysis.confidence < 60:
                confidence_ranges['40-60%'] += 1
            elif analysis.confidence < 80:
                confidence_ranges['60-80%'] += 1
            else:
                confidence_ranges['80-100%'] += 1
    
    # Prepare data for line chart (prediction trend over time)
    time_data = defaultdict(lambda: {'real': 0, 'fake': 0})
    for analysis in analyses:
        date_key = analysis.timestamp.strftime('%Y-%m-%d')
        if analysis.prediction == 'Real':
            time_data[date_key]['real'] += 1
        else:
            time_data[date_key]['fake'] += 1
    
    # Sort dates and prepare data
    sorted_dates = sorted(time_data.keys())
    real_counts = [time_data[date]['real'] for date in sorted_dates]
    fake_counts = [time_data[date]['fake'] for date in sorted_dates]
    
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
            'real': real_counts,
            'fake': fake_counts
        }
    }

# ===========================
# Run App
# ===========================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create DB table if not exist
    app.run(debug=True)