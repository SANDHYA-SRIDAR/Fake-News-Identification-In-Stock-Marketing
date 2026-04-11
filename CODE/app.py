from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pickle
import re

import requests
from flask import Flask, flash, g, jsonify, redirect, render_template, request, session, url_for
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash
from functools import wraps


BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__, instance_relative_config=True)
app.config["SECRET_KEY"] = "change-this-secret-key"
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    analyses = db.relationship("NewsAnalysis", backref="user", lazy=True)


class NewsAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    url = db.Column(db.String(500))
    content = db.Column(db.Text, nullable=False)
    prediction = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)


with open(BASE_DIR / "model.pickle", "rb") as model_file:
    model = pickle.load(model_file)

with open(BASE_DIR / "vector.pickle", "rb") as vector_file:
    vectorizer = pickle.load(vector_file)


cleanup_re = re.compile("[^a-z]+")
NEWS_API_KEY = "a158542351194272b57d9a5638e89986"
NEWS_API_URL = "https://newsapi.org/v2/everything"


@app.before_request
def load_logged_in_user():
    user_id = session.get("user_id")
    g.current_user = db.session.get(User, user_id) if user_id else None


@app.context_processor
def inject_current_user():
    return {"current_user": g.get("current_user")}


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.current_user is None:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped_view


def cleanup(sentence):
    return cleanup_re.sub(" ", str(sentence).lower()).strip()


def verify_password(user, password):
    stored_password = user.password or ""

    if stored_password.startswith("pbkdf2:") or stored_password.startswith("scrypt:"):
        return check_password_hash(stored_password, password)

    if stored_password == password:
        user.password = generate_password_hash(password)
        db.session.commit()
        return True

    return False


def build_prediction(content):
    cleaned_text = cleanup(content)
    vect_text = vectorizer.transform([cleaned_text])
    pred = model.predict(vect_text)[0]

    try:
        confidence = float(max(model.predict_proba(vect_text)[0]) * 100)
    except Exception:
        confidence = None

    prediction = "Real" if pred == 1 else "Fake"
    return prediction, confidence


def get_user_analyses():
    return NewsAnalysis.query.filter_by(user_id=g.current_user.id)


def get_recent_analyses():
    return get_user_analyses().order_by(NewsAnalysis.timestamp.desc()).limit(5).all()


def get_statistics():
    analyses = get_user_analyses().all()
    total = len(analyses)
    fake = sum(1 for analysis in analyses if analysis.prediction == "Fake")

    return {
        "total": total,
        "verified": total - fake,
        "fake": fake,
        "fake_rate": round((fake / total) * 100, 1) if total else 0,
    }


def get_chart_data():
    analyses = get_user_analyses().all()
    fake_count = sum(1 for analysis in analyses if analysis.prediction == "Fake")
    real_count = len(analyses) - fake_count

    confidence_ranges = {
        "0-20%": 0,
        "20-40%": 0,
        "40-60%": 0,
        "60-80%": 0,
        "80-100%": 0,
    }

    for analysis in analyses:
        if analysis.confidence is None:
            continue
        if analysis.confidence < 20:
            confidence_ranges["0-20%"] += 1
        elif analysis.confidence < 40:
            confidence_ranges["20-40%"] += 1
        elif analysis.confidence < 60:
            confidence_ranges["40-60%"] += 1
        elif analysis.confidence < 80:
            confidence_ranges["60-80%"] += 1
        else:
            confidence_ranges["80-100%"] += 1

    time_data = defaultdict(lambda: {"real": 0, "fake": 0})
    for analysis in analyses:
        date_key = analysis.timestamp.strftime("%Y-%m-%d")
        if analysis.prediction == "Real":
            time_data[date_key]["real"] += 1
        else:
            time_data[date_key]["fake"] += 1

    sorted_dates = sorted(time_data.keys())

    return {
        "pie": {"labels": ["Real", "Fake"], "data": [real_count, fake_count]},
        "bar": {"labels": list(confidence_ranges.keys()), "data": list(confidence_ranges.values())},
        "line": {
            "labels": sorted_dates,
            "real": [time_data[date]["real"] for date in sorted_dates],
            "fake": [time_data[date]["fake"] for date in sorted_dates],
        },
    }


def render_home(prediction=None, confidence=None):
    return render_template(
        "home.html",
        prediction=prediction,
        confidence=confidence,
        recent_analyses=get_recent_analyses(),
        stats=get_statistics(),
        chart_data=get_chart_data(),
    )


@app.route("/")
def index():
    if g.current_user is not None:
        return redirect(url_for("home"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.current_user is not None:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password:
            flash("Username and password are required.", "error")
        elif password != confirm_password:
            flash("Passwords do not match.", "error")
        elif User.query.filter_by(username=username).first():
            flash("Username already exists. Please log in.", "error")
        else:
            new_user = User(
                username=username,
                password=generate_password_hash(password),
            )
            db.session.add(new_user)
            db.session.commit()
            session.clear()
            session["user_id"] = new_user.id
            flash("Account created successfully.", "success")
            return redirect(url_for("home"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.current_user is not None:
        return redirect(url_for("home"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()

        if user and verify_password(user, password):
            session.clear()
            session["user_id"] = user.id
            flash("Logged in successfully.", "success")
            return redirect(url_for("home"))

        flash("Invalid username or password.", "error")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("You have been signed out.", "success")
    return redirect(url_for("login"))


@app.route("/home", methods=["GET", "POST"])
@login_required
def home():
    prediction = None
    confidence = None

    if request.method == "POST" and "content" in request.form:
        content = request.form.get("content", "").strip()
        title = request.form.get("title", "").strip() or "Manual Entry"
        url = request.form.get("url", "").strip()

        if not content:
            flash("Please enter news content to analyze.", "error")
        else:
            prediction, confidence = build_prediction(content)
            db.session.add(
                NewsAnalysis(
                    title=title,
                    url=url,
                    content=content,
                    prediction=prediction,
                    confidence=confidence,
                    user_id=g.current_user.id,
                )
            )
            db.session.commit()
            flash("News analyzed and saved successfully.", "success")

    return render_home(prediction=prediction, confidence=confidence)


@app.route("/api/analyze", methods=["POST"])
@login_required
def api_analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Text is required"}), 400

    prediction, confidence = build_prediction(text)
    return jsonify(
        {
            "prediction": prediction,
            "confidence": round(confidence, 2) if confidence is not None else None,
        }
    )


@app.route("/api/fetch-live-news", methods=["POST"])
@login_required
def fetch_live_news():
    params = {
        "q": "stock market OR shares OR finance OR trading",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 3,
        "apiKey": NEWS_API_KEY,
    }

    response = requests.get(NEWS_API_URL, params=params, timeout=15)
    data = response.json()

    if "articles" not in data:
        return jsonify({"error": "Failed to fetch news"}), 500

    results = []
    for article in data["articles"]:
        content = article.get("description") or article.get("content")
        if not content:
            continue

        prediction, confidence = build_prediction(content)
        news = NewsAnalysis(
            title=article.get("title") or "Live Stock News",
            url=article.get("url") or "",
            content=content,
            prediction=prediction,
            confidence=confidence,
            user_id=g.current_user.id,
        )
        db.session.add(news)
        results.append(
            {
                "title": news.title,
                "prediction": prediction,
                "confidence": round(confidence, 2) if confidence is not None else None,
                "url": news.url,
            }
        )

    db.session.commit()
    return jsonify(results)


with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True)
