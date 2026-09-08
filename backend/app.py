import os
from datetime import datetime

import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from transformers import AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()

db = SQLAlchemy()

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    label = db.Column(db.String(20), nullable=False)
    ai_probability = db.Column(db.Float, nullable=False)
    human_probability = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

MODEL_NAME = os.getenv("MODEL_NAME", "roberta-base-openai-detector")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "12000"))
tokenizer = None
model = None


def load_detector():
    global tokenizer, model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        model.eval()


def classify_text(text: str):
    load_detector()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    # The OpenAI RoBERTa detector uses labels 0=human, 1=AI.
    human_probability = float(probabilities[0])
    ai_probability = float(probabilities[1])
    label = "AI" if ai_probability >= human_probability else "Human"
    return label, ai_probability, human_probability


app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///detector.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)
CORS(app, resources={r"/api/*": {"origins": os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")}})

with app.app_context():
    db.create_all()


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.post("/api/detect")
def detect():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({"error": "Text is required."}), 400
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"Text must be {MAX_TEXT_LENGTH} characters or fewer."}), 400

    try:
        label, ai_probability, human_probability = classify_text(text)
        record = Detection(
            text=text,
            label=label,
            ai_probability=ai_probability,
            human_probability=human_probability,
        )
        db.session.add(record)
        db.session.commit()
        return jsonify({
            "id": record.id,
            "label": label,
            "ai_probability": round(ai_probability * 100, 2),
            "human_probability": round(human_probability * 100, 2),
            "created_at": record.created_at.isoformat(),
        })
    except Exception as exc:
        db.session.rollback()
        return jsonify({"error": "Detection failed", "details": str(exc)}), 500


@app.get("/api/history")
def history():
    records = Detection.query.order_by(Detection.created_at.desc()).limit(50).all()
    return jsonify([
        {
            "id": r.id,
            "text": r.text[:160],
            "label": r.label,
            "ai_probability": round(r.ai_probability * 100, 2),
            "human_probability": round(r.human_probability * 100, 2),
            "created_at": r.created_at.isoformat(),
        }
        for r in records
    ])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
