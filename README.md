# AI vs Human Text Detector

A full-stack AI/Human text detection application using React, Flask, RoBERTa, and MySQL.

## Stack
- Frontend: React + Vite
- Backend: Python Flask
- ML: Hugging Face Transformers / RoBERTa OpenAI detector
- Database: MySQL

## Features
- Paste text and classify it as AI-generated or human-written
- Confidence score and probability breakdown
- Detection history stored in MySQL
- REST API between React and Flask
- Responsive modern UI

## Run locally

### Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
copy .env.example .env
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Set `DATABASE_URL` in `backend/.env`, for example:
`mysql+pymysql://root:password@localhost:3306/ai_text_detector`

The backend downloads the RoBERTa model on first startup/use. For production, pin/cache the model and configure a production WSGI server.
