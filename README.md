# Sarcasm Detection API

A Flask + MongoDB API for detecting sarcasm in text. The app uses a pretrained
RoBERTa sentiment model via Hugging Face and combines it with rule-based sarcasm
heuristics to produce a final sarcasm score and label.

## Features
- JWT-based auth (signup/login)
- Single and batch sarcasm prediction
- Prediction history and per-item lookup
- Feedback collection to evaluate predictions
- Simple React pages served by Flask

## Tech Stack
- Python (Flask)
- MongoDB (PyMongo)
- JWT auth (flask-jwt-extended)
- Hugging Face inference API
- React (static JSX pages)

## Setup
1. Create and activate a virtual environment
   - `python -m venv .venv`
   - `.\.venv\Scripts\activate`
2. Install dependencies
   - `pip install -r requirements.txt`
3. Create your `.env` file
   - Copy `.env.example` to `.env` and fill in values
4. Start the server
   - `python app.py`

The API will be available at `http://localhost:5000` in development.

## Environment Variables
- `HUGGINGFACE_API_TOKEN` or `HF_API_TOKEN`: Hugging Face access token
- `JWT_SECRET_KEY`: secret used to sign JWTs
- `MONGO_URI`: MongoDB connection string
- `FLASK_DEBUG`: set `true` or `false` (default is `false`)
- `BATCH_PREDICT_WORKERS`: max threads for batch prediction (default `12`)

## API Endpoints
- `POST /api/signup` (public)
- `POST /api/login` (public)
- `GET /api/user/profile` (JWT required)
- `POST /api/predict` (JWT required)
- `POST /api/predict/batch` (JWT required)
- `GET /api/history` (JWT required)
- `GET /api/prediction/<prediction_id>` (JWT required)
- `POST /api/feedback` (JWT required)

## Prediction Flow (High Level)
1. The app validates input and filters short/greeting-only text.
2. The Hugging Face sentiment model returns negative/neutral/positive scores.
3. Rule-based sarcasm heuristics generate an additional sarcasm signal.
4. Both signals are combined into a final sarcasm score and label.
5. Results are stored in MongoDB and returned to the client.

## Notes
- The pretrained model is used as-is. This project does not fine-tune it.
- Tuning happens in the rule-based sarcasm layer and thresholds.
