# CyberShield AI – Phishing Defense System

An AI-powered cybersecurity platform for phishing URL detection using Machine Learning.

---

## Features

- Phishing URL detection (Random Forest, Decision Tree, Logistic Regression)
- User authentication with bcrypt password hashing
- JWT-protected REST API
- Interactive dashboard with Chart.js analytics
- Full scan history per user
- 15-feature URL analysis
- Responsive Bootstrap UI (mobile-friendly)
- SQLite database

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python run.py
```

Visit: **http://localhost:5000**

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/register` | No | Register new user |
| POST | `/api/login` | No | Login, returns JWT |
| POST | `/api/logout` | No | Clear JWT cookie |
| POST | `/api/predict` | JWT | Predict URL phishing |
| GET | `/api/history` | JWT | User scan history |
| GET | `/api/stats` | JWT | Dashboard statistics |
| GET | `/api/models` | JWT | ML model metrics |
| POST | `/api/retrain` | JWT | Retrain all models |

---

## Project Structure

```
cybersecurity_app/
├── app.py              # Flask app & API routes
├── run.py              # Startup script
├── requirements.txt
├── cybersec.db         # SQLite database (auto-created)
├── ml/
│   ├── model_trainer.py    # ML training & feature extraction
│   ├── best_model.pkl      # Saved best model (auto-created)
│   └── scaler.pkl          # Feature scaler (auto-created)
├── utils/
│   └── database.py     # DB helpers
└── templates/
    ├── login.html
    ├── register.html
    ├── dashboard.html
    └── scan.html
```

---

## URL Features Extracted (15 total)

1. URL Length
2. Number of Dots
3. Presence of @ Symbol
4. HTTPS Usage
5. Number of Hyphens
6. Number of Digits
7. Number of Special Characters
8. Number of Subdomains
9. URL Depth (slash count)
10. Suspicious Keywords
11. IP Address in URL
12. URL Shortener Detection
13. Double Slash Redirect
14. Hex Encoding
15. Query String Length

---

## Security

- Passwords hashed with **bcrypt**
- Authentication via **JWT tokens** (HTTP-only cookie + Bearer)
- SQL injection prevention via **parameterized queries**
- Input validation on all endpoints
- Protected routes require valid JWT

---

## Deployment (Render/Railway)

```bash
# Use gunicorn for production
gunicorn -w 2 -b 0.0.0.0:$PORT app:app
```

Set environment variables:
- `SECRET_KEY` – Flask secret key
- `JWT_SECRET_KEY` – JWT signing key
