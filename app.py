import os
import sys
import json
import sqlite3
from datetime import timedelta

from flask import Flask, request, jsonify, session, render_template, redirect, url_for
from flask_bcrypt import Bcrypt
from flask_jwt_extended import (
    JWTManager, create_access_token, jwt_required,
    get_jwt_identity, verify_jwt_in_request
)
from functools import wraps

sys.path.insert(0, os.path.dirname(__file__))
from utils.database import (
    init_db, get_db, add_scan_history, get_user_history,
    get_dashboard_stats, get_model_metrics, save_model_metrics
)
from ml.model_trainer import predict_url, train_models, FEATURE_NAMES

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'cybersec-super-secret-2024')
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'jwt-cybersec-secret-2024')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
app.config['JWT_TOKEN_LOCATION'] = ['headers', 'cookies']
app.config['JWT_COOKIE_SECURE'] = False
app.config['JWT_COOKIE_CSRF_PROTECT'] = False

bcrypt = Bcrypt(app)
jwt = JWTManager(app)

# ─── Helpers ───────────────────────────────────────────────────────────────────

def get_current_user_from_cookie():
    """Try to get current user from JWT cookie or header."""
    try:
        verify_jwt_in_request(optional=True)
        identity = get_jwt_identity()
        return identity
    except Exception:
        return None


def login_required_web(f):
    """Decorator for web routes – redirects to login if not authenticated."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('access_token_cookie')
        if not token:
            return redirect(url_for('login_page'))
        try:
            verify_jwt_in_request(locations=['cookies'])
            return f(*args, **kwargs)
        except Exception:
            return redirect(url_for('login_page'))
    return decorated


def validate_url(url: str) -> bool:
    import re
    pattern = re.compile(
        r'^(https?://)?'
        r'([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)*'
        r'[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?'
        r'(\.[a-zA-Z]{2,})'
    )
    return bool(pattern.match(url)) or url.startswith('http')


# ─── Page Routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    user = get_current_user_from_cookie()
    if user:
        return redirect(url_for('dashboard_page'))
    return redirect(url_for('login_page'))


@app.route('/login')
def login_page():
    return render_template('login.html')


@app.route('/register')
def register_page():
    return render_template('register.html')


@app.route('/dashboard')
@login_required_web
def dashboard_page():
    return render_template('dashboard.html')


@app.route('/scan')
@login_required_web
def scan_page():
    return render_template('scan.html')


# ─── Auth API ──────────────────────────────────────────────────────────────────

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400

        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')

        if not username or not email or not password:
            return jsonify({'success': False, 'message': 'All fields are required'}), 400
        if len(username) < 3:
            return jsonify({'success': False, 'message': 'Username must be at least 3 characters'}), 400
        if len(password) < 6:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters'}), 400
        if '@' not in email or '.' not in email:
            return jsonify({'success': False, 'message': 'Invalid email address'}), 400

        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        conn = get_db()
        try:
            conn.execute(
                'INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                (username, email, password_hash)
            )
            conn.commit()
        except sqlite3.IntegrityError as e:
            conn.close()
            if 'username' in str(e):
                return jsonify({'success': False, 'message': 'Username already taken'}), 409
            return jsonify({'success': False, 'message': 'Email already registered'}), 409
        finally:
            conn.close()

        return jsonify({'success': True, 'message': 'Account created successfully!'}), 201

    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400

        username = data.get('username', '').strip()
        password = data.get('password', '')

        if not username or not password:
            return jsonify({'success': False, 'message': 'Username and password required'}), 400

        conn = get_db()
        user = conn.execute(
            'SELECT * FROM users WHERE username = ? OR email = ?',
            (username, username)
        ).fetchone()
        conn.close()

        if not user or not bcrypt.check_password_hash(user['password_hash'], password):
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401

        access_token = create_access_token(identity=str(user['id']))
        response = jsonify({
            'success': True,
            'message': 'Login successful',
            'token': access_token,
            'user': {'id': user['id'], 'username': user['username'], 'email': user['email']}
        })
        # Set JWT in cookie for web pages
        from flask_jwt_extended import set_access_cookies
        set_access_cookies(response, access_token)
        return response, 200

    except Exception as e:
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500


@app.route('/api/logout', methods=['POST'])
def logout():
    from flask_jwt_extended import unset_jwt_cookies
    response = jsonify({'success': True, 'message': 'Logged out'})
    unset_jwt_cookies(response)
    return response, 200


# ─── Predict API ───────────────────────────────────────────────────────────────

@app.route('/api/predict', methods=['POST'])
@jwt_required()
def predict():
    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({'success': False, 'message': 'URL is required'}), 400

        url = data['url'].strip()
        if not url:
            return jsonify({'success': False, 'message': 'URL cannot be empty'}), 400
        if not validate_url(url):
            return jsonify({'success': False, 'message': 'Please enter a valid URL'}), 400
        if len(url) > 2000:
            return jsonify({'success': False, 'message': 'URL is too long'}), 400

        # Add http:// prefix if missing
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        result = predict_url(url)
        user_id = int(get_jwt_identity())
        add_scan_history(
            user_id, url, result['label'],
            result['confidence'], result['phishing_probability']
        )

        return jsonify({'success': True, 'data': result}), 200

    except Exception as e:
        return jsonify({'success': False, 'message': f'Prediction error: {str(e)}'}), 500


# ─── History API ───────────────────────────────────────────────────────────────

@app.route('/api/history', methods=['GET'])
@jwt_required()
def history():
    try:
        user_id = int(get_jwt_identity())
        limit = min(int(request.args.get('limit', 20)), 100)
        records = get_user_history(user_id, limit)
        return jsonify({'success': True, 'data': records}), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─── Stats API ─────────────────────────────────────────────────────────────────

@app.route('/api/stats', methods=['GET'])
@jwt_required()
def stats():
    try:
        user_id = int(get_jwt_identity())
        user_stats = get_dashboard_stats(user_id)
        global_stats = get_dashboard_stats()
        model_metrics = get_model_metrics()
        recent = get_user_history(user_id, 5)
        return jsonify({
            'success': True,
            'data': {
                'user_stats': user_stats,
                'global_stats': global_stats,
                'model_metrics': model_metrics,
                'recent_scans': recent
            }
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─── Model Info API ────────────────────────────────────────────────────────────

@app.route('/api/models', methods=['GET'])
@jwt_required()
def model_info():
    try:
        metrics = get_model_metrics()
        return jsonify({'success': True, 'data': metrics}), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/retrain', methods=['POST'])
@jwt_required()
def retrain():
    try:
        results, best = train_models()
        save_model_metrics(results)
        return jsonify({
            'success': True,
            'message': f'Models retrained. Best model: {best}',
            'data': results
        }), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


# ─── Error Handlers ────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'message': 'Endpoint not found'}), 404
    return redirect(url_for('login_page'))


@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'message': 'Internal server error'}), 500


@jwt.unauthorized_loader
def unauthorized_response(callback):
    if request.path.startswith('/api/'):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    return redirect(url_for('login_page'))


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    init_db()
    # Train models on first run
    model_path = os.path.join(os.path.dirname(__file__), 'ml', 'best_model.pkl')
    if not os.path.exists(model_path):
        print("Training ML models for the first time...")
        results, best = train_models()
        save_model_metrics(results)
        print("Training complete!")
    app.run(debug=True, host='0.0.0.0', port=5000)
