import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cybersec.db')


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()

    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
        CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            url TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL,
            phishing_probability REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_history_user_id ON history(user_id);
        CREATE INDEX IF NOT EXISTS idx_history_timestamp ON history(timestamp);

        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            accuracy REAL,
            precision_score REAL,
            recall_score REAL,
            f1_score REAL,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')

    conn.commit()
    conn.close()
    print("Database initialized successfully.")


def save_model_metrics(results: dict):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM model_metrics")
    for model_name, metrics in results.items():
        cursor.execute('''
            INSERT INTO model_metrics (model_name, accuracy, precision_score, recall_score, f1_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (model_name, metrics['accuracy'], metrics['precision'],
              metrics['recall'], metrics['f1_score']))
    conn.commit()
    conn.close()


def get_model_metrics():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM model_metrics ORDER BY accuracy DESC")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def add_scan_history(user_id: int, url: str, result: str, confidence: float, phishing_prob: float):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO history (user_id, url, result, confidence, phishing_probability)
        VALUES (?, ?, ?, ?, ?)
    ''', (user_id, url, result, confidence, phishing_prob))
    conn.commit()
    conn.close()


def get_user_history(user_id: int, limit: int = 20):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM history WHERE user_id = ?
        ORDER BY timestamp DESC LIMIT ?
    ''', (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_dashboard_stats(user_id: int = None):
    conn = get_db()
    cursor = conn.cursor()

    if user_id:
        cursor.execute("SELECT COUNT(*) as total FROM history WHERE user_id=?", (user_id,))
        total = cursor.fetchone()['total']
        cursor.execute("SELECT COUNT(*) as count FROM history WHERE user_id=? AND result='Phishing'", (user_id,))
        phishing = cursor.fetchone()['count']
    else:
        cursor.execute("SELECT COUNT(*) as total FROM history")
        total = cursor.fetchone()['total']
        cursor.execute("SELECT COUNT(*) as count FROM history WHERE result='Phishing'")
        phishing = cursor.fetchone()['count']

    legitimate = total - phishing
    conn.close()
    return {'total': total, 'phishing': phishing, 'legitimate': legitimate}
