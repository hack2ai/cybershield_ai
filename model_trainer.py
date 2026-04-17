import os
import re
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


def extract_features(url: str) -> list:
    """Extract features from a URL for phishing detection."""
    features = []

    # 1. URL length
    features.append(len(url))

    # 2. Number of dots
    features.append(url.count('.'))

    # 3. Presence of @ symbol
    features.append(1 if '@' in url else 0)

    # 4. HTTPS usage
    features.append(1 if url.startswith('https') else 0)

    # 5. Number of hyphens
    features.append(url.count('-'))

    # 6. Number of digits
    features.append(sum(c.isdigit() for c in url))

    # 7. Number of special characters
    features.append(len(re.findall(r'[!#$%^&*(),?":{}|<>]', url)))

    # 8. Number of subdomains
    try:
        domain = url.split('//')[-1].split('/')[0]
        features.append(len(domain.split('.')) - 2)
    except:
        features.append(0)

    # 9. URL depth (number of slashes)
    features.append(url.count('/'))

    # 10. Suspicious keywords
    suspicious_keywords = ['login', 'verify', 'update', 'secure', 'account',
                           'banking', 'confirm', 'password', 'paypal', 'ebay',
                           'microsoft', 'apple', 'google', 'phish', 'free', 'click']
    url_lower = url.lower()
    features.append(sum(1 for kw in suspicious_keywords if kw in url_lower))

    # 11. IP address in URL
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    features.append(1 if re.search(ip_pattern, url) else 0)

    # 12. URL shortener
    shorteners = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'short.link']
    features.append(1 if any(s in url_lower for s in shorteners) else 0)

    # 13. Double slash redirect
    features.append(1 if '//' in url[7:] else 0)

    # 14. Presence of hex encoding
    features.append(1 if '%' in url else 0)

    # 15. Query string length
    features.append(len(url.split('?')[-1]) if '?' in url else 0)

    return features


FEATURE_NAMES = [
    'url_length', 'num_dots', 'has_at', 'is_https', 'num_hyphens',
    'num_digits', 'num_special_chars', 'num_subdomains', 'url_depth',
    'suspicious_keywords', 'has_ip', 'is_shortener', 'double_slash',
    'has_hex', 'query_length'
]


def generate_synthetic_dataset(n_samples=2000):
    """Generate a synthetic dataset for phishing detection training."""
    np.random.seed(42)
    data = []

    # Generate legitimate URL features
    for _ in range(n_samples // 2):
        url_len = np.random.randint(15, 60)
        features = {
            'url_length': url_len,
            'num_dots': np.random.randint(1, 4),
            'has_at': 0,
            'is_https': np.random.choice([0, 1], p=[0.1, 0.9]),
            'num_hyphens': np.random.randint(0, 2),
            'num_digits': np.random.randint(0, 3),
            'num_special_chars': np.random.randint(0, 3),
            'num_subdomains': np.random.randint(0, 2),
            'url_depth': np.random.randint(1, 4),
            'suspicious_keywords': np.random.randint(0, 1),
            'has_ip': 0,
            'is_shortener': 0,
            'double_slash': 0,
            'has_hex': np.random.choice([0, 1], p=[0.9, 0.1]),
            'query_length': np.random.randint(0, 30),
            'label': 0  # Legitimate
        }
        data.append(features)

    # Generate phishing URL features
    for _ in range(n_samples // 2):
        url_len = np.random.randint(50, 200)
        features = {
            'url_length': url_len,
            'num_dots': np.random.randint(3, 8),
            'has_at': np.random.choice([0, 1], p=[0.5, 0.5]),
            'is_https': np.random.choice([0, 1], p=[0.7, 0.3]),
            'num_hyphens': np.random.randint(2, 8),
            'num_digits': np.random.randint(3, 12),
            'num_special_chars': np.random.randint(3, 10),
            'num_subdomains': np.random.randint(2, 6),
            'url_depth': np.random.randint(4, 10),
            'suspicious_keywords': np.random.randint(1, 5),
            'has_ip': np.random.choice([0, 1], p=[0.5, 0.5]),
            'is_shortener': np.random.choice([0, 1], p=[0.6, 0.4]),
            'double_slash': np.random.choice([0, 1], p=[0.5, 0.5]),
            'has_hex': np.random.choice([0, 1], p=[0.4, 0.6]),
            'query_length': np.random.randint(20, 150),
            'label': 1  # Phishing
        }
        data.append(features)

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def train_models():
    """Train all ML models and return results."""
    print("Generating dataset...")
    df = generate_synthetic_dataset(2000)

    X = df[FEATURE_NAMES].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

    results = {}
    best_model_name = None
    best_accuracy = 0
    best_model = None

    for name, model in models.items():
        print(f"Training {name}...")
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name] = {
            'accuracy': round(acc * 100, 2),
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'f1_score': round(f1 * 100, 2)
        }

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    # Save best model and scaler
    model_dir = os.path.join(os.path.dirname(__file__))
    with open(os.path.join(model_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\nBest model: {best_model_name} (Accuracy: {best_accuracy*100:.2f}%)")
    return results, best_model_name


def load_model():
    """Load the saved best model and scaler."""
    model_dir = os.path.dirname(__file__)
    model_path = os.path.join(model_dir, 'best_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')

    if not os.path.exists(model_path):
        train_models()

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_url(url: str) -> dict:
    """Predict whether a URL is phishing or legitimate."""
    model, scaler = load_model()
    features = extract_features(url)
    features_array = np.array(features).reshape(1, -1)

    # Logistic Regression needs scaled input
    from sklearn.linear_model import LogisticRegression as LR
    if isinstance(model, LR):
        features_array = scaler.transform(features_array)

    prediction = model.predict(features_array)[0]
    proba = model.predict_proba(features_array)[0]

    return {
        'prediction': int(prediction),
        'label': 'Phishing' if prediction == 1 else 'Legitimate',
        'confidence': round(float(max(proba)) * 100, 2),
        'phishing_probability': round(float(proba[1]) * 100, 2),
        'features': dict(zip(FEATURE_NAMES, features))
    }


if __name__ == '__main__':
    results, best = train_models()
    print("\nModel Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}%")
