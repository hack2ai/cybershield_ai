#!/usr/bin/env python3
"""
CyberShield AI – Startup Script
Initializes database, trains models if needed, and starts the Flask server.
"""
import os
import sys

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("=" * 55)
    print("  CyberShield AI – Cybersecurity Defense System")
    print("=" * 55)

    # Initialize database
    print("\n[1/3] Initializing database...")
    from utils.database import init_db, save_model_metrics
    init_db()

    # Train models if not present
    model_path = os.path.join('ml', 'best_model.pkl')
    if not os.path.exists(model_path):
        print("[2/3] Training ML models (first run only)...")
        from ml.model_trainer import train_models
        results, best = train_models()
        save_model_metrics(results)
        print(f"      ✓ Best model: {best}")
    else:
        print("[2/3] ML models already trained ✓")

    print("[3/3] Starting Flask server...")
    print("\n" + "=" * 55)
    print("  App running at: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 55 + "\n")

    from app import app
    app.run(debug=False, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    main()
