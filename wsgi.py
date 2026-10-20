"""WSGI entry point.  gunicorn wsgi:app   |   python wsgi.py (development server)"""
import os

from app.main import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), threaded=True)
